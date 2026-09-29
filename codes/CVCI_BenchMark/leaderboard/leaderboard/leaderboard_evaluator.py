#!/usr/bin/env python
# Copyright (c) 2018-2019 Intel Corporation.
# authors: German Ros (german.ros@intel.com), Felipe Codevilla (felipe.alcm@gmail.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
CARLA Challenge Evaluator Routes

Provisional code to evaluate Autonomous Agents for the CARLA Autonomous Driving challenge
"""
from __future__ import print_function

import traceback
import argparse
from argparse import RawTextHelpFormatter
from distutils.version import LooseVersion
import importlib
import gc
import os
import pkg_resources
import sys
import carla
import signal

from srunner.scenariomanager.carla_data_provider import *
from srunner.scenariomanager.timer import GameTime
from srunner.scenariomanager.watchdog import Watchdog

from leaderboard.scenarios.scenario_manager import ScenarioManager
from leaderboard.scenarios.route_scenario import RouteScenario
from leaderboard.envs.sensor_interface import SensorConfigurationInvalid
from leaderboard.autoagents.agent_wrapper import AgentError, validate_sensor_configuration, TickRuntimeError
from leaderboard.utils.statistics_manager import StatisticsManager, FAILURE_MESSAGES
from leaderboard.utils.route_indexer import RouteIndexer
import atexit
import subprocess
import time
import random
import xml.etree.ElementTree as ET
from datetime import datetime

sensors_to_icons = {
    'sensor.camera.rgb':        'carla_camera',
    'sensor.lidar.ray_cast':    'carla_lidar',
    'sensor.other.radar':       'carla_radar',
    'sensor.other.gnss':        'carla_gnss',
    'sensor.other.imu':         'carla_imu',
    'sensor.opendrive_map':     'carla_opendrive_map',
    'sensor.speedometer':       'carla_speedometer'
}

import socket
import shlex
import select


class TrafficManagerProxy(object):
    """Text-command proxy for a helper that owns the native TM handle."""

    def __init__(self, process, timeout):
        self.process = process
        self.timeout = float(timeout)

    def _request(self, command):
        if self.process.poll() is not None:
            raise RuntimeError("TrafficManager helper is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
        ready, _, _ = select.select(
            [self.process.stdout], [], [], self.timeout
        )
        if not ready:
            raise RuntimeError("TrafficManager helper command timed out")
        response = self.process.stdout.readline().strip()
        if response != "ACK":
            raise RuntimeError(
                "TrafficManager helper rejected {!r}: {!r}".format(
                    command, response
                )
            )

    def set_random_device_seed(self, seed):
        self._request("seed {}".format(int(seed)))

    def set_synchronous_mode(self, enabled):
        if not enabled:
            self._request("async")

    def set_hybrid_physics_mode(self, enabled):
        # The async command changes both modes atomically. The evaluator calls
        # these two setters consecutively; the second one is intentionally a
        # no-op to avoid sending the same transition twice.
        return None

    def shut_down(self):
        self._request("shutdown")
        self.process.wait(timeout=self.timeout)

def find_free_port(starting_port):
    port = starting_port
    while True:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("localhost", port))
                return port
        except OSError:
            port += 1

def get_weather_id(weather_conditions):
    from xml.etree import ElementTree as ET
    tree = ET.parse('leaderboard/data/weather.xml')
    root = tree.getroot()
    def conditions_match(weather, conditions):
        for (key, value) in weather:
            if key == 'route_percentage' : continue
            if str(getattr(conditions, key))!= value:
                return False
        return True
    for case in root.findall('case'):
        weather = case[0].items()
        if conditions_match(weather, weather_conditions):
            return case.items()[0][1]
    return None

class LeaderboardEvaluator(object):
    """
    Main class of the Leaderboard. Everything is handled from here,
    from parsing the given files, to preparing the simulation, to running the route.
    """

    # Tunable parameters
    client_timeout = 300.0  # in seconds
    frame_rate = 20.0      # in Hz

    def __init__(self, args, statistics_manager):
        """
        Setup CARLA client and world
        Setup ScenarioManager
        """
        self.world = None
        self.manager = None
        self.sensors = None
        self.sensors_initialized = False
        self.sensor_icons = []
        self.agent_instance = None
        self.route_scenario = None

        self.statistics_manager = statistics_manager

        # Independent timeout settings
        self.client_timeout = args.client_timeout
        self.scenario_timeout = args.scenario_timeout
        self.agent_timeout = args.agent_timeout

        # This is the ROS1 bridge server instance. This is not encapsulated inside the ROS1 agent because the same
        # instance is used on all the routes (i.e., the server is not restarted between routes). This is done
        # to avoid reconnection issues between the server and the roslibpy client.
        self._ros1_server = None

        # Setup the simulation
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(
            args, self._first_route_town(args)
        )

        dist = pkg_resources.get_distribution("carla")
        if dist.version != 'leaderboard':
            if LooseVersion(dist.version) < LooseVersion('0.9.10'):
                raise ImportError("CARLA version 0.9.10.1 or newer required. CARLA version found: {}".format(dist))

        # Load agent
        module_name = os.path.basename(args.agent).split('.')[0]
        sys.path.insert(0, os.path.dirname(args.agent))
        self.module_agent = importlib.import_module(module_name)

        # Create the ScenarioManager
        self.manager = ScenarioManager(args.scenario_timeout, self.statistics_manager, args.debug)

        # Time control for summary purposes
        self._start_time = GameTime.get_time()
        self._end_time = None

        # Prepare the agent timer
        self._agent_watchdog = None
        signal.signal(signal.SIGINT, self._signal_handler)

        self._client_timed_out = False
        self._dead_carla_handles = []
        self._requires_hard_exit = False
        self._successful_route_world_loads = 0

    def _signal_handler(self, signum, frame):
        """
        Terminate scenario ticking when receiving a signal interrupt.
        Either the agent initialization watchdog is triggered, or the runtime one at scenario manager
        """
        if self._agent_watchdog and not self._agent_watchdog.get_status():
            raise RuntimeError("Timeout: Agent took longer than {}s to setup".format(self.agent_timeout))
        elif self.manager:
            self.manager.signal_handler(signum, frame)

    def __del__(self):
        """
        Cleanup and delete actors, ScenarioManager and CARLA world
        """
        if hasattr(self, 'manager') and self.manager:
            del self.manager
        if hasattr(self, 'world') and self.world:
            del self.world

    def _get_running_status(self):
        """
        returns:
           bool: False if watchdog exception occured, True otherwise
        """
        if self._agent_watchdog:
            return self._agent_watchdog.get_status()
        return False

    def _cleanup(self):
        """
        Remove and destroy all actors
        """
        try:
            CarlaDataProvider.cleanup()
        except Exception:
            print("\n\033[91mFailed to clean CarlaDataProvider:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

        if self._agent_watchdog:
            self._agent_watchdog.stop()

        try:
            if self.agent_instance:
                self.agent_instance.destroy()
                self.agent_instance = None
        except Exception as e:
            print("\n\033[91mFailed to stop the agent:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

        if self.route_scenario:
            self.route_scenario.remove_all_actors()
            self.route_scenario = None
            if self.statistics_manager:
                self.statistics_manager.remove_scenario()

        if self.manager:
            self._client_timed_out = not self.manager.get_running_status()
            self.manager.cleanup()

        # Scenario classes may create temporary native TrafficManager handles.
        # Reclaim them while CARLA is still responsive so no dead-server TM
        # destructor remains when the next map switch starts.
        if self._server_process_alive():
            gc.collect()

        # A dead CARLA server makes every RPC wait for client_timeout.  Do not
        # issue another RPC while handling an already confirmed server crash.
        if self.world and not self._client_timed_out and self._server_process_alive():
            try:
                alive_sensors = self.world.get_actors().filter('*sensor*')
                for sensor in alive_sensors:
                    sensor.stop()
                    sensor.destroy()
            except Exception:
                print("\n\033[91mFailed to destroy remaining sensors:", flush=True)
                print(f"\n{traceback.format_exc()}\033[0m", flush=True)

    def _server_process_alive(self):
        server = getattr(self, "server", None)
        return server is not None and server.poll() is None

    @staticmethod
    def _stop_server_process(server, grace_seconds=10.0):
        """Stop one evaluator-owned CARLA process group without touching peers."""
        if server is None:
            return
        process_group = server.pid

        def group_alive():
            try:
                os.killpg(process_group, 0)
                return True
            except ProcessLookupError:
                return False
            except PermissionError:
                return True

        if not group_alive():
            try:
                server.wait(timeout=0.1)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass
            return
        try:
            os.killpg(process_group, signal.SIGTERM)
            deadline = time.monotonic() + grace_seconds
            while group_alive() and time.monotonic() < deadline:
                time.sleep(0.1)
            if group_alive():
                os.killpg(process_group, signal.SIGKILL)
                kill_deadline = time.monotonic() + 5.0
                while group_alive() and time.monotonic() < kill_deadline:
                    time.sleep(0.1)
        except ProcessLookupError:
            pass
        finally:
            try:
                server.wait(timeout=5.0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass

    @staticmethod
    def _stop_traffic_manager_proxy(proxy, grace_seconds=3.0):
        if not isinstance(proxy, TrafficManagerProxy):
            return
        process = proxy.process
        if process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=grace_seconds)
        except (ProcessLookupError, ChildProcessError):
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3.0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                pass

    def _start_traffic_manager_proxy(self, args, timeout):
        helper = os.path.join(
            os.path.dirname(__file__), "carla_traffic_manager_helper.py"
        )
        command = [
            sys.executable,
            helper,
            "--host", str(args.host),
            "--rpc-port", str(args.port),
            "--tm-port", str(args.traffic_manager_port),
            "--timeout", str(timeout),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        ready, _, _ = select.select([process.stdout], [], [], float(timeout))
        if not ready:
            self._stop_traffic_manager_proxy(
                TrafficManagerProxy(process, timeout)
            )
            raise RuntimeError("TrafficManager helper startup timed out")
        response = process.stdout.readline().strip()
        if response != "READY":
            returncode = process.poll()
            self._stop_traffic_manager_proxy(
                TrafficManagerProxy(process, timeout)
            )
            raise RuntimeError(
                "TrafficManager helper failed to start: response={!r} "
                "returncode={!r}".format(response, returncode)
            )
        proxy = TrafficManagerProxy(process, timeout)
        atexit.register(self._stop_traffic_manager_proxy, proxy)
        return proxy

    @staticmethod
    def _first_route_town(args):
        """Read only the first route town for renderer-safe CARLA startup."""
        try:
            route = ET.parse(args.routes).getroot().find(".//route")
            if route is not None:
                return str(route.attrib.get("town", "") or "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _renderer_args_for_town(town):
        """Select renderer quality by map without changing sensor blueprints."""
        low_memory_args = os.environ.get("CVCI_CARLA_EXTRA_ARGS", "").strip()
        standard_map_args = os.environ.get(
            "CVCI_CARLA_STANDARD_MAP_EXTRA_ARGS", low_memory_args
        ).strip()
        return (
            low_memory_args
            if str(town or "").startswith("Town10")
            else standard_map_args
        )

    def _restart_simulation(self, args, target_town=""):
        """Restart CARLA while preserving this evaluator process and route."""
        old_server = getattr(self, "server", None)
        old_returncode = old_server.poll() if old_server is not None else None
        # Release all C++ client/TM handles while the old server can still
        # answer.  Destroying these handles after killing CARLA can raise an
        # uncaught carla::client::TimeoutException and abort this evaluator.
        old_traffic_manager = getattr(self, "traffic_manager", None)
        if self._server_process_alive():
            if old_traffic_manager is not None:
                try:
                    old_traffic_manager.set_synchronous_mode(False)
                    old_traffic_manager.set_hybrid_physics_mode(False)
                    old_traffic_manager.shut_down()
                except Exception:
                    print("CVCI_CARLA_RECOVERY traffic_manager_shutdown_failed=1", flush=True)
                    print(traceback.format_exc(), flush=True)
            try:
                CarlaDataProvider.cleanup()
            except Exception:
                print("CVCI_CARLA_RECOVERY data_provider_cleanup_failed=1", flush=True)
                print(traceback.format_exc(), flush=True)
            self.traffic_manager = None
            self.world = None
            self.client = None
            del old_traffic_manager
            gc.collect()
        else:
            # CARLA Python objects own C++ RPC handles whose destructors may
            # throw after the server has disappeared. A C++ exception from a
            # noexcept destructor terminates the evaluator before Python can
            # catch it. Retain every dead-server handle until os._exit() and
            # clear DataProvider state without issuing another RPC.
            orphaned = [
                getattr(self, "client", None),
                getattr(self, "world", None),
                old_traffic_manager,
                CarlaDataProvider._client,
                CarlaDataProvider._world,
                CarlaDataProvider._map,
                CarlaDataProvider._blueprint_library,
                CarlaDataProvider._all_actors,
                CarlaDataProvider._spawn_points,
                CarlaDataProvider._grp,
                list(CarlaDataProvider._carla_actor_pool.values()),
                list(CarlaDataProvider._actor_velocity_map.keys()),
                list(CarlaDataProvider._actor_location_map.keys()),
                list(CarlaDataProvider._actor_transform_map.keys()),
                list(CarlaDataProvider._traffic_light_map.keys()),
                list(CarlaDataProvider._traffic_light_map.values()),
            ]
            self._dead_carla_handles.append(orphaned)
            self._stop_traffic_manager_proxy(old_traffic_manager)
            CarlaDataProvider._actor_velocity_map = {}
            CarlaDataProvider._actor_location_map = {}
            CarlaDataProvider._actor_transform_map = {}
            CarlaDataProvider._traffic_light_map = {}
            CarlaDataProvider._carla_actor_pool = {}
            CarlaDataProvider._client = None
            CarlaDataProvider._world = None
            CarlaDataProvider._map = None
            CarlaDataProvider._sync_flag = False
            CarlaDataProvider._spawn_points = None
            CarlaDataProvider._spawn_index = 0
            CarlaDataProvider._blueprint_library = None
            CarlaDataProvider._all_actors = None
            CarlaDataProvider._ego_vehicle_route = None
            CarlaDataProvider._grp = None
            CarlaDataProvider._runtime_init_flag = False
            self.traffic_manager = None
            self.world = None
            self.client = None
            self._requires_hard_exit = True
            print(
                "CVCI_CARLA_RECOVERY dead_handles_orphaned=1 evaluator_pid={}".format(
                    os.getpid()
                ),
                flush=True,
            )
        self._stop_server_process(old_server)
        self._client_timed_out = False
        self.client, self.client_timeout, self.traffic_manager = self._setup_simulation(
            args, target_town
        )
        self._carla_restart_count = getattr(self, "_carla_restart_count", 0) + 1
        print(
            "CVCI_CARLA_RECOVERY restarted=1 evaluator_pid={} restart_count={} "
            "old_server_returncode={} rpc_port={} tm_port={}".format(
                os.getpid(), self._carla_restart_count, old_returncode,
                args.port, args.traffic_manager_port,
            ),
            flush=True,
        )

    def _setup_simulation(self, args, target_town=""):
        """
        Prepares the simulation by getting the client, and setting up the world and traffic manager settings
        """
        self.carla_path = os.environ["CARLA_ROOT"]
        args.port = find_free_port(args.port)
        graphics_adapter = os.environ.get("CVCI_CARLA_GRAPHICS_ADAPTER", str(args.gpu_rank))
        extra_args = self._renderer_args_for_town(target_town)
        carla_server = os.environ.get(
            "CARLA_SERVER", os.path.join(self.carla_path, "CarlaUE4.sh")
        )
        cmd1_raw = f"{shlex.quote(carla_server)} -RenderOffScreen -nosound -carla-rpc-port={args.port} -graphicsadapter={graphics_adapter}"
        if extra_args:
            cmd1_raw = f"{cmd1_raw} {extra_args}"
        if os.geteuid() == 0 and os.environ.get("CVCI_CARLA_SKIP_SU", "0") != "1":
            visible_devices = os.environ.get("CVCI_CARLA_CUDA_VISIBLE", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
            env_prefix = f"CUDA_VISIBLE_DEVICES={shlex.quote(visible_devices)} " if visible_devices else ""
            cmd1 = "su -s /bin/bash cvci -c " + shlex.quote(env_prefix + cmd1_raw)
        else:
            cmd1 = cmd1_raw
        self.server = subprocess.Popen(cmd1, shell=True, preexec_fn=os.setsid)
        print(cmd1, self.server.returncode, flush=True)
        atexit.register(self._stop_server_process, self.server)
        # CARLA startup time depends on the host.  Keep the benchmark default
        # at 30 seconds, but let the harness avoid a fixed idle wait on a
        # warm local GPU host.  The connection loop below still retries every
        # five seconds, so a slower server remains fully guarded.
        carla_boot_wait = max(
            0.0, float(os.environ.get("CVCI_CARLA_BOOT_WAIT", "30") or 30)
        )
        time.sleep(carla_boot_wait)
            
        attempts = 0
        num_max_restarts = 20
        while attempts < num_max_restarts:
            try:
                client = carla.Client(args.host, args.port)
                client_timeout = self.client_timeout
                client.set_timeout(client_timeout)

                settings = carla.WorldSettings(
                    synchronous_mode = True,
                    fixed_delta_seconds = 1.0 / self.frame_rate,
                    deterministic_ragdolls = True,
                    spectator_as_ego = False
                )
                client.get_world().apply_settings(settings)
                print(f"load_world success , attempts={attempts}", flush=True)
                break
            except Exception as e:
                print(f"load_world failed , attempts={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        else:
            raise RuntimeError("CARLA server did not become ready after {} attempts".format(num_max_restarts))
        attempts = 0
        num_max_restarts = 40
        while attempts < num_max_restarts:
            try:
                args.traffic_manager_port = find_free_port(args.traffic_manager_port)
                traffic_manager = self._start_traffic_manager_proxy(
                    args, min(float(client_timeout), 30.0)
                )
                print(f"traffic_manager init success, try_time={attempts}", flush=True)
                break
            except Exception as e:
                print(f"traffic_manager init fail, try_time={attempts}", flush=True)
                print(e, flush=True)
                attempts += 1
                time.sleep(5)
        else:
            raise RuntimeError("TrafficManager did not become ready after {} attempts".format(num_max_restarts))
        return client, client_timeout, traffic_manager

    def _load_world_with_recovery(self, args, town):
        """Load a route world, restarting CARLA and retrying the same route on failure."""
        retries = max(0, int(os.environ.get("CVCI_CARLA_ROUTE_LOAD_RETRIES", "0") or 0))
        proactive_interval = max(
            0,
            int(
                os.environ.get(
                    "CVCI_CARLA_PROACTIVE_RESTART_EVERY_WORLD_LOADS", "0"
                )
                or 0
            ),
        )
        successful_loads = int(
            getattr(self, "_successful_route_world_loads", 0) or 0
        )
        if (
            proactive_interval > 0
            and successful_loads > 0
            and successful_loads % proactive_interval == 0
            and str(town or "").startswith("Town10")
        ):
            # CARLA 0.9.15 can segfault on the sixth consecutive Town10HD
            # load_world even at low renderer quality. Restart the still-live
            # evaluator-owned simulator at a clean route boundary before its
            # native RPC handles become poisoned. This keeps one evaluator,
            # one checkpoint, and the same current route; it only rotates the
            # disposable CARLA server process. Standard maps must not take
            # this Town10-specific path: rotating Town04's healthy server can
            # leave the replacement TrafficManager unavailable while the
            # scenario constructor is creating its private criteria, causing
            # the scenario (and all challenge facts) to be skipped.
            print(
                "CVCI_CARLA_RECOVERY proactive_restart=1 evaluator_pid={} "
                "successful_world_loads={} interval={} next_town={}".format(
                    os.getpid(), successful_loads, proactive_interval, town
                ),
                flush=True,
            )
            self._restart_simulation(args, town)
        load_timeout = max(
            5.0,
            min(
                float(self.client_timeout),
                float(os.environ.get("CVCI_CARLA_ROUTE_LOAD_TIMEOUT", "90") or 90),
            ),
        )
        for attempt in range(retries + 1):
            try:
                self.client.set_timeout(load_timeout)
                self._load_and_wait_for_world(args, town)
                self.client.set_timeout(self.client_timeout)
                self._successful_route_world_loads = successful_loads + 1
                return
            except Exception:
                print(
                    "CVCI_CARLA_RECOVERY load_failed=1 evaluator_pid={} "
                    "attempt={}/{} town={} server_alive={}".format(
                        os.getpid(), attempt + 1, retries + 1, town,
                        int(self._server_process_alive()),
                    ),
                    flush=True,
                )
                print(traceback.format_exc(), flush=True)
                if attempt >= retries:
                    self._client_timed_out = True
                    raise
                self._restart_simulation(args, town)
        raise RuntimeError("unreachable CARLA route-load recovery state")

    def _reset_world_settings(self):
        """
        Changes the modified world settings back to asynchronous
        """
        # Has simulation failed?
        if (
            self.world and self.manager and not self._client_timed_out
            and self._server_process_alive()
        ):
            # Reset to asynchronous mode
            self.world.tick()  # TODO: Make sure all scenario actors have been destroyed
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            settings.deterministic_ragdolls = False
            settings.spectator_as_ego = True
            self.world.apply_settings(settings)

            # Make the TM back to async
            self.traffic_manager.set_synchronous_mode(False)
            self.traffic_manager.set_hybrid_physics_mode(False)

    def _load_and_wait_for_world(self, args, town):
        """
        Load a new CARLA world without changing the settings and provide data to CarlaDataProvider
        """
        helper = os.path.join(os.path.dirname(__file__), "carla_load_world_helper.py")
        timeout = max(
            5.0,
            min(
                float(self.client_timeout),
                float(os.environ.get("CVCI_CARLA_ROUTE_LOAD_TIMEOUT", "90") or 90),
            ),
        )
        command = [
            sys.executable,
            helper,
            "--host", str(args.host),
            "--port", str(args.port),
            "--timeout", str(timeout),
            "--town", str(town),
            "--frame-rate", str(self.frame_rate),
        ]
        print(
            "CVCI_CARLA_LOAD_HELPER start=1 evaluator_pid={} town={}".format(
                os.getpid(), town
            ),
            flush=True,
        )
        completed = subprocess.run(
            command,
            check=False,
            timeout=timeout + 15.0,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "isolated CARLA load-world helper failed with return code {}".format(
                    completed.returncode
                )
            )
        if not self._server_process_alive():
            raise RuntimeError("CARLA server exited during isolated world load")

        # The risky map switch has completed in the helper. Attach the main
        # evaluator client to the already-loaded world without another switch.
        self.world = self.client.get_world()
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_traffic_manager_port(args.traffic_manager_port)
        CarlaDataProvider.set_world(self.world)

        # This must be here so that all route repetitions use the same 'unmodified' seed
        self.traffic_manager.set_random_device_seed(args.traffic_manager_seed)

        # Wait for the world to be ready
        self.world.tick()

        map_name = CarlaDataProvider.get_map().name.split("/")[-1]
        if map_name != town:
            raise Exception("The CARLA server uses the wrong map!"
                            " This scenario requires the use of map {}".format(town))

    def _register_statistics(self, route_index, entry_status, crash_message=""):
        """
        Computes and saves the route statistics
        """
        print("\033[1m> Registering the route statistics\033[0m", flush=True)
        self.statistics_manager.save_entry_status(entry_status)
        self.statistics_manager.compute_route_statistics(
            route_index, self.manager.scenario_duration_system, self.manager.scenario_duration_game, crash_message
        )

    def _load_and_run_scenario(self, args, config):
        """
        Load and run the scenario given by config.

        Depending on what code fails, the simulation will either stop the route and
        continue from the next one, or report a crash and stop.
        """
        crash_message = ""
        entry_status = "Started"

        print("\n\033[1m========= Preparing {} (repetition {}) =========\033[0m".format(config.name, config.repetition_index), flush=True)

        # Prepare the statistics of the route
        route_name = f"{config.name}_rep{config.repetition_index}"
        # Audit-only identifiers are refreshed for every route in a full-144
        # evaluator process.  The sensor agent uses these values solely to
        # label its independent online-recognition CSV; formal behavior still
        # runs with CVCI_ALLOW_ROUTE_PRIOR=0 and no forced macro.
        route_id = str(config.name).rsplit("_", 1)[-1]
        os.environ["CVCI_ROUTE_ID"] = route_id
        macro_table = os.environ.get("CVCI_ROUTE_MACRO_TABLE", "").strip()
        if macro_table and os.path.exists(macro_table):
            import csv
            with open(macro_table, encoding="utf-8", newline="") as handle:
                expected = next(
                    (
                        row.get("expected_macro", "")
                        for row in csv.DictReader(handle)
                        if str(row.get("route_id", "")) == route_id
                    ),
                    "",
                )
            os.environ["EXPECTED_MACRO"] = expected
        scenario_name = config.scenario_configs[0].name
        town_name = str(config.town)
        weather_id = get_weather_id(config.weather[0][1])
        currentDateAndTime = datetime.now()
        currentTime = currentDateAndTime.strftime("%m_%d_%H_%M_%S")
        save_name = f"{route_name}_{town_name}_{scenario_name}_{weather_id}_{currentTime}"
        self.statistics_manager.create_route_data(route_name, scenario_name, weather_id, save_name, town_name, config.index)

        print("\033[1m> Loading the world\033[0m", flush=True)

        # Load the world and the scenario
        try:
            self._load_world_with_recovery(args, config.town)
            self.route_scenario = RouteScenario(world=self.world, config=config, debug_mode=args.debug)
            self.statistics_manager.set_scenario(self.route_scenario)

        except Exception:
            # The scenario is wrong -> set the ejecution to crashed and stop
            print("\n\033[91mThe scenario could not be loaded:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Setting up the agent\033[0m", flush=True)

        # Set up the user's agent, and the timer to avoid freezing the simulation
        try:
            self._agent_watchdog = Watchdog(self.agent_timeout)
            self._agent_watchdog.start()
            agent_class_name = getattr(self.module_agent, 'get_entry_point')()
            agent_class_obj = getattr(self.module_agent, agent_class_name)

            # Start the ROS1 bridge server only for ROS1 based agents.
            if getattr(agent_class_obj, 'get_ros_version')() == 1 and self._ros1_server is None:
                from leaderboard.autoagents.ros1_agent import ROS1Server
                self._ros1_server = ROS1Server()
                self._ros1_server.start()

            self.agent_instance = agent_class_obj(args.host, args.port, args.debug)
            self.agent_instance.set_global_plan(self.route_scenario.gps_route, self.route_scenario.route)
            # Keep the base config immutable across a multi-route evaluator
            # process.  Mutating args.agent_config here used to append every
            # prior route's save name to the next route configuration.
            route_agent_config = args.agent_config + '+' + save_name
            self.agent_instance.setup(route_agent_config)

            # Check and store the sensors
            if not self.sensors:
                self.sensors = self.agent_instance.sensors()
                track = self.agent_instance.track

                validate_sensor_configuration(self.sensors, track, args.track)

                self.sensor_icons = [sensors_to_icons[sensor['type']] for sensor in self.sensors]
                self.statistics_manager.save_sensors(self.sensor_icons)
                self.statistics_manager.write_statistics()

                self.sensors_initialized = True

            self._agent_watchdog.stop()
            self._agent_watchdog = None

        except SensorConfigurationInvalid as e:
            # The sensors are invalid -> set the ejecution to rejected and stop
            print("\n\033[91mThe sensor's configuration used is invalid:", flush=True)
            print(f"{e}\033[0m\n", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Sensors"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        except Exception as e:
            # The agent setup has failed -> start the next route
            print("\n\033[91mCould not set up the required agent:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)
            print(f"{e}\033[0m\n", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Agent_init"]
            self._register_statistics(config.index, entry_status, crash_message)
            self._cleanup()
            return True

        print("\033[1m> Running the route\033[0m", flush=True)

        # Run the scenario
        try:
            # Load scenario and run it
            if args.record:
                self.client.start_recorder("{}/{}_rep{}.log".format(args.record, config.name, config.repetition_index))
            self.manager.load_scenario(self.route_scenario, self.agent_instance, config.index, config.repetition_index)
            self.manager.tick_count = 0
            self.manager.run_scenario()

        except AgentError:
            # The agent has failed -> stop the route
            print("\n\033[91mStopping the route, the agent has crashed:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m")

            entry_status, crash_message = FAILURE_MESSAGES["Agent_runtime"]

        except KeyboardInterrupt:
            return True
        
        except TickRuntimeError:
            entry_status, crash_message = "Started", "TickRuntime"
        
        except Exception:
            print("\n\033[91mError during the simulation:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            entry_status, crash_message = FAILURE_MESSAGES["Simulation"]

        # Stop the scenario
        try:
            print("\033[1m> Stopping the route\033[0m", flush=True)
            self.manager.stop_scenario()
            self._register_statistics(config.index, entry_status, crash_message)

            if args.record:
                self.client.stop_recorder()

            self._cleanup()

        except Exception:
            print("\n\033[91mFailed to stop the scenario, the statistics might be empty:", flush=True)
            print(f"\n{traceback.format_exc()}\033[0m", flush=True)

            _, crash_message = FAILURE_MESSAGES["Simulation"]

        # If the simulation crashed, stop the leaderboard, for the rest, move to the next route
        return crash_message == "Simulation crashed"

    def run(self, args):
        """
        Run the challenge mode
        """
        route_indexer = RouteIndexer(args.routes, args.repetitions, args.routes_subset)

        if args.resume:
            resume = route_indexer.validate_and_resume(args.checkpoint)
        else:
            resume = False

        if resume:
            self.statistics_manager.add_file_records(args.checkpoint)
        else:
            self.statistics_manager.clear_records()
        self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
        self.statistics_manager.write_statistics()

        crashed = False
        t1 = time.time()
        while route_indexer.peek() and not crashed:

            # Run the scenario
            config = route_indexer.get_next_config()
            route_run_retries = max(
                0,
                int(
                    os.environ.get(
                        "CVCI_CARLA_ROUTE_RUN_RETRIES", "0"
                    ) or 0
                ),
            )
            route_run_attempt = 0
            while True:
                crashed = self._load_and_run_scenario(args, config)
                if not crashed or route_run_attempt >= route_run_retries:
                    break
                route_run_attempt += 1
                print(
                    "CVCI_CARLA_RECOVERY route_run_retry=1 "
                    "evaluator_pid={} route={} attempt={}/{}".format(
                        os.getpid(), config.name, route_run_attempt,
                        route_run_retries,
                    ),
                    flush=True,
                )
                # get_next_config() has already advanced the indexer.  Retry
                # this same immutable config in place; create_route_data()
                # overwrites its route index, so a failed infrastructure
                # attempt cannot create a duplicate record.  The next world
                # load performs the evaluator-owned CARLA restart.
                crashed = False
            # Save the progress and write the route statistics
            self.statistics_manager.save_progress(route_indexer.index, route_indexer.total)
            self.statistics_manager.write_statistics()
            if crashed:
                print(f'{route_indexer.index} crash, [{route_indexer.index}/{route_indexer.total}], please restart', flush=True)
                break

        # Shutdown ROS1 bridge server if necessary
        if self._ros1_server is not None:
            self._ros1_server.shutdown()

        # Go back to asynchronous mode
        self._reset_world_settings()

        if not crashed:
            # Save global statistics
            print(f"cost time={time.time()-t1}", flush=True)
            print("\033[1m> Registering the global statistics\033[0m", flush=True)
            self.statistics_manager.compute_global_statistics()
            self.statistics_manager.validate_and_write_statistics(self.sensors_initialized, crashed)
        
        if crashed:
            self._stop_server_process(getattr(self, "server", None))

        return crashed

def main():
    description = "CARLA AD Leaderboard Evaluation: evaluate your Agent in CARLA scenarios\n"

    # general parameters
    parser = argparse.ArgumentParser(description=description, formatter_class=RawTextHelpFormatter)
    parser.add_argument('--host', default='localhost',
                        help='IP of the host server (default: localhost)')
    parser.add_argument('--port', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--traffic-manager-port', default=8000, type=int,
                        help='Port to use for the TrafficManager (default: 8000)')
    parser.add_argument('--traffic-manager-seed', default=0, type=int,
                        help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--debug', type=int,
                        help='Run with debug output', default=0)
    parser.add_argument('--record', type=str, default='',
                        help='Use CARLA recording feature to create a recording of the scenario')
    parser.add_argument('--client-timeout', default=300.0, type=float,
                        help='CARLA client/RPC timeout in seconds')
    parser.add_argument('--scenario-timeout', default=300.0, type=float,
                        help='Maximum allowed scenario duration in seconds')
    parser.add_argument('--agent-timeout', default=20.0, type=float,
                        help='Maximum allowed agent setup/tick time in seconds')

    # simulation setup
    parser.add_argument('--routes', required=True,
                        help='Name of the routes file to be executed.')
    parser.add_argument('--routes-subset', default='', type=str,
                        help='Execute a specific set of routes')
    parser.add_argument('--repetitions', type=int, default=1,
                        help='Number of repetitions per route.')

    # agent-related options
    parser.add_argument("-a", "--agent", type=str,
                        help="Path to Agent's py file to evaluate", required=True)
    parser.add_argument("--agent-config", type=str,
                        help="Path to Agent's configuration file", default="")

    parser.add_argument("--track", type=str, default='SENSORS',
                        help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='./simulation_results.json',
                        help="Path to checkpoint used for saving statistics and resuming")
    parser.add_argument("--debug-checkpoint", type=str, default='./live_results.txt',
                        help="Path to checkpoint used for saving live results")
    parser.add_argument("--gpu-rank", type=int, default=0)
    arguments = parser.parse_args()

    statistics_manager = StatisticsManager(arguments.checkpoint, arguments.debug_checkpoint)
    leaderboard_evaluator = LeaderboardEvaluator(arguments, statistics_manager)
    crashed = leaderboard_evaluator.run(arguments)

    if crashed:
        # Python/CUDA worker pools can deadlock during interpreter teardown
        # after a CARLA process crash.  The evaluator has already flushed the
        # checkpoint and stopped its CARLA process, so let the harness wrapper
        # continue deterministically instead of hanging forever.
        if os.environ.get("CVCI_HARD_EXIT_ON_SIM_CRASH", "0") == "1":
            leaderboard_evaluator._stop_traffic_manager_proxy(
                getattr(leaderboard_evaluator, "traffic_manager", None)
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(255)
        del leaderboard_evaluator
        sys.exit(-1)
    else:
        try:
            leaderboard_evaluator.traffic_manager.shut_down()
        except Exception:
            print("CVCI traffic_manager final shutdown failed", flush=True)
            print(traceback.format_exc(), flush=True)
        leaderboard_evaluator._stop_traffic_manager_proxy(
            getattr(leaderboard_evaluator, "traffic_manager", None)
        )
        if (
            getattr(leaderboard_evaluator, "_requires_hard_exit", False)
            and os.environ.get("CVCI_HARD_EXIT_ON_SIM_CRASH", "0") == "1"
        ):
            leaderboard_evaluator._stop_server_process(
                getattr(leaderboard_evaluator, "server", None)
            )
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
        del leaderboard_evaluator
        sys.exit(0)

if __name__ == '__main__':
    main()
