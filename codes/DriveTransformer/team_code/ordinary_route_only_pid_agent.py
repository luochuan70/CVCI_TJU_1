"""Sensor-only ordinary-route recovery agent.

This agent is an ordinary-220 diagnostic/recovery path.  It does not load the
DriveTransformer checkpoint, so it cannot exercise the frozen 144-route
implementation and cannot trigger the six-camera/CUDA rendering failure.
Steering is still CARLA's public PID controller, fed by the leaderboard route
plan converted from GNSS and oriented by IMU compass.  LiDAR is used only for
observable obstacle braking and a bounded open-side offset.
"""
from __future__ import absolute_import

import math
import os

import carla
import numpy as np
from scipy.optimize import fsolve

from DriveTransformer.team_code.cvci_auxiliary_system import CVCIAuxiliarySystem
from DriveTransformer.team_code.planner import RoutePlanner
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track


EARTH_RADIUS_EQUA = 6378137.0


class OrdinaryRouteOnlyPIDAgent(AutonomousAgent):
    def setup(self, path_to_conf_file):
        del path_to_conf_file
        self.track = Track.SENSORS
        self.step = -1
        self._target_speed = float(
            os.environ.get("CVCI_ORDINARY_TARGET_SPEED", "6.0")
        )
        self._throttle_limit = float(
            os.environ.get("CVCI_ORDINARY_THROTTLE_LIMIT", "0.55")
        )
        self._pid_log_period = max(
            0, int(os.environ.get("CVCI_ORDINARY_PID_LOG_PERIOD", "0") or 0)
        )
        self._hard_brake_distance = float(
            os.environ.get("CVCI_ORDINARY_LIDAR_HARD_BRAKE", "2.5")
        )
        self._soft_brake_distance = float(
            os.environ.get("CVCI_ORDINARY_LIDAR_SOFT_BRAKE", "4.0")
        )
        self._caution_distance = float(
            os.environ.get("CVCI_ORDINARY_LIDAR_CAUTION", "5.5")
        )
        self._pass_offset_sign = float(
            os.environ.get("CVCI_ORDINARY_PASS_OFFSET_SIGN", "1.0")
        )
        self._lidar_safety_enabled = os.environ.get(
            "CVCI_ORDINARY_LIDAR_SAFETY", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._use_lidar = os.environ.get(
            "CVCI_ORDINARY_ROUTE_ONLY_LIDAR", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._pass_offset = 0.0
        self._pass_clear_frames = 0
        self._last_tick = None

        self.lat_ref, self.lon_ref = self._solve_gps_reference()
        self._route_planner = RoutePlanner(
            4.0, 50.0, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._route_planner.set_route(self._global_plan, True)
        self._command_planner = RoutePlanner(
            7.5, 25.0, 257, lat_ref=self.lat_ref, lon_ref=self.lon_ref
        )
        self._command_planner.set_route(self._global_plan, True)
        self._auxiliary = CVCIAuxiliarySystem()

    def _solve_gps_reference(self):
        try:
            locx = self._global_plan_world_coord[0][0].location.x
            locy = self._global_plan_world_coord[0][0].location.y
            lon = self._global_plan[0][0]["lon"]
            lat = self._global_plan[0][0]["lat"]

            def equations(values):
                x, y = values
                eq1 = lon * math.cos(x * math.pi / 180.0) - (
                    locx * x * 180.0
                ) / (math.pi * EARTH_RADIUS_EQUA) - math.cos(
                    x * math.pi / 180.0
                ) * y
                eq2 = (
                    math.log(math.tan((lat + 90.0) * math.pi / 360.0))
                    * EARTH_RADIUS_EQUA
                    * math.cos(x * math.pi / 180.0)
                    + locy
                    - math.cos(x * math.pi / 180.0)
                    * EARTH_RADIUS_EQUA
                    * math.log(math.tan((90.0 + x) * math.pi / 360.0))
                )
                return [eq1, eq2]

            solution = fsolve(equations, [0.0, 0.0])
            return float(solution[0]), float(solution[1])
        except Exception as exc:
            print("ordinary route GPS reference fallback: {}".format(exc), flush=True)
            return 0.0, 0.0

    def sensors(self):
        # No RGB/BEV sensors are requested.  This is deliberately separate
        # from the frozen six-camera suite and is selected only by the
        # ordinary recovery launcher.
        sensors = [
            {
                "type": "sensor.other.imu",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.05,
                "id": "IMU",
            },
            {
                "type": "sensor.other.gnss",
                "x": -1.4,
                "y": 0.0,
                "z": 0.0,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "sensor_tick": 0.01,
                "id": "GPS",
            },
            {"type": "sensor.speedometer", "reading_frequency": 20, "id": "SPEED"},
        ]
        if self._use_lidar:
            sensors.append(
                {
                "type": "sensor.lidar.ray_cast",
                "x": 0.0,
                "y": 0.0,
                "z": 2.4,
                "roll": 0.0,
                "pitch": 0.0,
                "yaw": 0.0,
                "channels": 64,
                "range": 85.0,
                "points_per_second": 120000,
                "rotation_frequency": 10.0,
                "upper_fov": 10.0,
                "lower_fov": -25.0,
                "id": "LIDAR",
                }
            )
        return sensors

    @staticmethod
    def _lidar_corridor(points):
        if points is None:
            return None, None, None
        try:
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] < 2 or not len(points):
                return None, None, None
            finite = points[np.isfinite(points[:, :2]).all(axis=1)]
            if not len(finite):
                return None, None, None
            front = finite[
                (finite[:, 0] > 0.5)
                & (finite[:, 0] < 25.0)
                & (np.abs(finite[:, 1]) < 1.0)
            ]
            front_distance = (
                None if not len(front) else float(np.min(front[:, 0]))
            )
            side = finite[(finite[:, 0] > 0.5) & (finite[:, 0] < 10.0)]
            right = side[side[:, 1] > 1.0]
            left = side[side[:, 1] < -1.0]
            right_clearance = None if not len(right) else float(np.min(right[:, 0]))
            left_clearance = None if not len(left) else float(np.min(left[:, 0]))
            return front_distance, left_clearance, right_clearance
        except Exception:
            return None, None, None

    @staticmethod
    def _clearance_value(value):
        return 10.0 if value is None else float(np.clip(value, 0.0, 10.0))

    def _tick(self, input_data):
        self.step += 1
        gps = np.asarray(input_data["GPS"][1][:2], dtype=float)
        speed = max(0.0, float(input_data["SPEED"][1]["speed"]))
        compass = float(input_data["IMU"][1][-1])
        if not np.isfinite(compass):
            compass = 0.0
        pos = self._route_planner.gps_to_location(gps)
        near_node, near_command = self._route_planner.run_step(pos)
        far_node, far_command = self._command_planner.run_step(pos)
        tick_data = {
            "gps": gps,
            "pos": pos,
            "speed": speed,
            "compass": compass,
            "command_near": near_command,
            "command_near_xy": near_node,
            "command_far": far_command,
            "command_far_xy": far_node,
            "lidar_points": (
                input_data.get("LIDAR", (None, None))[1]
                if self._use_lidar
                else None
            ),
        }
        self._last_tick = tick_data
        return tick_data

    def run_step(self, input_data, timestamp):
        del timestamp
        tick_data = self._tick(input_data)
        speed = tick_data["speed"]
        front, left, right = self._lidar_corridor(tick_data["lidar_points"])
        steer = self._auxiliary._carla_route_pid_steer(
            phase="missing_car_route_rejoin",
            tick_data=tick_data,
            route_lateral_error=None,
            target_route_offset=self._pass_offset,
            steer_limit=0.85,
            fallback_steer=0.0,
        )

        # If a near obstacle persists, select the side with more observable
        # clearance and temporarily offset the live route target.  The route
        # PID remains the only steering controller; this is just a bounded
        # sensor-derived target offset.
        if self._lidar_safety_enabled and front is not None and front < self._caution_distance:
            left_value = self._clearance_value(left)
            right_value = self._clearance_value(right)
            if max(left_value, right_value) > 4.0:
                self._pass_offset = (
                    self._pass_offset_sign * (1.5 if right_value >= left_value else -1.5)
                )
                self._pass_clear_frames = 0
            else:
                self._pass_clear_frames += 1
        elif self._pass_offset != 0.0:
            self._pass_clear_frames += 1
            if self._pass_clear_frames > 40:
                self._pass_offset = 0.0
                self._pass_clear_frames = 0

        if self._pass_offset != 0.0:
            steer = self._auxiliary._carla_route_pid_steer(
                phase="missing_car_route_rejoin",
                tick_data=tick_data,
                route_lateral_error=None,
                target_route_offset=self._pass_offset,
                steer_limit=0.85,
                fallback_steer=steer,
            )

        throttle = float(
            np.clip(
                0.20 + 0.10 * (self._target_speed - speed),
                0.0,
                self._throttle_limit,
            )
        )
        brake = 0.0
        if self._lidar_safety_enabled and front is not None:
            if front < self._hard_brake_distance:
                throttle, brake = 0.0, 1.0
            elif front < self._soft_brake_distance and self._pass_offset == 0.0:
                throttle, brake = 0.0, 0.65
            elif front < self._caution_distance:
                throttle = min(throttle, 0.20)
        if self.step <= 20:
            throttle, brake = 0.0, 1.0

        if self._pid_log_period > 0 and self.step % self._pid_log_period == 0:
            print(
                "CVCI_ORDINARY_ROUTE_ONLY frame={} speed={:.4f} steer={:.4f} "
                "throttle={:.4f} brake={:.4f} front={} left={} right={} offset={:.2f} "
                "pos={}".format(
                    self.step,
                    speed,
                    float(steer),
                    throttle,
                    brake,
                    front,
                    left,
                    right,
                    self._pass_offset,
                    np.asarray(tick_data["pos"], dtype=float)[:2].tolist(),
                ),
                flush=True,
            )
        return carla.VehicleControl(
            steer=float(steer),
            throttle=float(throttle),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
        )


def get_entry_point():
    return "OrdinaryRouteOnlyPIDAgent"
