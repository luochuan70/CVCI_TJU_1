"""Ordinary-route-only recovery wrapper for the frozen 0629 agent.

The parent class is the unchanged 0629 DriveTransformer agent.  This wrapper
is selected only by an ordinary-route launcher and only post-processes a
control when the auxiliary system has already identified the route as
``ordinary_iter25000``.  Dedicated CVCI routes therefore keep the original
control path, sensors, recognition, and rules.

The recovery is deliberately narrow: after a sustained static-obstacle
collision brake or clear-road startup brake at near-zero speed, it applies a
short, bounded forward pulse.  It never overrides a confirmed red light, a
pedestrian, or a detected front vehicle.
"""

from __future__ import absolute_import

import os

import carla

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as Frozen0629DriveTransformerAgent,
)


class OrdinaryRecoveryDriveTransformerAgent(Frozen0629DriveTransformerAgent):
    """Frozen 0629 model plus an ordinary-only anti-deadlock pulse."""

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self._ordinary_recovery_enabled = os.environ.get(
            "CVCI_ORDINARY_RECOVERY_ENABLED", "0"
        ).lower() in ("1", "true", "yes", "on")
        self._ordinary_recovery_hold_frames = max(
            1, int(os.environ.get("CVCI_ORDINARY_RECOVERY_HOLD_FRAMES", "20"))
        )
        self._ordinary_recovery_pulse_frames = max(
            1, int(os.environ.get("CVCI_ORDINARY_RECOVERY_PULSE_FRAMES", "12"))
        )
        self._ordinary_recovery_throttle = min(
            0.45,
            max(
                0.10,
                float(os.environ.get("CVCI_ORDINARY_RECOVERY_THROTTLE", "0.30")),
            ),
        )
        self._ordinary_recovery_steer_bias = min(
            0.22,
            max(
                0.04,
                float(os.environ.get("CVCI_ORDINARY_RECOVERY_STEER_BIAS", "0.12")),
            ),
        )
        self._ordinary_recovery_min_step = max(
            1, int(os.environ.get("CVCI_ORDINARY_RECOVERY_MIN_STEP", "40"))
        )
        self._ordinary_recovery_hold = 0
        self._ordinary_recovery_pulse = 0
        self._ordinary_recovery_event_count = 0

    @staticmethod
    def _ordinary_debug_is_active(debug):
        return bool(
            isinstance(debug, dict)
            and debug.get("standard_fallback") is True
            and debug.get("standard_only_mode") is True
            and debug.get("route_mode") == "ordinary_iter25000"
            and str(debug.get("first_frame_macro") or "unknown") == "unknown"
        )

    @staticmethod
    def _control_copy(control, steer=None, throttle=None, brake=None):
        return carla.VehicleControl(
            steer=float(control.steer if steer is None else steer),
            throttle=float(control.throttle if throttle is None else throttle),
            brake=float(control.brake if brake is None else brake),
            hand_brake=bool(getattr(control, "hand_brake", False)),
            reverse=bool(getattr(control, "reverse", False)),
        )

    def _ordinary_recovery_postprocess(self, control, input_data):
        if not self._ordinary_recovery_enabled:
            return control

        debug = getattr(self.cvci_auxiliary_system, "last_debug", {}) or {}
        if not self._ordinary_debug_is_active(debug):
            self._ordinary_recovery_hold = 0
            self._ordinary_recovery_pulse = 0
            return control

        if self._ordinary_recovery_pulse > 0:
            self._ordinary_recovery_pulse -= 1
            return self._control_copy(
                control,
                throttle=self._ordinary_recovery_throttle,
                brake=0.0,
            )

        try:
            speed = abs(float(input_data["SPEED"][1]["speed"]))
        except Exception:
            speed = abs(float(debug.get("ego_speed") or 0.0))

        reason = str(debug.get("reason") or "")
        red_active = bool(debug.get("red_light_active"))
        red_distance = debug.get("red_stop_distance")
        pedestrian = debug.get("front_pedestrian_distance")
        vehicle = debug.get("front_vehicle_distance")
        obstacle = debug.get("front_obstacle_distance")
        collision_brake = reason == "standard_immediate_collision_brake"
        lidar_front = debug.get("lidar_front_distance")
        clear_path_candidate = (
            int(getattr(self, "step", 0)) >= self._ordinary_recovery_min_step
            and speed < 0.25
            and not red_active
            and red_distance is None
            and pedestrian is None
            and vehicle is None
            and obstacle is None
            and (lidar_front is None or float(lidar_front) > 8.0)
            and (
                float(control.brake) > 0.45
                or float(control.throttle) < 0.05
            )
        )
        static_collision_candidate = (
            collision_brake
            and speed < 0.25
            and not red_active
            and red_distance is None
            and pedestrian is None
            and vehicle is None
            and obstacle is not None
        )
        recovery_candidate = clear_path_candidate or static_collision_candidate
        if not recovery_candidate:
            self._ordinary_recovery_hold = 0
            return control

        self._ordinary_recovery_hold += 1
        if self._ordinary_recovery_hold < self._ordinary_recovery_hold_frames:
            return control

        recovery_reason = (
            "ordinary_clear_path_start"
            if clear_path_candidate
            else "ordinary_static_obstacle_unwedge"
        )
        open_side = str(
            debug.get("advanced_lidar_semantic_open_side")
            or debug.get("lidar_open_side")
            or ""
        ).lower()
        if clear_path_candidate:
            steer = float(control.steer)
        elif open_side == "right":
            steer = float(control.steer) + self._ordinary_recovery_steer_bias
        elif open_side == "left":
            steer = float(control.steer) - self._ordinary_recovery_steer_bias
        else:
            sign = 1.0 if self._ordinary_recovery_event_count % 2 == 0 else -1.0
            steer = float(control.steer) + sign * self._ordinary_recovery_steer_bias
        steer = max(-0.45, min(0.45, steer))
        self._ordinary_recovery_event_count += 1
        self._ordinary_recovery_hold = 0
        self._ordinary_recovery_pulse = self._ordinary_recovery_pulse_frames - 1
        print(
            "CVCI_ORDINARY_RECOVERY event={} reason={} speed={:.4f} "
            "obstacle={} open_side={} steer={:.4f}".format(
                self._ordinary_recovery_event_count,
                recovery_reason,
                speed,
                obstacle,
                open_side,
                steer,
            ),
            flush=True,
        )
        return self._control_copy(
            control,
            steer=steer,
            throttle=self._ordinary_recovery_throttle,
            brake=0.0,
        )

    def run_step(self, input_data, timestamp):
        control = super().run_step(input_data, timestamp)
        return self._ordinary_recovery_postprocess(control, input_data)


def get_entry_point():
    return "OrdinaryRecoveryDriveTransformerAgent"
