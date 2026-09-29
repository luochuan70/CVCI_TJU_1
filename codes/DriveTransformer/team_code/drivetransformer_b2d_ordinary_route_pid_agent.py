"""Ordinary-route route-PID recovery agent.

This module is intentionally separate from the frozen 0629 agent.  It keeps
the frozen model alive for perception, but replaces its unreliable ordinary
route longitudinal/steering output with sensor-derived route following.  The
CARLA public lateral PID receives only GNSS/IMU route geometry; LiDAR and the
auxiliary debug state provide conservative braking.  The module is never
selected by the 144-route frozen launcher.
"""
from __future__ import annotations

import os

import carla
import numpy as np

from DriveTransformer.team_code.drivetransformer_b2d_four_camera_recovery_agent import (
    FourCameraRecoveryAgent,
)


class OrdinaryRoutePIDAgent(FourCameraRecoveryAgent):
    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self._ordinary_target_speed = float(
            os.environ.get("CVCI_ORDINARY_TARGET_SPEED", "5.5")
        )
        self._ordinary_throttle_limit = float(
            os.environ.get("CVCI_ORDINARY_THROTTLE_LIMIT", "0.55")
        )
        self._ordinary_pid_log_period = max(
            0, int(os.environ.get("CVCI_ORDINARY_PID_LOG_PERIOD", "0") or 0)
        )

    @staticmethod
    def _front_lidar_distance(tick_data):
        points = tick_data.get("lidar_points")
        if points is None:
            return None
        try:
            points = np.asarray(points)
            if points.ndim != 2 or points.shape[1] < 2 or points.shape[0] == 0:
                return None
            front = points[
                (points[:, 0] > 0.5)
                & (points[:, 0] < 25.0)
                & (np.abs(points[:, 1]) < 1.2)
            ]
            if front.shape[0] == 0:
                return None
            return float(np.min(front[:, 0]))
        except Exception:
            return None

    def run_step(self, input_data, timestamp):
        # The frozen model still supplies detections and the ordinary
        # auxiliary system still logs its decision.  Only the final ordinary
        # control arbitration is replaced here.
        model_control = super().run_step(input_data, timestamp)
        tick_data = getattr(self, "_recovery_last_tick", None)
        if not tick_data:
            return model_control

        speed = max(0.0, float(tick_data.get("speed", 0.0)))
        steer = self.cvci_auxiliary_system._carla_route_pid_steer(
            phase="missing_car_route_rejoin",
            tick_data=tick_data,
            route_lateral_error=None,
            target_route_offset=0.0,
            steer_limit=0.85,
            fallback_steer=float(model_control.steer),
        )

        # A small, bounded longitudinal controller keeps moving on ordinary
        # routes while retaining hard sensor-derived safety stops.
        target_speed = self._ordinary_target_speed
        throttle = float(np.clip(0.18 + 0.12 * (target_speed - speed), 0.0, self._ordinary_throttle_limit))
        brake = 0.0
        lidar_distance = self._front_lidar_distance(tick_data)
        if lidar_distance is not None:
            if lidar_distance < 2.5:
                throttle, brake = 0.0, 1.0
            elif lidar_distance < 5.0:
                throttle, brake = 0.0, 0.65
            elif lidar_distance < 8.0:
                throttle = min(throttle, 0.12)

        debug = getattr(self.cvci_auxiliary_system, "last_debug", {}) or {}
        if bool(debug.get("red_light_active", False)):
            red_distance = debug.get("red_stop_distance")
            try:
                if red_distance is None or float(red_distance) < 12.0:
                    throttle, brake = 0.0, 1.0
            except Exception:
                throttle, brake = 0.0, 1.0

        if self.step <= 20:
            throttle, brake = 0.0, 1.0

        if (
            self._ordinary_pid_log_period > 0
            and self.step % self._ordinary_pid_log_period == 0
        ):
            print(
                "CVCI_ORDINARY_PID frame={} speed={:.4f} steer={:.4f} "
                "throttle={:.4f} brake={:.4f} lidar_front={} "
                "near={} far={} pos={} compass={:.4f}".format(
                    self.step,
                    speed,
                    float(steer),
                    float(throttle),
                    float(brake),
                    "None" if lidar_distance is None else "{:.3f}".format(lidar_distance),
                    np.asarray(tick_data.get("command_near_xy"), dtype=float)[:2].tolist(),
                    np.asarray(tick_data.get("command_far_xy"), dtype=float)[:2].tolist(),
                    np.asarray(tick_data.get("pos"), dtype=float)[:2].tolist(),
                    float(tick_data.get("compass")),
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
    return "OrdinaryRoutePIDAgent"
