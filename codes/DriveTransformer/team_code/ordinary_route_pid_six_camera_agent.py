"""Ordinary-route six-camera route-PID recovery adapter.

This module is selected only by the ordinary 28-route recovery launcher.  It
inherits the frozen 0629 agent unchanged, so the same six RGB cameras,
checkpoint, model inference, and ordinary perception remain active.  After
the frozen agent has produced its ordinary control, this adapter replaces
only the ordinary longitudinal/route-following arbitration with CARLA's
public lateral PID fed by live GNSS/IMU route geometry.  LiDAR is used only
for a bounded, sensor-derived obstacle stop/side offset.

The frozen 144-route launcher never selects this entry point.
"""
from __future__ import absolute_import

import os

import carla
import cv2
import numpy as np

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as _FrozenDriveTransformerAgent,
)


class OrdinaryRoutePIDSixCameraAgent(_FrozenDriveTransformerAgent):
    """Frozen 0629 perception/model with ordinary-only route-PID arbitration."""

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self._ordinary_target_speed = float(
            os.environ.get("CVCI_ORDINARY_TARGET_SPEED", "5.5")
        )
        self._ordinary_throttle_limit = float(
            os.environ.get("CVCI_ORDINARY_THROTTLE_LIMIT", "0.55")
        )
        self._ordinary_lidar_safety = os.environ.get(
            "CVCI_ORDINARY_LIDAR_SAFETY", "1"
        ).lower() in ("1", "true", "yes", "on")
        self._ordinary_pass_offset = 0.0
        self._ordinary_pass_clear_frames = 0
        self._ordinary_pid_log_period = max(
            0,
            int(os.environ.get("CVCI_ORDINARY_PID_LOG_PERIOD", "0") or 0),
        )
        self._ordinary_camera_lowres = os.environ.get(
            "CVCI_ORDINARY_CAMERA_LOWRES", "0"
        ).lower() in ("1", "true", "yes", "on")
        if self._ordinary_camera_lowres:
            # Keep all six physical camera IDs and poses, but reduce CARLA's
            # render target for ordinary-route crash recovery. The
            # frozen model still receives its historical 1600x900 tensor after
            # the adapter upsamples each frame before frozen preprocessing.
            width = max(
                160,
                int(os.environ.get("CVCI_ORDINARY_CAMERA_WIDTH", "800")),
            )
            height = max(
                90,
                int(os.environ.get("CVCI_ORDINARY_CAMERA_HEIGHT", "450")),
            )
            for camera_name in self.cameras:
                sensor = dict(self.all_sensors[camera_name])
                sensor["width"] = width
                sensor["height"] = height
                self.all_sensors[camera_name] = sensor
        self._ordinary_last_tick = None

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        self._ordinary_last_tick = tick_data
        return tick_data

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
            side = finite[(finite[:, 0] > 0.5) & (finite[:, 0] < 10.0)]
            left = side[side[:, 1] < -1.0]
            right = side[side[:, 1] > 1.0]
            return (
                None if not len(front) else float(np.min(front[:, 0])),
                None if not len(left) else float(np.min(left[:, 0])),
                None if not len(right) else float(np.min(right[:, 0])),
            )
        except Exception:
            return None, None, None

    @staticmethod
    def _clearance(value):
        return 10.0 if value is None else float(np.clip(value, 0.0, 10.0))

    def _route_pid_steer(self, tick_data, fallback):
        return self.cvci_auxiliary_system._carla_route_pid_steer(
            phase="missing_car_route_rejoin",
            tick_data=tick_data,
            route_lateral_error=None,
            target_route_offset=self._ordinary_pass_offset,
            steer_limit=0.85,
            fallback_steer=float(fallback),
        )

    def run_step(self, input_data, timestamp):
        # Keep the frozen model, six-camera sensor path, recognition and
        # auxiliary perception running.  This call also updates the live
        # route-plan/GNSS/IMU tick retained by tick().
        adapted_input = input_data
        if self._ordinary_camera_lowres:
            adapted_input = dict(input_data)
            for camera_name in self.cameras:
                packet = input_data.get(camera_name)
                if packet is None or len(packet) != 2:
                    continue
                frame, image = packet
                if getattr(image, "shape", (0, 0))[:2] != (900, 1600):
                    image = cv2.resize(image, (1600, 900), interpolation=cv2.INTER_LINEAR)
                    adapted_input[camera_name] = (frame, image)
        model_control = super().run_step(adapted_input, timestamp)
        tick_data = self._ordinary_last_tick
        if not tick_data:
            return model_control

        speed = max(0.0, float(tick_data.get("speed", 0.0)))
        front, left, right = self._lidar_corridor(
            tick_data.get("lidar_points")
        )

        # Select a bounded live side offset only when LiDAR observes a close
        # obstacle and one side is visibly more open.  The CARLA public PID
        # remains the only steering controller.
        if (
            self._ordinary_lidar_safety
            and front is not None
            and front < 5.5
        ):
            left_clearance = self._clearance(left)
            right_clearance = self._clearance(right)
            if max(left_clearance, right_clearance) > 4.0:
                self._ordinary_pass_offset = (
                    1.5 if right_clearance >= left_clearance else -1.5
                )
                self._ordinary_pass_clear_frames = 0
            else:
                self._ordinary_pass_clear_frames = 0
        elif self._ordinary_pass_offset != 0.0:
            self._ordinary_pass_clear_frames += 1
            if self._ordinary_pass_clear_frames > 40:
                self._ordinary_pass_offset = 0.0
                self._ordinary_pass_clear_frames = 0

        steer = self._route_pid_steer(tick_data, model_control.steer)
        throttle = float(
            np.clip(
                0.18
                + 0.12 * (self._ordinary_target_speed - speed),
                0.0,
                self._ordinary_throttle_limit,
            )
        )
        brake = 0.0

        if self._ordinary_lidar_safety and front is not None:
            if front < 2.5:
                throttle, brake = 0.0, 1.0
            elif front < 4.0 and self._ordinary_pass_offset == 0.0:
                throttle, brake = 0.0, 0.65
            elif front < 5.5:
                throttle = min(throttle, 0.20)

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
                "throttle={:.4f} brake={:.4f} front={} left={} right={} "
                "offset={:.2f}".format(
                    self.step,
                    speed,
                    float(steer),
                    throttle,
                    brake,
                    front,
                    left,
                    right,
                    self._ordinary_pass_offset,
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
    return "OrdinaryRoutePIDSixCameraAgent"
