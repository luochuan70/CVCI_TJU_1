"""Ordinary-route recovery adapter for CARLA sensor-memory crashes.

The frozen 0629 agent remains unchanged. This adapter asks CARLA for half-size
camera frames and restores them to the model's expected 1600x900 shape before
calling the frozen agent. It is only for ordinary-route crash recovery.
"""
from __future__ import annotations

import cv2

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as _FrozenDriveTransformerAgent,
)


class LowResRecoveryAgent(_FrozenDriveTransformerAgent):
    """Keep the frozen model/control path while lowering CARLA camera load."""

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        for camera_name in self.cameras:
            sensor = dict(self.all_sensors[camera_name])
            sensor["width"] = 800
            sensor["height"] = 450
            self.all_sensors[camera_name] = sensor

    def run_step(self, input_data, timestamp):
        adapted = dict(input_data)
        for camera_name in self.cameras:
            packet = input_data.get(camera_name)
            if packet is None or len(packet) != 2:
                continue
            frame, image = packet
            if getattr(image, "shape", (0, 0))[:2] != (900, 1600):
                image = cv2.resize(image, (1600, 900), interpolation=cv2.INTER_LINEAR)
                adapted[camera_name] = (frame, image)
        return super().run_step(adapted, timestamp)


def get_entry_point():
    return "LowResRecoveryAgent"

