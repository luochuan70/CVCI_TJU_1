"""Ordinary-route four-camera recovery adapter.

CARLA 0.9.15 crashes on a subset of the ordinary routes when the frozen
six-camera suite is kept alive.  This adapter leaves the frozen model and
control path untouched, physically requests only the four views through
CAM_BACK, then fills the two rear-side model slots from the stable rear view.
It is not used by the 144-route frozen agent.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as _FrozenDriveTransformerAgent,
)


class FourCameraRecoveryAgent(_FrozenDriveTransformerAgent):
    _PHYSICAL_CAMERAS = {
        "CAM_FRONT",
        "CAM_FRONT_LEFT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK",
    }

    def setup(self, path_to_conf_file):
        super().setup(path_to_conf_file)
        self._active_physical_cameras = set(self._PHYSICAL_CAMERAS)
        camera_count = int(os.environ.get("CVCI_RECOVERY_CAMERA_COUNT", "4"))
        if camera_count == 3:
            self._active_physical_cameras.discard("CAM_BACK")
        elif camera_count == 1:
            self._active_physical_cameras = {"CAM_FRONT"}
        elif camera_count != 4:
            raise ValueError("CVCI_RECOVERY_CAMERA_COUNT must be 1, 3, or 4")
        if os.environ.get("CVCI_RECOVERY_CAMERA_LOWRES", "0") == "1":
            width = int(os.environ.get("CVCI_RECOVERY_CAMERA_WIDTH", "800"))
            height = int(os.environ.get("CVCI_RECOVERY_CAMERA_HEIGHT", "450"))
            for camera_name in self._active_physical_cameras:
                sensor = dict(self.all_sensors[camera_name])
                sensor["width"] = width
                sensor["height"] = height
                self.all_sensors[camera_name] = sensor

    def sensors(self):
        requested = super().sensors()
        active = getattr(self, "_active_physical_cameras", self._PHYSICAL_CAMERAS)
        return [sensor for sensor in requested if sensor["id"] in active or sensor["id"] not in self.cameras]

    def tick(self, input_data):
        tick_data = super().tick(input_data)
        self._recovery_last_tick = tick_data
        return tick_data

    def run_step(self, input_data, timestamp):
        adapted = dict(input_data)
        source = adapted.get("CAM_FRONT")
        if source is None:
            source = adapted.get("CAM_BACK")
        if source is None:
            raise RuntimeError("No camera is available for the recovery model slots")
        source_frame, source_image = source
        rear = adapted.get("CAM_BACK", source)
        rear_frame, rear_image = rear
        for camera_name in self.cameras:
            if camera_name in adapted:
                continue
            if camera_name == "CAM_BACK_LEFT":
                adapted[camera_name] = (
                    rear_frame,
                    np.ascontiguousarray(rear_image[:, ::-1, :]),
                )
            else:
                adapted[camera_name] = (source_frame, np.ascontiguousarray(source_image))
        for camera_name in self.cameras:
            packet = adapted.get(camera_name)
            if packet is None or len(packet) != 2:
                continue
            camera_frame, camera_image = packet
            if getattr(camera_image, "shape", (0, 0))[:2] != (900, 1600):
                camera_image = cv2.resize(camera_image, (1600, 900), interpolation=cv2.INTER_LINEAR)
                adapted[camera_name] = (camera_frame, camera_image)
        control = super().run_step(adapted, timestamp)
        if (
            os.environ.get("CVCI_RECOVERY_EMPTY_CUDA_CACHE", "0") == "1"
            and torch.cuda.is_available()
        ):
            torch.cuda.empty_cache()
        return control


def get_entry_point():
    return "FourCameraRecoveryAgent"
