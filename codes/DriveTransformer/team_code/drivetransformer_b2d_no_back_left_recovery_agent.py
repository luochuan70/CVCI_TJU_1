"""Ordinary-route recovery adapter for the CARLA CAM_BACK_LEFT crash.

The 0629 agent and its model/control code are imported unchanged.  Only the
ordinary-route sensor contract is adapted: CARLA does not spawn
CAM_BACK_LEFT, and the missing sixth model view is reconstructed by
horizontally mirroring CAM_BACK_RIGHT.  The 144-route frozen agent never
imports this module.
"""
from __future__ import annotations

import numpy as np

from DriveTransformer.team_code.drivetransformer_b2d_agent import (
    DriveTransformerAgent as _FrozenDriveTransformerAgent,
)


class NoBackLeftRecoveryAgent(_FrozenDriveTransformerAgent):
    """Keep six model slots while physically using five CARLA cameras."""

    _MISSING_CAMERA = "CAM_BACK_LEFT"

    def sensors(self):
        requested = super().sensors()
        return [sensor for sensor in requested if sensor["id"] != self._MISSING_CAMERA]

    def run_step(self, input_data, timestamp):
        adapted = dict(input_data)
        if self._MISSING_CAMERA not in adapted:
            source = adapted.get("CAM_BACK_RIGHT") or adapted.get("CAM_BACK")
            if source is None:
                raise RuntimeError("No rear camera is available to reconstruct CAM_BACK_LEFT")
            frame, image = source
            mirrored = np.ascontiguousarray(image[:, ::-1, :])
            adapted[self._MISSING_CAMERA] = (frame, mirrored)
        return super().run_step(adapted, timestamp)


def get_entry_point():
    return "NoBackLeftRecoveryAgent"
