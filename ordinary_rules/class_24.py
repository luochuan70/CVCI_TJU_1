"""Class-24 longitudinal yield for a vehicle cutting in from a parked row."""

from __future__ import absolute_import

import math

from .common import clone_control
from .configs import get_config


class Rule24(object):
    """Keep the learned route steering and yield to the actual cut-in body.

    StaticCutIn is not a static road blockage: one member of a parked side
    row enters the ego lane and subsequently drives away.  A lateral bypass
    around the whole row is therefore the wrong manoeuvre.  This rule owns
    longitudinal safety only; the learned controller continues to steer the
    official route.
    """

    HOLD_DISTANCE_M = 10.0
    RELEASE_DISTANCE_M = 13.0
    CLEAR_CONFIRM_FRAMES = 5
    MOTION_CONFIRM_FRAMES = 2

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self._yield_hold = False
        self._clear_frames = 0
        self._motion_frames = 0

    def reset(self):
        self._yield_hold = False
        self._clear_frames = 0
        self._motion_frames = 0

    @staticmethod
    def _vehicle(item):
        return (
            isinstance(item, dict)
            and str(item.get("class_name", "") or "").lower()
            in {"car", "vehicle", "van", "truck", "bus", "motorcycle"}
        )

    def _nearest_corridor_vehicle(self, features):
        candidates = []
        # Detector coordinates follow the CARLA/LiDAR convention used by the
        # adapter: x is forward and y is lateral.  Class24's parked row is
        # static; acquire only a vehicle with measured motion as it cuts into
        # the ego corridor.  A generic static front box can be a parked-row
        # projection and must not create a permanent hold before the official
        # adversary is triggered.
        for item in list(getattr(features, "tracked_objects", None) or []):
            if not self._vehicle(item):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                distance = float(item.get("x"))
                lateral = float(item.get("y"))
                speed = math.hypot(
                    float(item.get("vx", 0.0) or 0.0),
                    float(item.get("vy", 0.0) or 0.0),
                )
            except (TypeError, ValueError):
                continue
            if (
                score >= 0.55
                and math.isfinite(lateral)
                and math.isfinite(distance)
                and abs(lateral) <= 2.2
                and 0.0 < distance <= 35.0
                and speed >= 3.0
            ):
                candidates.append(distance)
        # Raw per-frame detector velocity is deliberately not used here: the
        # failed formal trace assigned an 8 m/s lateral velocity to the same
        # physically stationary parked body.  The temporal tracker is the
        # online motion evidence required for acquisition and retention.
        return min(candidates) if candidates else None

    def apply(self, control, radar, features):
        result = clone_control(control)
        nearest = self._nearest_corridor_vehicle(features)
        moving_close = bool(
            nearest is not None and nearest <= self.HOLD_DISTANCE_M
        )
        self._motion_frames = (
            self._motion_frames + 1 if moving_close else 0
        )
        if (
            not self._yield_hold
            and self._motion_frames >= self.MOTION_CONFIRM_FRAMES
        ):
            self._yield_hold = True
            self._clear_frames = 0
        if self._yield_hold:
            clear = nearest is None or nearest >= self.RELEASE_DISTANCE_M
            self._clear_frames = self._clear_frames + 1 if clear else 0
            if self._clear_frames >= self.CLEAR_CONFIRM_FRAMES:
                self._yield_hold = False
                self._clear_frames = 0

        state = "CLASS24_CUTIN_MONITOR"
        if self._yield_hold:
            result.throttle = 0.0
            result.brake = max(float(getattr(result, "brake", 0.0) or 0.0), 0.90)
            result.steer = 0.0
            if hasattr(result, "reverse"):
                result.reverse = False
            state = "CLASS24_YIELD_CUTIN"
        return result, {
            "state": state,
            "class24_yield_hold": bool(self._yield_hold),
            "class24_clear_frames": int(self._clear_frames),
            "class24_motion_frames": int(self._motion_frames),
            "class24_nearest_corridor_vehicle_m": nearest,
            "output_control": {
                "throttle": float(getattr(result, "throttle", 0.0) or 0.0),
                "steer": float(getattr(result, "steer", 0.0) or 0.0),
                "brake": float(getattr(result, "brake", 0.0) or 0.0),
            },
        }


def build_rule():
    return Rule24(get_config(24))
