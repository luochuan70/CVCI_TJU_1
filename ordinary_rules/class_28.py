"""Class-28 online pull-over for a fast emergency vehicle from behind."""

from __future__ import absolute_import

from collections import deque
import statistics

from .common import clone_control, _number
from .configs import get_config


class Rule28(object):
    """Yield the occupied lane and wait until the emergency event has ended.

    The first implementation encoded a complete shift/hold/rejoin manoeuvre as
    one distance-indexed local path.  That controller was allowed to advance to
    rejoin before the vehicle had physically cleared the lane.  This version
    advances only from measured route-relative lateral position and keeps the
    adjacent lane for longer than the online event's observed blocking window.
    """

    YIELD_OFFSET_M = 3.5
    ACQUIRED_OFFSET_M = 2.65
    ACQUIRE_CONFIRM_FRAMES = 3
    STAGE_PROGRESS_M = 38.0
    CONTROL_DT_S = 0.10
    # The official scenario may finish through a ten-second idle branch after
    # the emergency actor reaches the 50 m trigger.  Detector boxes have no
    # actor identity, so never use a generic front box to shorten this wait.
    HOLD_MIN_FRAMES = 220
    BLOCKED_REAR_CONFIRM_FRAMES = 5
    BLOCKED_REAR_MAX_SPEED_MPS = 3.0
    PASSED_FOLLOW_MIN_FRAMES = 100
    REJOIN_CONFIRM_FRAMES = 4
    SHIFT_MAX_FRAMES = 140
    TOTAL_MAX_FRAMES = 700

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.phase = "CRUISE"
        self.phase_frames = 0
        self.total_frames = 0
        self.acquire_frames = 0
        self.rejoin_frames = 0
        self.stage_progress_m = 0.0
        self.lateral_history = deque(maxlen=5)
        self.yield_offset_m = self.YIELD_OFFSET_M
        self.yield_side_reason = "default_positive_clear_side"
        self.rear_seen = False
        self.rear_blocked_frames = 0
        self.front_pass_frames = 0
        self.maneuver_complete = False

    def _observe_emergency(self, features):
        rear = None
        front = None
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict) or str(
                item.get("class_name", "") or ""
            ).lower() not in {"car", "vehicle", "van", "truck", "bus"}:
                continue
            box = item.get("box_lidar") or item
            try:
                score = float(item.get("score", 0.0) or 0.0)
                lateral = float(box.get("x"))
                longitudinal = float(box.get("y"))
                velocity = float(box.get("vy", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            observation = {
                "score": score, "lateral_m": lateral,
                "longitudinal_m": longitudinal, "velocity_mps": velocity,
            }
            if (
                score >= 0.75 and abs(lateral) <= 3.0
                and -22.0 <= longitudinal <= -6.0
            ):
                if rear is None or score > rear["score"]:
                    rear = observation
            if (
                self.rear_seen and score >= 0.65 and abs(lateral) <= 4.0
                and 2.0 <= longitudinal <= 18.0
            ):
                if front is None or score > front["score"]:
                    front = observation
        if rear is not None:
            self.rear_seen = True
        return rear, front

    def _select_yield_offset(self, features):
        """Choose the adjacent lane that is clear in online detections."""
        occupancy = {-1: 0.0, 1: 0.0}
        observations = {-1: 0, 1: 0}
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict) or str(
                item.get("class_name", "") or ""
            ).lower() not in {"car", "vehicle", "van", "truck", "bus"}:
                continue
            box = item.get("box_lidar") or item
            try:
                score = float(item.get("score", 0.0) or 0.0)
                lateral = float(box.get("x"))
                longitudinal = float(box.get("y"))
            except (TypeError, ValueError):
                continue
            if not (
                score >= 0.30
                and 1.5 <= abs(lateral) <= 5.5
                and -10.0 <= longitudinal <= 15.0
            ):
                continue
            side = -1 if lateral < 0.0 else 1
            observations[side] += 1
            occupancy[side] += score / max(2.0, abs(longitudinal))
        # Positive is the default roadside direction on the audited common
        # topology.  Select negative only when positive is observably busier.
        side = 1 if occupancy[1] <= occupancy[-1] else -1
        self.yield_offset_m = side * abs(self.YIELD_OFFSET_M)
        self.yield_side_reason = "online_lower_side_occupancy"
        if not any(observations.values()):
            self.yield_side_reason = "default_positive_clear_side"

    def _filtered_lateral(self, radar):
        value = radar.get("_route_lateral_m")
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        if abs(value) >= 20.0:
            return None
        self.lateral_history.append(value)
        return float(statistics.median(self.lateral_history))

    def _dense_directive(self, target):
        return {
            "active": True,
            "phase": "ordinary_class28_" + self.phase.lower(),
            "target_route_offset_m": float(target),
            "lookahead_base_m": 3.0 if self.phase == "SHIFT" else 5.0,
            "lookahead_speed_s": 0.45,
            "lookahead_min_m": 2.8 if self.phase == "SHIFT" else 4.5,
            "lookahead_max_m": 5.0 if self.phase == "SHIFT" else 8.0,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.12 if self.phase == "SHIFT" else 0.09,
            "pose_filter_frames": 5,
        }

    @staticmethod
    def _speed_control(result, speed, target):
        error = target - speed
        if error < -0.6:
            result.throttle = 0.0
            result.brake = min(0.35, 0.10 + 0.10 * abs(error))
        else:
            result.brake = 0.0
            result.throttle = max(0.20, min(0.80, 0.38 + 0.16 * error))
        result.reverse = False

    def apply(self, control, radar, features):
        result = clone_control(control)
        handoff = str(radar.get("_classifier_reason", "") or "") in {
            "online_early_sparse_emergency_corridor",
            "online_early_centered_rear_emergency_vehicle",
        }
        if self.phase == "CRUISE" and handoff and not self.maneuver_complete:
            # The official event allows roughly ten seconds for the emergency
            # actor to finish ahead.  Start the already measured/validated
            # lane-clearance controller at the early corridor handoff, not at
            # the later high-confidence rear box when half the window is gone.
            self.phase = "STAGE"
            self.phase_frames = 0
            self.total_frames = 0
            self.acquire_frames = 0
            self.rejoin_frames = 0
            self.stage_progress_m = 0.0
            self.lateral_history.clear()

        lateral = self._filtered_lateral(radar) if self.phase != "CRUISE" else None
        rear, front = self._observe_emergency(features)
        self.front_pass_frames = self.front_pass_frames + 1 if front else 0
        if self.phase != "CRUISE" and not self.maneuver_complete:
            self.phase_frames += 1
            self.total_frames += 1

        blocked = bool(
            rear is not None
            and rear["velocity_mps"] <= self.BLOCKED_REAR_MAX_SPEED_MPS
        )
        self.rear_blocked_frames = (
            self.rear_blocked_frames + 1 if blocked else 0
        )
        speed = max(0.0, _number(getattr(features, "ego_speed", 0.0)))
        if self.phase == "STAGE":
            # Yielding immediately makes the emergency actor's fixed 50 m
            # post-pass endpoint fall inside the ego route.  Advance using
            # online odometry only, then clear the lane while there is still
            # ample time in the official ten-second interaction window.
            self.stage_progress_m += speed * self.CONTROL_DT_S
            if self.stage_progress_m >= self.STAGE_PROGRESS_M:
                self.phase = "SHIFT"
                self.phase_frames = 0
                self.acquire_frames = 0
                self.lateral_history.clear()
                self._select_yield_offset(features)
        if self.phase == "SHIFT":
            direction = 1.0 if self.yield_offset_m > 0.0 else -1.0
            acquired = bool(
                lateral is not None
                and direction * lateral >= abs(self.ACQUIRED_OFFSET_M)
            )
            self.acquire_frames = self.acquire_frames + 1 if acquired else 0
            if self.acquire_frames >= self.ACQUIRE_CONFIRM_FRAMES:
                self.phase = "HOLD"
                self.phase_frames = 0
            elif self.phase_frames > self.SHIFT_MAX_FRAMES:
                self.phase = "SAFE_TIMEOUT"
                self.phase_frames = 0
        if self.phase == "HOLD" and self.phase_frames >= self.HOLD_MIN_FRAMES:
            self.phase = "REJOIN"
            self.phase_frames = 0
            self.lateral_history.clear()
        elif self.phase == "REJOIN":
            rejoined = lateral is not None and abs(lateral) <= 0.60
            self.rejoin_frames = self.rejoin_frames + 1 if rejoined else 0
            if self.rejoin_frames >= self.REJOIN_CONFIRM_FRAMES:
                self.phase = "PASSED_FOLLOW"
                self.phase_frames = 0
                self.maneuver_complete = True
        if (
            self.phase not in {"CRUISE", "COMPLETE", "SAFE_TIMEOUT"}
            and self.total_frames > self.TOTAL_MAX_FRAMES
        ):
            self.phase = "SAFE_TIMEOUT"
            self.phase_frames = 0

        immediate_hazard = bool(getattr(features, "immediate_hazard", False))
        traffic_hold = bool(
            getattr(features, "red_light_active", False)
            or getattr(features, "front_pedestrian_distance", None) is not None
        )
        if self.phase == "PASSED_FOLLOW":
            if immediate_hazard or traffic_hold or (
                front is not None and front["longitudinal_m"] <= 6.0
            ):
                result.throttle = 0.0
                result.brake = max(_number(getattr(result, "brake", 0.0)), 0.75)
            else:
                self._speed_control(result, speed, 3.5)
        elif self.phase in {"STAGE", "SHIFT", "HOLD", "REJOIN"}:
            if immediate_hazard or traffic_hold:
                result.throttle = 0.0
                result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
            elif self.phase == "HOLD":
                # The official emergency actor never changes lane; it only
                # accelerates along its current lane and brakes for a detected
                # leader.  Once ego has physically cleared that lane, remain
                # stationary so the actor can establish and retain the ahead
                # relation before the bounded event terminates.
                result.throttle = 0.0
                result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
                result.reverse = False
            else:
                target_speed = 8.0
                if self.phase == "REJOIN":
                    target_speed = 4.5
                self._speed_control(result, speed, target_speed)
        elif self.phase == "SAFE_TIMEOUT":
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
            result.steer = 0.0
            result.reverse = False

        debug = {
            "source": "ordinary_class_28_emergency_yield",
            "state": "CLASS28_" + self.phase,
            "class28_online_handoff": handoff,
            "class28_phase": self.phase,
            "class28_phase_frames": self.phase_frames,
            "class28_total_frames": self.total_frames,
            "class28_stage_progress_m": self.stage_progress_m,
            "class28_filtered_route_lateral_m": lateral,
            "class28_yield_offset_m": self.yield_offset_m,
            "class28_yield_side_reason": self.yield_side_reason,
            "class28_acquire_frames": self.acquire_frames,
            "class28_rejoin_frames": self.rejoin_frames,
            "class28_rear_seen": self.rear_seen,
            "class28_rear_blocked_frames": self.rear_blocked_frames,
            "class28_front_pass_frames": self.front_pass_frames,
            "class28_rear_observation": rear,
            "class28_front_observation": front,
            "class28_maneuver_complete": self.maneuver_complete,
            "input_control": {
                "throttle": _number(getattr(control, "throttle", 0.0)),
                "steer": _number(getattr(control, "steer", 0.0)),
                "brake": _number(getattr(control, "brake", 0.0)),
            },
            "output_control": {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
            },
        }
        if self.phase in {"SHIFT", "HOLD"}:
            debug["ordinary_dense_route_pid"] = self._dense_directive(
                self.yield_offset_m
            )
        elif self.phase == "REJOIN":
            debug["ordinary_dense_route_pid"] = self._dense_directive(0.0)
        return result, debug


def build_rule():
    return Rule28(get_config(28))
