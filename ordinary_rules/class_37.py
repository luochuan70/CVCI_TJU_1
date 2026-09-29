"""Class-37 dynamic-crossing controller.

The learned controller already reacts correctly to the crossing actor when it
keeps ownership of the online route.  Class37 therefore deliberately owns no
lateral detour: it preserves the model command after sensor-only rejection of
an unsupported Class30 door plan or a false Class04 construction plan.
"""

from .common import _number, clone_control, feature_value
from .configs import get_config


class Rule37(object):
    """Preserve learned route control after rejecting an unsupported detour."""

    ACQUIRE_DISTANCE_M = 5.5
    RADAR_ONLY_DISTANCE_M = 3.0
    CLEAR_DISTANCE_M = 6.5
    CLEAR_CONFIRM_FRAMES = 15
    STATIC_TIMEOUT_FRAMES = 100
    ARMED_CREEP_TIMEOUT_FRAMES = 600
    ARMED_CREEP_TARGET_MPS = 1.8

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self._crossing_hold = False
        self._crossing_completed = False
        self._hold_frames = 0
        self._clear_frames = 0
        self._trigger_reason = ""
        self._awaiting_crossing_event = False
        self._armed_creep = False
        self._armed_creep_frames = 0

    def reset(self):
        self._crossing_hold = False
        self._crossing_completed = False
        self._hold_frames = 0
        self._clear_frames = 0
        self._trigger_reason = ""
        self._awaiting_crossing_event = False
        self._armed_creep = False
        self._armed_creep_frames = 0

    def prime_crossing_hold(self, reason, await_event=False):
        """Carry a multi-frame pre-handoff stop into Class37 ownership."""
        if self._crossing_completed or self._crossing_hold:
            return
        self._crossing_hold = True
        self._hold_frames = 0
        self._clear_frames = 0
        self._trigger_reason = str(reason or "pre_handoff_crossing_evidence")
        self._awaiting_crossing_event = bool(await_event)

    def apply(self, control, radar, features):
        result = clone_control(control)
        speed = feature_value(features, "ego_speed", radar.get("ego_speed", 0.0))
        # The actor enters laterally from behind the blocker.  Its first
        # reliable return is therefore the nearest all-angle depth, not the
        # narrow +/-22 degree front slice.
        nearest_depth = _number(radar.get("depth_min", 0.0))
        front_dynamic = _number(radar.get("front_dynamic_count", 0.0))
        front_closing = _number(radar.get("front_closing_count", 0.0))
        front_ttc = _number(radar.get("front_ttc", 0.0))
        input_brake = _number(getattr(control, "brake", 0.0))

        model_range_brake = bool(
            0.0 < nearest_depth <= self.ACQUIRE_DISTANCE_M
            and input_brake >= 0.50
        )
        corroborated_radar_hazard = bool(
            0.0 < nearest_depth <= self.RADAR_ONLY_DISTANCE_M
            and (
                front_dynamic >= 1.0
                or front_closing >= 2.0
                or 0.0 < front_ttc <= 1.5
            )
        )
        crossing_event_observed = bool(
            model_range_brake or corroborated_radar_hazard
        )
        if self._awaiting_crossing_event and crossing_event_observed:
            self._awaiting_crossing_event = False
            self._armed_creep = False
            self._armed_creep_frames = 0
            self._clear_frames = 0
        if (
            not self._crossing_completed
            and not self._crossing_hold
            and (model_range_brake or corroborated_radar_hazard)
        ):
            self._crossing_hold = True
            self._hold_frames = 0
            self._clear_frames = 0
            self._trigger_reason = (
                "model_range_brake"
                if model_range_brake else "corroborated_radar_hazard"
            )

        state = "CLASS37_DYNAMIC_CROSSING_MONITOR"
        if self._crossing_hold:
            self._hold_frames += 1
            radar_clear = bool(
                nearest_depth >= self.CLEAR_DISTANCE_M
                and front_dynamic < 1.0
                and front_closing < 1.0
            )
            stopped_clear = bool(speed <= 0.35 and radar_clear)
            static_timeout_clear = bool(
                self._hold_frames >= self.STATIC_TIMEOUT_FRAMES
                and front_dynamic < 1.0
                and front_closing < 1.0
            )
            self._clear_frames = (
                self._clear_frames + 1
                if stopped_clear else 0
            )
            armed_creep_ready = bool(
                self._awaiting_crossing_event
                and (
                    self._clear_frames >= self.CLEAR_CONFIRM_FRAMES
                    or static_timeout_clear
                )
            )
            crossing_clear = bool(
                not self._awaiting_crossing_event
                and (
                    self._clear_frames >= self.CLEAR_CONFIRM_FRAMES
                    or static_timeout_clear
                )
            )
            if armed_creep_ready:
                self._crossing_hold = False
                self._armed_creep = True
                self._armed_creep_frames = 0
                self._clear_frames = 0
                state = "CLASS37_ARMED_CREEP"
            elif crossing_clear:
                self._crossing_hold = False
                self._crossing_completed = True
                self._armed_creep = False
                self._clear_frames = 0
                state = "CLASS37_CROSSING_CLEARED"
            else:
                result.throttle = 0.0
                result.brake = 1.0
                if hasattr(result, "reverse"):
                    result.reverse = False
                state = "CLASS37_HOLD_CROSSING"

        if self._armed_creep and not self._crossing_hold:
            self._armed_creep_frames += 1
            if self._armed_creep_frames >= self.ARMED_CREEP_TIMEOUT_FRAMES:
                result.throttle = 0.0
                result.brake = 1.0
                if hasattr(result, "reverse"):
                    result.reverse = False
                state = "CLASS37_CREEP_TIMEOUT_SAFE_STOP"
            else:
                # Preserve the learned route/steering while approaching the
                # occluded crossing slowly enough for the existing bounded
                # hazard acquisition to stop before the conflict point.
                if input_brake >= 0.20:
                    result.throttle = 0.0
                elif speed < self.ARMED_CREEP_TARGET_MPS - 0.15:
                    speed_error = self.ARMED_CREEP_TARGET_MPS - speed
                    result.throttle = min(
                        0.60, max(0.28, 0.28 + 0.18 * speed_error)
                    )
                    result.brake = 0.0
                elif speed > self.ARMED_CREEP_TARGET_MPS + 0.10:
                    result.throttle = 0.0
                    result.brake = max(
                        _number(getattr(result, "brake", 0.0)), 0.35
                    )
                else:
                    result.throttle = min(
                        _number(getattr(result, "throttle", 0.0)), 0.10
                    )
                if hasattr(result, "reverse"):
                    result.reverse = False
                state = "CLASS37_ARMED_CREEP"

        return result, {
            "state": state,
            "class37_model_owned_route": True,
            "class37_handoff_reason": str(
                radar.get("_classifier_reason", "") or ""
            ),
            "class37_crossing_hold": bool(self._crossing_hold),
            "class37_crossing_completed": bool(self._crossing_completed),
            "class37_hold_frames": int(self._hold_frames),
            "class37_clear_frames": int(self._clear_frames),
            "class37_trigger_reason": str(self._trigger_reason),
            "class37_awaiting_crossing_event": bool(
                self._awaiting_crossing_event
            ),
            "class37_armed_creep": bool(self._armed_creep),
            "class37_armed_creep_frames": int(self._armed_creep_frames),
            "class37_nearest_depth_m": float(nearest_depth),
            "class37_front_dynamic_count": float(front_dynamic),
            "class37_front_closing_count": float(front_closing),
            "class37_front_ttc_s": float(front_ttc),
            "class37_ego_speed_mps": float(speed),
            "output_control": {
                "throttle": float(getattr(result, "throttle", 0.0) or 0.0),
                "steer": float(getattr(result, "steer", 0.0) or 0.0),
                "brake": float(getattr(result, "brake", 0.0) or 0.0),
            },
        }


def build_rule():
    return Rule37(get_config(37))
