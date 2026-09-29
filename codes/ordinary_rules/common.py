"""Bounded control primitives shared by category-specific entries."""

from __future__ import absolute_import

import copy
import math
from collections import deque


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def clone_control(control):
    # The harness unit tests use a small duck-typed control under the base
    # Python interpreter.  Importing the Python-3.7 CARLA egg there can
    # segfault before Python can raise an import error, so only construct a
    # native CARLA control for an object that already came from the CARLA
    # module.  The formal evaluator still takes the native branch.
    control_module = getattr(type(control), "__module__", "")
    if not str(control_module).split(".", 1)[0] == "carla":
        return copy.copy(control)
    try:
        import carla

        result = carla.VehicleControl()
        for name in ("throttle", "steer", "brake", "hand_brake", "reverse", "manual_gear_shift", "gear"):
            if hasattr(control, name):
                setattr(result, name, getattr(control, name))
        return result
    except Exception:
        return copy.copy(control)


def set_control(control, throttle=None, steer=None, brake=None):
    result = clone_control(control)
    if throttle is not None and hasattr(result, "throttle"):
        result.throttle = max(0.0, min(1.0, _number(throttle)))
    if steer is not None and hasattr(result, "steer"):
        result.steer = max(-1.0, min(1.0, _number(steer)))
    if brake is not None and hasattr(result, "brake"):
        result.brake = max(0.0, min(1.0, _number(brake)))
    return result


def feature_value(features, name, default=0.0):
    if isinstance(features, dict):
        return _number(features.get(name, default), default)
    return _number(getattr(features, name, default), default)


def optional_feature_value(features, name):
    if isinstance(features, dict):
        value = features.get(name)
    else:
        value = getattr(features, name, None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


class _TemporalApproachGate(object):
    """Require corroborated radar approach evidence across adjacent frames."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.history = deque(maxlen=3)

    def reset(self):
        self.history.clear()

    def observe(self, radar):
        dynamic = _number(radar.get("front_dynamic_count", 0.0))
        closing = _number(radar.get("front_closing_count", 0.0))
        trend = _number(radar.get("front_depth_trend", 0.0))
        ttc = _number(radar.get("front_ttc", 0.0))
        ttc_limit = _number(self.cfg.get("ttc_s"), 1.8)
        trend_limit = _number(self.cfg.get("approach_trend_m"), 0.15)
        signal = bool(
            closing >= 1.0
            or (dynamic >= 2.0 and trend > trend_limit)
        )
        self.history.append({
            "signal": signal,
            "ttc": 0.0 < ttc <= ttc_limit,
            "cluster": bool(
                dynamic >= 2.0
                and _number(radar.get("front_close_count", 0.0)) >= 2.0
                and trend > trend_limit
                and _number(radar.get("front_min", 0.0)) > 0.0
                and _number(radar.get("front_min", 0.0)) <= _number(
                    self.cfg.get("approaching_cluster_distance_m"), 6.0
                )
            ),
        })
        recent = list(self.history)
        required = max(1, int(self.cfg.get("approach_confirm_frames", 2)))
        confirmed = len(recent) >= required and sum(
            1 for item in recent[-3:] if item["signal"]
        ) >= required
        ttc_confirmed = confirmed and any(
            item["signal"] and item["ttc"] for item in recent[-3:]
        )
        cluster_confirmed = confirmed and sum(
            1 for item in recent[-3:] if item["cluster"]
        ) >= required
        return {
            "signal": signal,
            "confirmed": confirmed,
            "ttc_confirmed": ttc_confirmed,
            "cluster_confirmed": cluster_confirmed,
            "frames": len(recent),
            "signal_frames": sum(1 for item in recent[-3:] if item["signal"]),
        }


def radar_safety_action(
    control,
    radar,
    features,
    cfg,
    source,
    approach_gate=None,
    approach_state=None,
    static_side_pass_active=False,
):
    """Apply only monotonic braking/throttle caps and a bounded side bias."""

    result = clone_control(control)
    current_throttle = _number(getattr(control, "throttle", 0.0))
    current_steer = _number(getattr(control, "steer", 0.0))
    current_brake = _number(getattr(control, "brake", 0.0))
    speed = feature_value(features, "ego_speed", radar.get("ego_speed", 0.0))
    front_min = _number(radar.get("front_min", 0.0))
    front_close = _number(radar.get("front_close_count", 0.0))
    front_dynamic = _number(radar.get("front_dynamic_count", 0.0))
    ttc = _number(radar.get("front_ttc", 0.0))
    depth_trend = _number(radar.get("front_depth_trend", 0.0))
    point_count = _number(radar.get("point_count", 0.0))
    evidence = point_count >= 2.0 and front_min > 0.0
    approach = approach_state or (
        approach_gate.observe(radar) if approach_gate is not None else {
            "signal": False,
            "confirmed": False,
            "ttc_confirmed": False,
            "cluster_confirmed": False,
            "frames": 0,
            "signal_frames": 0,
        }
    )
    raw_emergency = evidence and (
        front_min <= _number(cfg.get("emergency_distance_m"), 5.0)
        or approach["ttc_confirmed"]
        or (
            approach["cluster_confirmed"]
            and front_close >= 3.0
            and depth_trend > 0.35
        )
    )
    static_only_emergency = bool(
        evidence
        and front_min <= _number(cfg.get("emergency_distance_m"), 5.0)
        and front_dynamic < 2.0
        and _number(radar.get("front_closing_count", 0.0)) < 1.0
        and not approach["ttc_confirmed"]
        and not approach["cluster_confirmed"]
    )
    release_brake = _number(
        cfg.get("static_side_pass_release_brake"), 0.12
    )
    suppress_static_emergency = bool(
        static_side_pass_active
        and static_only_emergency
        and current_brake <= release_brake
    )
    emergency = bool(raw_emergency and not suppress_static_emergency)
    # A range-only caution is unsafe for this sensor: ordinary CARLA radar
    # returns road furniture, lane-side surfaces, and distant parked actors
    # even when they are not a threat to the current trajectory.  In the
    # first live A/B this condition held for a routine ~6--9 m return while
    # the ego vehicle was stationary, so the fallback kept applying brake and
    # the official evaluator eventually marked the route blocked.  Require a
    # temporal approach signal (TTC or a close, repeatedly approaching
    # cluster) for *caution*.  Keep the shorter emergency-distance guard as a
    # last-resort collision boundary.
    ttc_caution = approach["ttc_confirmed"] and ttc > 0.0 and ttc <= _number(
        cfg.get("ttc_s"), 1.8
    )
    approaching_cluster = approach["cluster_confirmed"]
    caution = evidence and (ttc_caution or approaching_cluster)
    desired_brake = current_brake
    desired_throttle = current_throttle
    state = "NORMAL"
    if emergency and speed > 0.2:
        desired_brake = max(desired_brake, _number(cfg.get("brake_emergency"), 0.70))
        desired_throttle = min(desired_throttle, _number(cfg.get("throttle_cap_emergency"), 0.05))
        state = "RADAR_EMERGENCY_BRAKE"
    elif caution:
        desired_brake = max(desired_brake, _number(cfg.get("brake_caution"), 0.12))
        desired_throttle = min(desired_throttle, _number(cfg.get("throttle_cap_caution"), 0.28))
        state = "RADAR_CAUTION"

    desired_steer = current_steer
    left = _number(radar.get("left_nearest", 0.0))
    right = _number(radar.get("right_nearest", 0.0))
    if (
        caution
        and _number(cfg.get("side_pass", 0.0))
        and desired_brake < 0.5
        and left > 0.0
        and right > 0.0
    ):
        margin = _number(cfg.get("side_clear_margin_m"), 2.0)
        if left > right + margin:
            desired_steer = current_steer - 0.08
        elif right > left + margin:
            desired_steer = current_steer + 0.08
        desired_steer = max(-0.45, min(0.45, desired_steer))

    if hasattr(result, "throttle"):
        result.throttle = max(0.0, min(1.0, desired_throttle))
    if hasattr(result, "brake"):
        result.brake = max(0.0, min(1.0, desired_brake))
    if hasattr(result, "steer"):
        result.steer = max(-1.0, min(1.0, desired_steer))
    if hasattr(result, "brake") and result.brake > 0.01 and hasattr(result, "throttle"):
        result.throttle = min(result.throttle, 0.05)
    debug = {
        "source": source,
        "state": state,
        "evidence": bool(evidence),
        "emergency": bool(emergency),
        "caution": bool(caution),
        "front_min": front_min,
        "front_close_count": front_close,
        "front_dynamic_count": front_dynamic,
        "front_closing_count": _number(radar.get("front_closing_count", 0.0)),
        "point_count": point_count,
        "left_nearest": left,
        "right_nearest": right,
        "ego_speed": speed,
        "front_ttc": ttc,
        "depth_trend": depth_trend,
        "ttc_caution": bool(ttc_caution),
        "approaching_cluster": bool(approaching_cluster),
        "approach_signal": bool(approach["signal"]),
        "approach_confirmed": bool(approach["confirmed"]),
        "approach_frames": approach["frames"],
        "approach_signal_frames": approach["signal_frames"],
        "ttc_confirmed": bool(approach["ttc_confirmed"]),
        "raw_emergency": bool(raw_emergency),
        "static_only_emergency": bool(static_only_emergency),
        # Preserve the standard traffic-control state in the ordinary
        # recording.  The watchdog must distinguish a legal red-light stop
        # from a rule-induced safety deadlock without changing control.
        "red_light_active": bool(getattr(features, "red_light_active", False)),
        "red_stop_distance": optional_feature_value(features, "red_stop_distance"),
        "traffic_control_kind": str(
            getattr(features, "traffic_control_kind", "") or ""
        ),
        "static_side_pass_active": bool(static_side_pass_active),
        "static_side_pass_suppressed_emergency": bool(suppress_static_emergency),
        "input_control": {
            "throttle": current_throttle, "steer": current_steer, "brake": current_brake,
        },
        "output_control": {
            "throttle": _number(getattr(result, "throttle", 0.0)),
            "steer": _number(getattr(result, "steer", 0.0)),
            "brake": _number(getattr(result, "brake", 0.0)),
        },
    }
    return result, debug


class _StaticSidePass(object):
    """Choose a clear side before a verified static obstacle becomes a block."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.candidate_side = None
        self.candidate_frames = 0
        self.active_side = None
        self.active_frames = 0
        self.clear_frames = 0
        self.block_reason = ""
        self.dynamic_override = False
        self.hold_reason = ""
        self.hard_hold_frames = 0

    def reset(self):
        self.candidate_side = None
        self.candidate_frames = 0
        self.active_side = None
        self.active_frames = 0
        self.clear_frames = 0
        self.block_reason = ""
        self.dynamic_override = False
        self.hold_reason = ""
        self.hard_hold_frames = 0

    def _candidate(self, radar, features, approach):
        if not self.cfg.get("static_side_pass", False):
            return None, "disabled"

        # CARLA's signed speedometer can report a tiny negative numerical
        # residue while the vehicle is stopped.  Treat that residue as zero;
        # do not make a configured zero-speed category gate unreachable.
        speed = max(0.0, feature_value(features, "ego_speed", radar.get("ego_speed", 0.0)))
        front_min = _number(radar.get("front_min", 0.0))
        front_close = _number(radar.get("front_close_count", 0.0))
        point_count = _number(radar.get("point_count", 0.0))
        dynamic = _number(radar.get("front_dynamic_count", 0.0))
        closing = _number(radar.get("front_closing_count", 0.0))
        left = _number(radar.get("left_nearest", 0.0))
        right = _number(radar.get("right_nearest", 0.0))
        traffic_kind = str(
            getattr(features, "traffic_control_kind", "") or ""
        ).strip().lower()

        if speed < _number(self.cfg.get("static_side_pass_min_speed_mps"), 0.8):
            return None, "speed_too_low"
        if speed > _number(self.cfg.get("static_side_pass_max_speed_mps"), 12.0):
            return None, "speed_too_high"
        if front_min < _number(self.cfg.get("static_side_pass_front_min_m"), 3.0):
            return None, "front_too_close"
        if front_min > _number(self.cfg.get("static_side_pass_front_max_m"), 8.0):
            return None, "front_too_far"
        if front_close < _number(
            self.cfg.get("static_side_pass_min_front_close_count"), 3.0
        ):
            return None, "insufficient_front_close_points"
        if point_count < _number(self.cfg.get("static_side_pass_min_points"), 6.0):
            return None, "insufficient_points"

        side_min = _number(self.cfg.get("static_side_pass_side_min_m"), 5.0)
        side_margin = _number(
            self.cfg.get("static_side_pass_side_margin_m"), 2.5
        )
        clear_left = left >= side_min and left > right + side_margin
        clear_right = right >= side_min and right > left + side_margin
        clear_side = clear_left or clear_right
        # ConstructionObstacleTwoWays can contain opposing-flow returns.  A
        # category must explicitly opt in, and the selected lateral side must
        # still satisfy the ordinary stable-clearance test.
        dynamic_override = bool(
            self.cfg.get("static_side_pass_allow_dynamic_if_clear", False)
            and clear_side
            and front_min >= _number(
                self.cfg.get("static_side_pass_dynamic_front_min_m"), 2.5
            )
        )
        self.dynamic_override = dynamic_override and bool(
            dynamic >= 2.0 or closing >= 1.0
        )
        if dynamic >= 2.0 and not dynamic_override:
            return None, "front_dynamic"
        if closing >= 1.0 and not dynamic_override:
            return None, "front_closing"
        if approach and (
            approach.get("ttc_confirmed") or approach.get("cluster_confirmed")
        ) and not dynamic_override:
            return None, "confirmed_approach"
        if traffic_kind in {
            "traffic_light", "trafficlight", "stop_sign", "stopsign"
        }:
            return None, "traffic_control"
        if bool(getattr(features, "red_light_active", False)):
            return None, "red_light"
        if bool(getattr(features, "immediate_hazard", False)):
            return None, "immediate_hazard"
        if optional_feature_value(features, "front_pedestrian_distance") is not None:
            return None, "pedestrian"

        if right >= side_min and right > left + side_margin:
            return "right", ""
        if left >= side_min and left > right + side_margin:
            return "left", ""
        return None, "no_stable_clear_side"

    def observe(self, radar, features, approach):
        candidate, reason = self._candidate(radar, features, approach)
        self.block_reason = reason
        self.hold_reason = ""

        hold_active = bool(self.cfg.get("static_side_pass_hold_active", False))
        exit_frames = max(1, int(self.cfg.get("static_side_pass_exit_frames", 3)))
        hold_frames = max(
            exit_frames,
            int(self.cfg.get("static_side_pass_hold_frames", exit_frames)),
        )
        max_frames = max(1, int(self.cfg.get("static_side_pass_max_frames", 100)))
        min_frames = max(1, int(self.cfg.get("static_side_pass_min_frames", 3)))

        # A newly observed dynamic/closing/control/VRU hazard immediately
        # invalidates an old pass direction. A merely missing side geometry
        # gets a short clear hysteresis, but only an exact current candidate
        # can suppress safety on the current frame.
        if reason in {
            "front_dynamic", "front_closing", "confirmed_approach",
            "traffic_control", "red_light", "immediate_hazard", "pedestrian",
        }:
            dynamic = _number(radar.get("front_dynamic_count", 0.0))
            closing = _number(radar.get("front_closing_count", 0.0))
            approach_confirmed = bool((approach or {}).get("confirmed", False))
            hard_hold_limit = max(
                0, int(self.cfg.get("static_side_pass_hard_hold_frames", 0))
            )
            preserve_hard_side = bool(
                hold_active
                and self.cfg.get("static_side_pass_preserve_hard_side", False)
                and reason in {
                    "front_dynamic", "front_closing", "confirmed_approach",
                }
                and (approach_confirmed or dynamic >= 2.0 or closing >= 2.0)
                and self.active_side is not None
                and self.active_frames < max_frames
                and self.hard_hold_frames < hard_hold_limit
            )
            if preserve_hard_side:
                # Keep the already selected lateral intent for a bounded
                # interval, but do not turn this into a safety override. The
                # current radar frame still reaches radar_safety_action and
                # can request emergency braking; apply_static_side_pass will
                # suppress steering/throttle while that hard evidence holds.
                self.hard_hold_frames += 1
                self.active_frames += 1
                self.clear_frames = 0
                self.hold_reason = "hard_risk_lateral_intent"
                return self.active_side

            # One isolated dynamic/closing return is common at the edge of a
            # construction cluster.  In the opt-in class-04 profile, retain
            # the committed side for a bounded interval only when the
            # temporal approach gate has not confirmed it.  The regular
            # safety primitive still sees the current dynamic evidence and
            # can brake; this branch only prevents a single noisy return from
            # deleting the lateral state.
            soft_dynamic = bool(
                hold_active
                and self.cfg.get("static_side_pass_hold_soft_dynamic", False)
                and reason in {"front_dynamic", "front_closing"}
                and not (approach or {}).get("confirmed", False)
                and dynamic <= _number(
                    self.cfg.get("static_side_pass_soft_dynamic_max_count"), 1.0
                )
                and closing <= _number(
                    self.cfg.get("static_side_pass_soft_dynamic_max_count"), 1.0
                )
                and self.active_side is not None
                and self.active_frames < max_frames
            )
            if soft_dynamic:
                self.hard_hold_frames = 0
                self.active_frames += 1
                self.clear_frames += 1
                if self.clear_frames < hold_frames:
                    self.hold_reason = reason
                    return self.active_side
            self.candidate_side = None
            self.candidate_frames = 0
            self.active_side = None
            self.active_frames = 0
            self.clear_frames = 0
            self.hard_hold_frames = 0
            return None

        if self.active_side is not None:
            if candidate == self.active_side:
                self.hard_hold_frames = 0
                self.active_frames += 1
                self.clear_frames = 0
                if self.active_frames <= max_frames:
                    return self.active_side
                self.active_side = None
                self.active_frames = 0
                self.hard_hold_frames = 0
            else:
                self.hard_hold_frames = 0
                self.clear_frames += 1
                if hold_active:
                    self.active_frames += 1
                    if (
                        self.active_frames <= max_frames
                        and self.clear_frames < hold_frames
                    ):
                        self.hold_reason = reason or "temporary_geometry_gap"
                        return self.active_side
                if self.clear_frames >= (hold_frames if hold_active else exit_frames):
                    self.active_side = None
                    self.active_frames = 0
                    self.clear_frames = 0
                    self.hard_hold_frames = 0

        if candidate is None:
            self.candidate_side = None
            self.candidate_frames = 0
            return None

        if candidate == self.candidate_side:
            self.candidate_frames += 1
        else:
            self.candidate_side = candidate
            self.candidate_frames = 1
        if self.active_side is None and self.candidate_frames >= min_frames:
            self.active_side = candidate
            self.active_frames = 1
            self.clear_frames = 0
            self.hard_hold_frames = 0
            return candidate
        return None

    def debug(self):
        return {
            "side_pass_side": self.active_side,
            "side_pass_candidate_side": self.candidate_side,
            "side_pass_candidate_frames": self.candidate_frames,
            "side_pass_active_frames": self.active_frames,
            "side_pass_clear_frames": self.clear_frames,
            "static_side_pass_dynamic_override": bool(self.dynamic_override),
            "side_pass_block_reason": self.block_reason,
            "side_pass_hold_reason": self.hold_reason,
            "side_pass_hard_hold_frames": self.hard_hold_frames,
            "static_side_pass_hold_active": bool(
                self.cfg.get("static_side_pass_hold_active", False)
            ),
        }


def apply_static_side_pass(result, debug, side_pass, side, source):
    """Apply a bounded side pass only after the temporal candidate gate."""

    debug.update(side_pass.debug())
    debug["static_side_pass"] = bool(side)
    if side is None:
        return result, debug

    # A preserved hard-risk side is state memory only. Never release a
    # confirmed dynamic/TTC/emergency response merely because an earlier
    # frame selected a side. Fresh class-04 dynamic-override candidates keep
    # their existing behavior; this guard is limited to the explicit
    # preserved-hard state.
    hard_safety = bool(
        debug.get("side_pass_hold_reason") == "hard_risk_lateral_intent"
        and side_pass.cfg.get("static_side_pass_hard_safety_guard", False)
        and (
            (
                debug.get("emergency", False)
                and not debug.get("static_only_emergency", False)
            )
            or debug.get("ttc_confirmed", False)
            or (
                debug.get("approach_confirmed", False)
                and debug.get("approach_signal", False)
                and _number(debug.get("front_dynamic_count", 0.0)) >= 2.0
            )
        )
    )
    if hard_safety:
        debug["static_side_pass"] = False
        debug["side_pass_suppressed"] = "preserved_hard_risk"
        return result, debug

    current_brake = _number(getattr(result, "brake", 0.0))
    release_brake = _number(
        side_pass.cfg.get("static_side_pass_release_brake"), 0.12
    )
    if current_brake > release_brake:
        debug["static_side_pass"] = False
        debug["side_pass_suppressed"] = "active_brake"
        return result, debug

    current_steer = _number(getattr(result, "steer", 0.0))
    bias = _number(side_pass.cfg.get("static_side_pass_steer_bias"), 0.18)
    signed_bias = bias if side == "right" else -bias
    if hasattr(result, "steer"):
        steer_limit = max(
            0.05,
            min(1.0, _number(
                side_pass.cfg.get("static_side_pass_steer_limit"), 0.45
            )),
        )
        result.steer = max(
            -steer_limit, min(steer_limit, current_steer + signed_bias)
        )
    if hasattr(result, "brake"):
        result.brake = 0.0
    if hasattr(result, "throttle"):
        floor = _number(side_pass.cfg.get("static_side_pass_throttle_floor"), 0.18)
        cap = _number(side_pass.cfg.get("static_side_pass_throttle_cap"), 0.65)
        result.throttle = max(0.0, min(cap, max(_number(result.throttle), floor)))
    debug["state"] = "RADAR_STATIC_SIDE_PASS"
    debug["source"] = source
    debug["side_pass_reason"] = "temporal_static_front_with_clear_side"
    debug["side_pass_steer_limit"] = max(
        0.05,
        min(1.0, _number(
            side_pass.cfg.get("static_side_pass_steer_limit"), 0.45
        )),
    )
    debug["output_control"] = {
        "throttle": _number(getattr(result, "throttle", 0.0)),
        "steer": _number(getattr(result, "steer", 0.0)),
        "brake": _number(getattr(result, "brake", 0.0)),
    }
    return result, debug


class _StaticSideRecovery(object):
    """Temporal, sensor-only escape from a verified low-speed static block."""

    def __init__(self, cfg):
        self.cfg = dict(cfg)
        self.candidate_side = None
        self.candidate_frames = 0
        self.active_side = None
        self.active_frames = 0
        self.clear_frames = 0

    def reset(self):
        self.candidate_side = None
        self.candidate_frames = 0
        self.active_side = None
        self.active_frames = 0
        self.clear_frames = 0

    def _candidate(self, radar, features, control):
        if not self.cfg.get("static_side_recovery", True):
            return None
        speed = feature_value(features, "ego_speed", radar.get("ego_speed", 0.0))
        front_min = _number(radar.get("front_min", 0.0))
        front_close = _number(radar.get("front_close_count", 0.0))
        point_count = _number(radar.get("point_count", 0.0))
        left = _number(radar.get("left_nearest", 0.0))
        right = _number(radar.get("right_nearest", 0.0))
        dynamic = _number(radar.get("front_dynamic_count", 0.0))
        closing = _number(radar.get("front_closing_count", 0.0))
        ttc = _number(radar.get("front_ttc", 0.0))
        depth_trend = _number(radar.get("front_depth_trend", 0.0))
        traffic_kind = str(
            getattr(features, "traffic_control_kind", "") or ""
        ).strip().lower()

        # The recovery is only for a stopped/near-stopped vehicle.  It never
        # overrides a confirmed traffic control, VRU, immediate hazard,
        # active emergency brake, or a short radar TTC.
        if speed > _number(self.cfg.get("recovery_max_speed_mps"), 0.8):
            return None
        if front_min <= 0.0 or front_min > _number(
            self.cfg.get("recovery_front_max_m"), 3.6
        ):
            return None
        if front_close < 1.0 or point_count < 4.0:
            return None
        if traffic_kind in {
            "traffic_light", "trafficlight", "stop_sign", "stopsign"
        }:
            return None
        if bool(getattr(features, "red_light_active", False)):
            return None
        if bool(getattr(features, "immediate_hazard", False)):
            return None
        if optional_feature_value(features, "front_pedestrian_distance") is not None:
            return None
        short_ttc = 0.0 < ttc <= _number(self.cfg.get("recovery_min_ttc"), 1.3)
        confirmed_ttc_approach = short_ttc and (
            closing >= 1.0
            or depth_trend > _number(self.cfg.get("recovery_ttc_trend_m"), 0.35)
        )
        if confirmed_ttc_approach:
            return None
        # A radar-stationary return is a plausible static obstacle.  Several
        # front dynamic returns are instead consistent with a cut-in/lead
        # vehicle; a side bias there can steer into the actor even when a
        # large lateral range appears open for a few frames.
        if dynamic >= 2.0:
            return None
        if closing >= 1.0 and depth_trend > 0.35:
            return None

        side_min = _number(self.cfg.get("recovery_side_min_m"), 5.0)
        side_margin = _number(self.cfg.get("recovery_side_margin_m"), 2.5)
        if right >= side_min and right > left + side_margin:
            return "right"
        if left >= side_min and left > right + side_margin:
            return "left"
        return None

    def observe(self, radar, features, control):
        candidate = self._candidate(radar, features, control)
        min_frames = max(1, int(self.cfg.get("recovery_min_frames", 12)))
        exit_frames = max(1, int(self.cfg.get("recovery_exit_frames", 8)))
        max_frames = max(min_frames, int(self.cfg.get("recovery_max_frames", 100)))

        if candidate == self.candidate_side:
            self.candidate_frames += 1
        elif candidate is None:
            self.candidate_side = None
            self.candidate_frames = 0
        else:
            self.candidate_side = candidate
            self.candidate_frames = 1

        if self.active_side is None and self.candidate_side is not None:
            if self.candidate_frames >= min_frames:
                self.active_side = self.candidate_side
                self.active_frames = 0
                self.clear_frames = 0

        if self.active_side is not None:
            self.active_frames += 1
            if candidate == self.active_side:
                self.clear_frames = 0
            else:
                self.clear_frames += 1
            if self.clear_frames >= exit_frames or self.active_frames > max_frames:
                self.active_side = None
                self.active_frames = 0
                self.clear_frames = 0

        return self.active_side

    def debug(self):
        return {
            "recovery_side": self.active_side,
            "recovery_candidate_side": self.candidate_side,
            "recovery_candidate_frames": self.candidate_frames,
            "recovery_active_frames": self.active_frames,
            "recovery_clear_frames": self.clear_frames,
        }


def apply_static_side_recovery(result, debug, radar, features, recovery, source):
    """Add a bounded forward/side bias only after temporal verification."""

    side = recovery.observe(radar, features, result)
    debug.update(recovery.debug())
    debug["static_side_recovery"] = bool(side)
    if side is None:
        return result, debug

    current_brake = _number(getattr(result, "brake", 0.0))
    release_brake = _number(
        recovery.cfg.get(
            "recovery_release_brake",
            recovery.cfg.get("recovery_max_brake"),
        ),
        0.05,
    )
    if current_brake > release_brake:
        debug["static_side_recovery"] = False
        debug["recovery_suppressed"] = "active_brake"
        return result, debug

    current_steer = _number(getattr(result, "steer", 0.0))
    bias = _number(recovery.cfg.get("recovery_steer_bias"), 0.18)
    signed_bias = bias if side == "right" else -bias
    if hasattr(result, "steer"):
        result.steer = max(-0.45, min(0.45, current_steer + signed_bias))
    if hasattr(result, "brake"):
        # Only release the explicitly configured mild caution brake after the
        # temporal static-block gate.  A stronger emergency brake remains
        # untouched and suppresses recovery above.
        result.brake = 0.0 if current_brake <= release_brake else current_brake
    if hasattr(result, "throttle") and current_brake <= release_brake:
        floor = _number(recovery.cfg.get("recovery_throttle_floor"), 0.18)
        cap = _number(recovery.cfg.get("recovery_throttle_cap"), 0.32)
        result.throttle = max(0.0, min(cap, max(_number(result.throttle), floor)))
    debug["state"] = "RADAR_STATIC_SIDE_RECOVERY"
    debug["source"] = source
    debug["recovery_reason"] = "temporal_static_front_block_with_clear_side"
    debug["output_control"] = {
        "throttle": _number(getattr(result, "throttle", 0.0)),
        "steer": _number(getattr(result, "steer", 0.0)),
        "brake": _number(getattr(result, "brake", 0.0)),
    }
    return result, debug


class OrdinaryFallbackRule(object):
    """Safe output when radar evidence cannot lock one of 44 classes."""

    def __init__(self):
        self.config = {
            "caution_distance_m": 10.0,
            "emergency_distance_m": 4.0,
            "ttc_s": 1.3,
            "brake_caution": 0.10,
            "brake_emergency": 0.65,
            "throttle_cap_caution": 0.35,
            "throttle_cap_emergency": 0.05,
            "approach_trend_m": 0.15,
            "approach_confirm_frames": 2,
            "side_pass": False,
            "static_side_pass": False,
            "static_side_recovery": True,
            "recovery_min_frames": 12,
            "recovery_exit_frames": 8,
            "recovery_max_frames": 300,
            "recovery_max_speed_mps": 0.8,
            "recovery_front_max_m": 3.6,
            "recovery_side_min_m": 5.0,
            "recovery_side_margin_m": 2.5,
            "recovery_min_ttc": 1.3,
            "recovery_max_brake": 0.05,
            "recovery_release_brake": 0.12,
            "recovery_ttc_trend_m": 0.75,
            "recovery_steer_bias": 0.35,
            "recovery_throttle_floor": 0.18,
            "recovery_throttle_cap": 0.32,
        }
        self.approach_gate = _TemporalApproachGate(self.config)
        self.side_pass = _StaticSidePass(self.config)
        self.recovery = _StaticSideRecovery(self.config)
        self.prelock_hold = False
        self.prelock_clear_frames = 0
        self.pending_brake_hold = False
        self.pending_brake_clear_frames = 0

    def reset(self):
        self.approach_gate.reset()
        self.side_pass.reset()
        self.recovery.reset()
        self.prelock_hold = False
        self.prelock_clear_frames = 0
        self.pending_brake_hold = False
        self.pending_brake_clear_frames = 0

    def apply(self, control, radar, features):
        approach = self.approach_gate.observe(radar)
        side = self.side_pass.observe(radar, features, approach)
        result, debug = radar_safety_action(
            control,
            radar,
            features,
            self.config,
            "ordinary_fallback",
            approach_gate=self.approach_gate,
            approach_state=approach,
            static_side_pass_active=bool(side),
        )
        result, debug = apply_static_side_pass(
            result, debug, self.side_pass, side, "ordinary_fallback"
        )
        speed = feature_value(features, "ego_speed", radar.get("ego_speed", 0.0))
        front_min = _number(radar.get("front_min", 0.0))
        front_close = _number(radar.get("front_close_count", 0.0))
        traffic_kind = str(
            getattr(features, "traffic_control_kind", "") or ""
        ).strip().lower()
        traffic_hold = bool(
            getattr(features, "red_light_active", False)
            or traffic_kind in {"traffic_light", "trafficlight", "stop_sign", "stopsign"}
            or optional_feature_value(features, "front_pedestrian_distance") is not None
        )
        pending_reason = str(radar.get("_classifier_reason", "") or "")
        if pending_reason in {
            "semantic_flow_prehold", "semantic_flow_disambiguating",
        }:
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
            debug["state"] = (
                "UNKNOWN_FLOW_PREHOLD"
                if pending_reason == "semantic_flow_prehold"
                else "UNKNOWN_FLOW_DISAMBIGUATION_HOLD"
            )
            debug["classifier_reason"] = pending_reason
            debug["output_control"] = {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
            }
            return result, debug
        pending_obstacle_reason = pending_reason == "semantic_broad_candidate_pending"
        # Radar depth is measured from the sensor, not the front bumper.  A
        # fixed 4m emergency line therefore leaves almost no physical braking
        # distance at ordinary driving speed.  Once the classifier has online
        # evidence for an unresolved obstacle event, use a speed-dependent
        # stopping envelope: 3m covers sensor-to-bumper geometry and margin,
        # v^2/(2*4) is a conservative dry-road braking distance, and 0.25v
        # covers one short perception/control reaction interval.
        pending_stop_distance = 3.0 + speed * speed / 8.0 + 0.25 * speed
        pending_obstacle_brake_candidate = bool(
            not traffic_hold
            and pending_obstacle_reason
            and speed > 0.8
            # Completed class02 recordings remain above 7m until their
            # specific static lock.  The construction-warning precursor in
            # both observed class04 geometries enters this compact corridor.
            and 0.0 < front_min <= min(6.5, pending_stop_distance)
            and front_close >= 3.0
            and _number(radar.get("front_dynamic_count", 0.0)) <= 1.0
            and _number(radar.get("point_count", 0.0)) >= 4.0
            and _number(radar.get("left_nearest", 0.0)) > 0.0
            and _number(radar.get("right_nearest", 0.0)) > 0.0
            and abs(
                _number(radar.get("left_nearest", 0.0))
                - _number(radar.get("right_nearest", 0.0))
            ) <= 2.0
        )
        if pending_obstacle_brake_candidate:
            self.pending_brake_hold = True
            self.pending_brake_clear_frames = 0
        elif self.pending_brake_hold and front_min >= 7.0:
            self.pending_brake_clear_frames += 1
            if self.pending_brake_clear_frames >= 5:
                self.pending_brake_hold = False
                self.pending_brake_clear_frames = 0
        elif self.pending_brake_hold:
            self.pending_brake_clear_frames = 0
        pending_obstacle_brake = bool(self.pending_brake_hold and not traffic_hold)
        if pending_obstacle_brake:
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
            debug["state"] = "UNKNOWN_PENDING_OBSTACLE_BRAKE"
            debug["pending_obstacle_brake"] = True
            debug["pending_obstacle_brake_candidate"] = pending_obstacle_brake_candidate
            debug["pending_obstacle_brake_hold"] = self.pending_brake_hold
            debug["pending_obstacle_brake_clear_frames"] = self.pending_brake_clear_frames
            debug["pending_stop_distance_m"] = pending_stop_distance
            debug["classifier_reason"] = pending_reason
        stopped_near_obstacle = bool(
            not traffic_hold
            and speed <= 0.8
            and 0.0 < front_min <= 3.0
            and front_close >= 3.0
            and _number(radar.get("point_count", 0.0)) >= 4.0
        )
        if stopped_near_obstacle:
            self.prelock_hold = True
            self.prelock_clear_frames = 0
        elif self.prelock_hold and front_min >= 4.5:
            self.prelock_clear_frames += 1
            if self.prelock_clear_frames >= 5:
                self.prelock_hold = False
                self.prelock_clear_frames = 0
        elif self.prelock_hold:
            self.prelock_clear_frames = 0
        if self.prelock_hold:
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.70)
            debug["state"] = "UNKNOWN_PRELOCK_OBSTACLE_HOLD"
            debug["prelock_obstacle_hold"] = True
            debug["prelock_clear_frames"] = self.prelock_clear_frames
            debug.update(self.recovery.debug())
            debug["static_side_recovery"] = False
            debug["output_control"] = {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
            }
            return result, debug
        if pending_obstacle_brake:
            debug["prelock_obstacle_hold"] = False
            debug["prelock_clear_frames"] = self.prelock_clear_frames
            debug.update(self.recovery.debug())
            debug["static_side_recovery"] = False
            debug["output_control"] = {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
            }
            return result, debug
        debug["pending_obstacle_brake"] = False
        debug["pending_obstacle_brake_candidate"] = pending_obstacle_brake_candidate
        debug["pending_obstacle_brake_hold"] = self.pending_brake_hold
        debug["pending_obstacle_brake_clear_frames"] = self.pending_brake_clear_frames
        debug["pending_stop_distance_m"] = pending_stop_distance
        debug["classifier_reason"] = pending_reason
        debug["prelock_obstacle_hold"] = False
        debug["prelock_clear_frames"] = self.prelock_clear_frames
        return apply_static_side_recovery(
            result, debug, radar, features, self.recovery, "ordinary_fallback"
        )


class CategoryRule(object):
    def __init__(self, cfg):
        self.config = dict(cfg)
        self.approach_gate = _TemporalApproachGate(self.config)
        self.side_pass = _StaticSidePass(self.config)
        self.recovery = _StaticSideRecovery(self.config)

    def reset(self):
        self.approach_gate.reset()
        self.side_pass.reset()
        self.recovery.reset()

    def apply(self, control, radar, features):
        approach = self.approach_gate.observe(radar)
        side = self.side_pass.observe(radar, features, approach)
        result, debug = radar_safety_action(
            control,
            radar,
            features,
            self.config,
            "ordinary_class_{:02d}".format(self.config["class_id"]),
            approach_gate=self.approach_gate,
            approach_state=approach,
            static_side_pass_active=bool(side),
        )
        result, debug = apply_static_side_pass(
            result,
            debug,
            self.side_pass,
            side,
            "ordinary_class_{:02d}".format(self.config["class_id"]),
        )
        if side is not None:
            return result, debug
        return apply_static_side_recovery(
            result,
            debug,
            radar,
            features,
            self.recovery,
            "ordinary_class_{:02d}".format(self.config["class_id"]),
        )
