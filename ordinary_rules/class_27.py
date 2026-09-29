"""Class-27 parked-vehicle occluded pedestrian crossing.

The parked vehicle is a physical lane blockage, not merely a cue to creep at
it until the hidden pedestrian appears.  This rule therefore stages a short,
sensor-relative left bypass, waits for a reproducible opposing-traffic gap,
and keeps a separate multi-frame pedestrian latch.  Runtime code consumes no
route id, XML coordinate, map coordinate, or evaluator label.
"""

from __future__ import absolute_import

import math

from .class_04 import Rule04
from .class_37 import Rule37
from .common import _number, clone_control, feature_value
from .configs import get_config


class Rule27(Rule37):
    """Stop, plan a compact left bypass, then rejoin the online route."""

    INITIAL_HOLD_FRAMES = 12
    APPROACH_TARGET_MPS = 5.0
    APPROACH_MAX_THROTTLE = 0.85
    PREPARE_PROGRESS_M = 13.0
    PREPARE_TARGET_MPS = 2.5
    APPROACH_MIN_PROGRESS_M = 16.0
    STAGING_DISTANCE_M = 9.0
    STAGING_CONFIRM_FRAMES = 2
    STOP_CONFIRM_FRAMES = 3
    GAP_CLEAR_FRAMES = 5
    GAP_SAFE_TTC_S = 7.0
    GAP_TIMEOUT_FRAMES = 900
    BYPASS_TARGET_MPS = 3.5
    BYPASS_TIMEOUT_FRAMES = 500
    ONCOMING_CRITICAL_TTC_S = 2.5

    # Choose the online side opposite the observed parked vehicle.  A narrow
    # corridor is sufficient; the sign is sensor-derived because detector
    # lateral and route-lateral conventions are not interchangeable.
    CORRIDOR_MAGNITUDE_M = 1.55
    SHIFT_END_M = 6.0
    HOLD_END_M = 22.0
    REJOIN_END_M = 30.0
    FINISH_M = 34.0
    # A completed 34 m window already separates obstacle episodes.  Requiring
    # additional travel can deadlock against a new blocker closer than the
    # cooldown; fresh two-frame online evidence is the re-arm hysteresis.
    REARM_PROGRESS_M = 0.0
    # The category routes contain repeated parked rows, not one isolated
    # blocker.  Four finite 34 m windows cover the longest common route while
    # each window is still armed only by fresh online near-obstacle evidence.
    MAX_BYPASS_CYCLES = 4

    DYNAMIC_CONFIRM_FRAMES = 2
    DYNAMIC_MIN_POINTS = 2.0
    COLLISION_PULSE_PAUSE_FRAMES = 8
    EVENT_MIN_HOLD_FRAMES = 60
    EVENT_CLEAR_FRAMES = 15
    OCCLUSION_CUE_MIN_FORWARD_M = 3.0
    OCCLUSION_CUE_MAX_FORWARD_M = 4.0
    OCCLUSION_CUE_MIN_SCORE = 0.50
    OCCLUSION_CUE_MIN_PROGRESS_M = 30.0

    def __init__(self, cfg):
        super(Rule27, self).__init__(cfg)
        self._reset_class27()

    def _reset_class27(self):
        self.phase = "MONITOR"
        self.phase_frames = 0
        self.cycle = 0
        self.initial_hold_frames = 0
        self.approach_progress_m = 0.0
        self.stage_progress_floor_m = self.APPROACH_MIN_PROGRESS_M
        self.last_timestamp_s = None
        self.staging_frames = 0
        self.stop_frames = 0
        self.gap_clear_frames = 0
        self.plan = None
        self.blocker_lateral_m = None
        self.bypass_corridor_m = self.CORRIDOR_MAGNITUDE_M
        self.dynamic_frames = 0
        self.dynamic_clear_frames = 0
        self.event_hold_frames = 0
        self.pulse_pause_frames = 0
        self.resume_phase = "WAIT_GAP"
        self.occlusion_vehicle_primed = False
        self._awaiting_crossing_event = False
        self._crossing_hold = False
        self._crossing_completed = False
        self._armed_creep = False

    def reset(self):
        super(Rule27, self).reset()
        self._reset_class27()

    def prime_crossing_hold(self, reason, await_event=False):
        """Carry the router's pre-handoff stop into this independent FSM."""
        if self.phase != "MONITOR":
            return
        self.phase = "INITIAL_HOLD"
        self.phase_frames = 0
        self.initial_hold_frames = 0
        self._trigger_reason = str(
            reason or "parked_vehicle_occluded_crossing"
        )
        self._awaiting_crossing_event = bool(await_event)
        self._crossing_hold = True

    @staticmethod
    def _stop(control):
        control.throttle = 0.0
        control.brake = 1.0
        control.steer = 0.0
        if hasattr(control, "reverse"):
            control.reverse = False
        if hasattr(control, "manual_gear_shift"):
            control.manual_gear_shift = False

    @staticmethod
    def _track_speed(control, speed, target):
        error = target - speed
        if error < -0.5:
            control.throttle = 0.0
            control.brake = min(0.35, 0.12 + 0.08 * abs(error))
        elif error > 0.20:
            control.throttle = min(0.85, max(0.28, 0.32 + 0.16 * error))
            control.brake = 0.0
        else:
            control.throttle = min(
                _number(getattr(control, "throttle", 0.0)), 0.12
            )
            control.brake = 0.0
        if hasattr(control, "reverse"):
            control.reverse = False

    def _advance_progress(self, speed, now_s):
        if self.last_timestamp_s is None:
            self.last_timestamp_s = now_s
            return
        dt = max(0.0, min(0.25, now_s - self.last_timestamp_s))
        self.last_timestamp_s = now_s
        self.approach_progress_m += max(0.0, speed) * dt

    @staticmethod
    def _observed_blocker_lateral(features):
        candidates = []
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "van", "truck", "bus",
            }:
                continue
            score = _number(item.get("score"), 0.0)
            box = item.get("box_lidar") or item
            lateral = _number(box.get("x"), 99.0)
            forward = _number(box.get("y"), -1.0)
            if score >= 0.70 and 1.5 <= forward <= 14.0 and abs(lateral) >= 0.5:
                candidates.append((forward, -score, lateral))
        return min(candidates)[2] if candidates else None

    @classmethod
    def _occlusion_vehicle_cue(
        cls, features, max_forward=None, min_score=None
    ):
        """Return the nearest stable vehicle hiding the crossing zone."""
        max_forward = (
            cls.OCCLUSION_CUE_MAX_FORWARD_M
            if max_forward is None else float(max_forward)
        )
        min_score = (
            cls.OCCLUSION_CUE_MIN_SCORE
            if min_score is None else float(min_score)
        )
        candidates = []
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "van", "truck", "bus",
            }:
                continue
            score = _number(item.get("score"), 0.0)
            box = item.get("box_lidar") or item
            lateral = _number(box.get("x"), 99.0)
            forward = _number(box.get("y"), -1.0)
            if (
                score >= min_score
                and cls.OCCLUSION_CUE_MIN_FORWARD_M <= forward
                <= max_forward
                and 1.0 <= abs(lateral) <= 5.0
            ):
                candidates.append((forward, lateral, score))
        return min(candidates) if candidates else None

    def _build_plan(self, route_lateral_m):
        origin = _number(route_lateral_m, 0.0)
        corridor = self.bypass_corridor_m
        target_y = corridor - origin
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y * 0.5 * (1.0 - math.cos(math.pi * fraction))
            waypoints.append([self.SHIFT_END_M * fraction, y])
        waypoints.append([self.HOLD_END_M, target_y])
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y + (-origin - target_y) * 0.5 * (
                1.0 - math.cos(math.pi * fraction)
            )
            waypoints.append([
                self.HOLD_END_M
                + (self.REJOIN_END_M - self.HOLD_END_M) * fraction,
                y,
            ])
        waypoints.append([self.FINISH_M, -origin])
        return {
            "active": True,
            "class_id": 27,
            "cycle": self.cycle,
            "path_id": "c27_sensor_left_bypass_v4",
            "target_speed_mps": self.BYPASS_TARGET_MPS,
            "waypoints": waypoints,
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.42,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": True,
            "use_dense_route_pid": True,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class27_occluded_left_bypass",
            "pid_step_rejoin": True,
            "origin_route_lateral_m": origin,
            "radar_corridor_m": corridor,
            "shift_start_m": 0.0,
            "shift_end_m": self.SHIFT_END_M,
            "hold_end_m": self.HOLD_END_M,
            "rejoin_end_m": self.REJOIN_END_M,
            "stabilize_end_m": self.FINISH_M,
            "finish_lateral_tolerance_m": 0.30,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": self.BYPASS_TIMEOUT_FRAMES,
        }

    def _oncoming(self, packet, speed):
        return Rule04._oncoming_metrics(
            packet, speed, self.BYPASS_TARGET_MPS
        )

    def _observe_dynamic(self, radar, features):
        dynamic = _number(radar.get("front_dynamic_count", 0.0))
        semantic = getattr(features, "front_pedestrian_distance", None)
        signal = bool(semantic is not None or dynamic >= self.DYNAMIC_MIN_POINTS)
        self.dynamic_frames = self.dynamic_frames + 1 if signal else 0
        confirmed = self.dynamic_frames >= self.DYNAMIC_CONFIRM_FRAMES
        return signal, confirmed, dynamic

    def apply(self, control, radar, features):
        result = clone_control(control)
        speed = abs(feature_value(
            features, "ego_speed", radar.get("ego_speed", 0.0)
        ))
        now_s = _number(radar.get("_timestamp"), 0.0)
        nearest_depth = _number(radar.get("depth_min"), 0.0)
        packet = radar.get("_online_radar_packet")
        route_lateral = radar.get("_route_lateral_m")
        feedback = radar.get("_local_path_feedback") or {}
        feedback_matches_plan = False
        self.phase_frames += 1
        self._advance_progress(speed, now_s)

        dynamic_signal, dynamic_confirmed, dynamic_points = (
            self._observe_dynamic(radar, features)
        )
        if (
            self._awaiting_crossing_event
            and dynamic_confirmed
            and self.phase != "EVENT_HOLD"
        ):
            # Once the real pedestrian has crossed, the official scenario
            # removes both pedestrian and parked blocker.  A path already in
            # progress must be finished to rejoin safely; before path entry,
            # the learned route can continue without an unnecessary detour.
            self.resume_phase = (
                "BYPASS" if self.phase == "BYPASS" else "COMPLETE"
            )
            self.phase = "EVENT_HOLD"
            self.phase_frames = 0
            self.event_hold_frames = 0
            self.dynamic_clear_frames = 0
            self._awaiting_crossing_event = False
            self._crossing_hold = True
            self._armed_creep = False
        elif (
            self._awaiting_crossing_event
            and dynamic_signal
            and self.dynamic_frames == 1
        ):
            # A one-frame cluster can be the collision impulse observed in v3.
            # Stop immediately, but never mark the crossing complete from it.
            self.pulse_pause_frames = self.COLLISION_PULSE_PAUSE_FRAMES

        oncoming = self._oncoming(packet, speed)
        oncoming_ttc = oncoming[2] if oncoming is not None else None
        occlusion_prime = self._occlusion_vehicle_cue(
            features, max_forward=10.0, min_score=0.70
        )
        if (
            self.phase == "BYPASS"
            and self.cycle == 1
            and self._awaiting_crossing_event
            and self.approach_progress_m >= self.OCCLUSION_CUE_MIN_PROGRESS_M
            and occlusion_prime is not None
        ):
            self.occlusion_vehicle_primed = True
        occlusion_cue = self._occlusion_vehicle_cue(features)
        if (
            self.phase == "BYPASS"
            and self.cycle == 1
            and self._awaiting_crossing_event
            and self.approach_progress_m >= self.OCCLUSION_CUE_MIN_PROGRESS_M
            and self.occlusion_vehicle_primed
            and occlusion_cue is not None
        ):
            # The walker is perpendicular and may have near-zero radial
            # velocity while hidden.  Stop on the persistent occluding-car
            # geometry before entering its crossing zone, hold for the common
            # crossing duration, then resume the same finite local path.
            self.resume_phase = "BYPASS"
            self.phase = "EVENT_HOLD"
            self.phase_frames = 0
            self.event_hold_frames = 0
            self.dynamic_clear_frames = 0
            self._awaiting_crossing_event = False
            self._crossing_hold = True
            self._armed_creep = False
        override = "class27_model_monitor"

        if self.phase == "MONITOR":
            pass
        elif self.phase == "INITIAL_HOLD":
            self.initial_hold_frames += 1
            self._stop(result)
            override = "class27_initial_safety_hold"
            if self.initial_hold_frames >= self.INITIAL_HOLD_FRAMES:
                self.phase = "APPROACH"
                self.phase_frames = 0
                self._crossing_hold = False
                self._armed_creep = True
                self.last_timestamp_s = now_s
        elif self.phase == "APPROACH":
            prepare_active = bool(
                self.approach_progress_m >= self.PREPARE_PROGRESS_M
            )
            close_staging = bool(
                self.approach_progress_m >= self.stage_progress_floor_m
                and 0.0 < nearest_depth <= self.STAGING_DISTANCE_M
            )
            self.staging_frames = self.staging_frames + 1 if close_staging else 0
            if self.staging_frames >= self.STAGING_CONFIRM_FRAMES:
                observed_lateral = self._observed_blocker_lateral(features)
                if observed_lateral is not None:
                    self.blocker_lateral_m = observed_lateral
                    self.bypass_corridor_m = (
                        self.CORRIDOR_MAGNITUDE_M
                        if observed_lateral < 0.0
                        else -self.CORRIDOR_MAGNITUDE_M
                    )
                self.phase = "BRAKE_FOR_PLAN"
                self.phase_frames = 0
                self.stop_frames = 0
                self._stop(result)
                override = "class27_staging_stop"
            else:
                model_brake = _number(getattr(control, "brake", 0.0))
                if model_brake >= 0.20:
                    result.throttle = 0.0
                    result.brake = max(model_brake, result.brake)
                    override = "class27_model_brake_veto"
                else:
                    self._track_speed(
                        result,
                        speed,
                        (
                            self.PREPARE_TARGET_MPS
                            if prepare_active else self.APPROACH_TARGET_MPS
                        ),
                    )
                    override = (
                        "class27_prepare_bypass_speed"
                        if prepare_active
                        else "class27_bounded_obstacle_approach"
                    )
        elif self.phase == "BRAKE_FOR_PLAN":
            self._stop(result)
            override = "class27_staging_stop"
            self.stop_frames = self.stop_frames + 1 if speed <= 0.25 else 0
            if self.stop_frames >= self.STOP_CONFIRM_FRAMES:
                self.cycle += 1
                # The stopped-frame navigation lateral can jump by metres as
                # its near waypoint advances.  The local planner is anchored
                # at the current ego pose, so its displacement origin is zero;
                # the public route PID independently owns the absolute ±1.55m
                # corridor and rejoin target.
                self.plan = self._build_plan(0.0)
                self.phase = "WAIT_GAP"
                self.phase_frames = 0
                self.gap_clear_frames = 0
        elif self.phase == "WAIT_GAP":
            self._stop(result)
            override = "class27_wait_stable_gap"
            safe_gap = bool(
                oncoming is None or oncoming_ttc >= self.GAP_SAFE_TTC_S
            )
            self.gap_clear_frames = self.gap_clear_frames + 1 if safe_gap else 0
            if self.gap_clear_frames >= self.GAP_CLEAR_FRAMES:
                self.phase = "BYPASS"
                self.phase_frames = 0
            elif self.phase_frames >= self.GAP_TIMEOUT_FRAMES:
                override = "class27_gap_timeout_safe_stop"
        elif self.phase == "BYPASS":
            feedback_path_key = feedback.get("path_key") or []
            feedback_matches_plan = bool(
                self.plan is not None
                and len(feedback_path_key) >= 3
                and int(feedback_path_key[0]) == 27
                and int(feedback_path_key[1]) == self.cycle
                and str(feedback_path_key[2])
                == str(self.plan.get("path_id", ""))
            )
            # Controller feedback arrives one frame after this rule call. A
            # new cycle must not consume the prior cycle's terminal sample.
            progress = (
                _number(feedback.get("x_m"), 0.0)
                if feedback_matches_plan else 0.0
            )
            finished = bool(
                feedback_matches_plan
                and (
                    feedback.get("done", False)
                    or feedback.get("finished", False)
                )
            )
            if oncoming_ttc is not None and oncoming_ttc <= self.ONCOMING_CRITICAL_TTC_S:
                self._stop(result)
                override = "class27_critical_oncoming_hold"
            else:
                self._track_speed(
                    result, speed, self.BYPASS_TARGET_MPS
                )
                override = "class27_execute_left_bypass"
            if finished or progress >= self.FINISH_M:
                should_rearm = bool(
                    self.cycle < self.MAX_BYPASS_CYCLES
                )
                self.phase = "APPROACH" if should_rearm else "COMPLETE"
                self.phase_frames = 0
                self.staging_frames = 0
                self.plan = None
                if should_rearm:
                    self.stage_progress_floor_m = (
                        self.approach_progress_m + self.REARM_PROGRESS_M
                    )
                    override = "class27_rearm_after_first_bypass"
                else:
                    self._armed_creep = False
                    self._crossing_completed = True
                    override = "class27_bypass_complete"
        elif self.phase == "EVENT_HOLD":
            self._stop(result)
            override = "class27_confirmed_pedestrian_hold"
            self.event_hold_frames += 1
            self.dynamic_clear_frames = (
                self.dynamic_clear_frames + 1 if not dynamic_signal else 0
            )
            if (
                self.event_hold_frames >= self.EVENT_MIN_HOLD_FRAMES
                and self.dynamic_clear_frames >= self.EVENT_CLEAR_FRAMES
            ):
                self.phase = self.resume_phase
                self.phase_frames = 0
                self._crossing_hold = False
                if self.resume_phase == "COMPLETE":
                    self._crossing_completed = True
        elif self.phase == "COMPLETE":
            override = "class27_model_after_rejoin"

        if self.pulse_pause_frames > 0 and self.phase != "EVENT_HOLD":
            self.pulse_pause_frames -= 1
            self._stop(result)
            override = "class27_unconfirmed_dynamic_pause"

        if bool(getattr(features, "red_light_active", False)):
            self._stop(result)
            override = "class27_red_light_hold"

        state = "CLASS27_{}".format(self.phase)
        debug = {
            "state": state,
            "class27_parked_vehicle_crossing": True,
            "class27_phase": self.phase,
            "class27_override": override,
            "class27_trigger_reason": str(self._trigger_reason),
            "class27_approach_progress_m": float(self.approach_progress_m),
            "class27_stage_progress_floor_m": float(
                self.stage_progress_floor_m
            ),
            "class27_cycle": int(self.cycle),
            "class27_feedback_matches_plan": bool(feedback_matches_plan),
            "class27_blocker_lateral_m": (
                None
                if self.blocker_lateral_m is None
                else float(self.blocker_lateral_m)
            ),
            "class27_bypass_corridor_m": float(self.bypass_corridor_m),
            "class27_prepare_active": bool(
                self.phase == "APPROACH"
                and self.approach_progress_m >= self.PREPARE_PROGRESS_M
            ),
            "class27_nearest_depth_m": float(nearest_depth),
            "class27_staging_frames": int(self.staging_frames),
            "class27_stop_frames": int(self.stop_frames),
            "class27_gap_clear_frames": int(self.gap_clear_frames),
            "class27_oncoming_ttc_s": (
                None if oncoming_ttc is None else float(oncoming_ttc)
            ),
            "class27_occlusion_vehicle_cue": (
                None if occlusion_cue is None else {
                    "forward_m": float(occlusion_cue[0]),
                    "lateral_m": float(occlusion_cue[1]),
                    "score": float(occlusion_cue[2]),
                }
            ),
            "class27_occlusion_vehicle_primed": bool(
                self.occlusion_vehicle_primed
            ),
            "class27_dynamic_signal": bool(dynamic_signal),
            "class27_dynamic_confirmed": bool(dynamic_confirmed),
            "class27_dynamic_frames": int(self.dynamic_frames),
            "class27_dynamic_points": float(dynamic_points),
            "class27_pulse_pause_frames": int(self.pulse_pause_frames),
            "class27_awaiting_crossing_event": bool(
                self._awaiting_crossing_event
            ),
            "class27_crossing_hold": bool(self._crossing_hold),
            "class27_crossing_completed": bool(self._crossing_completed),
            # Compatibility keys consumed by existing diagnostics.
            "class37_crossing_hold": bool(self._crossing_hold),
            "class37_crossing_completed": bool(self._crossing_completed),
            "class37_awaiting_crossing_event": bool(
                self._awaiting_crossing_event
            ),
            "class37_armed_creep": bool(self._armed_creep),
            "output_control": {
                "throttle": float(getattr(result, "throttle", 0.0) or 0.0),
                "steer": float(getattr(result, "steer", 0.0) or 0.0),
                "brake": float(getattr(result, "brake", 0.0) or 0.0),
            },
        }
        if self.plan is not None and self.phase in {"WAIT_GAP", "BYPASS"}:
            debug["ordinary_local_path"] = dict(self.plan)
        return result, debug


def build_rule():
    return Rule27(get_config(27))
