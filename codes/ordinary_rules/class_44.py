"""Class-44 signalized left-turn entry behind a turning traffic flow.

The rule waits for the currently approaching flow actor to pass the merge
point, then commits once behind it. Runtime decisions use only online radar,
traffic-control evidence and ego motion; no route or XML identity is read.
"""

from __future__ import absolute_import

import math
from collections import deque

import numpy as np

from .common import clone_control, feature_value, _number
from .configs import get_config


class Rule44(object):
    WAIT_FOR_PASS = "WAIT_FOR_PASS"
    COMMITTED_ENTRY = "COMMITTED_ENTRY"
    STAGED_BRAKE = "STAGED_BRAKE"
    STAGED_ADVANCE = "STAGED_ADVANCE"
    FINAL_WAIT = "FINAL_WAIT"
    MERGE_ENTRY = "MERGE_ENTRY"
    ROUTE_FOLLOW = "ROUTE_FOLLOW"

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.state = self.WAIT_FOR_PASS
        self.pass_seen = False
        self.clear_frames = 0
        self.wait_frames = 0
        self.committed_distance_m = 0.0
        self.last_timestamp = None
        self.owned_green_frames = 0
        self.owned_green_latched = False
        self.track_history = {}
        self.staged_near_active = False
        self.staged_near_episode_count = 0
        self.staged_near_clear_frames = 0
        self.staged_stop_frames = 0
        self.staged_advance_distance_m = 0.0
        self.staged_advance_total_m = 0.0
        self.staged_advance_completed_cycles = 0
        self.staged_advance_braking_latched = False
        self.staged_advance_exhausted = False
        self.final_near_active = False
        self.final_near_clear_frames = 0
        self.final_pass_seen = False
        self.staged_release_latched = False
        self.dense_handoff_frames = 0
        self.dense_handoff_latched = False

    @staticmethod
    def _packet_points(packet):
        if isinstance(packet, (tuple, list)) and len(packet) == 2:
            packet = packet[1]
        try:
            points = np.asarray(packet, dtype=np.float32)
        except Exception:
            return np.empty((0, 4), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            return np.empty((0, 4), dtype=np.float32)
        points = points[:, :4]
        return points[np.isfinite(points).all(axis=1)]

    def _flow_observation(self, radar, speed):
        points = self._packet_points(radar.get("_online_radar_packet"))
        if len(points) == 0:
            return {"count": 0, "distance_m": 0.0, "ttc_s": 0.0,
                    "closing_speed_mps": 0.0}
        depth, azimuth, velocity = points[:, 0], points[:, 2], points[:, 3]
        relative_velocity = velocity + abs(speed) * np.cos(azimuth)
        mask = (
            (depth > 0.0)
            & (depth <= _number(self.config.get("flow_radar_max_m"), 60.0))
            & (np.abs(azimuth) <= math.radians(_number(
                self.config.get("flow_radar_half_angle_deg"), 60.0
            )))
            & (relative_velocity <= -_number(
                self.config.get("flow_min_closing_speed_mps"), 2.0
            ))
        )
        if not np.any(mask):
            return {"count": 0, "distance_m": 0.0, "ttc_s": 0.0,
                    "closing_speed_mps": 0.0}
        indices = np.where(mask)[0]
        index = indices[int(np.argmin(depth[indices]))]
        distance = float(depth[index])
        closing = max(0.1, -float(relative_velocity[index]))
        return {
            "count": int(len(indices)),
            "distance_m": distance,
            "ttc_s": distance / closing,
            "closing_speed_mps": closing,
        }

    def _signal_observation(self, radar, features):
        kind = str(getattr(features, "traffic_control_kind", "") or "").lower()
        raw_red = bool(getattr(features, "red_light_active", False))
        waypoint = radar.get("_route_waypoint_observation") or {}
        far = waypoint.get("far") or {}
        far_error = _number(far.get("heading_error_rad"), float("nan"))
        owned_side = None
        if math.isfinite(far_error) and abs(far_error) >= _number(
            self.config.get("signal_owner_min_turn_rad"), 0.25
        ):
            # Online navigation geometry projects the governing turn head to
            # the same image side used by the existing traffic-light adapter.
            # Negative heading error is on the right of the front image.
            owned_side = "right" if far_error < 0.0 else "left"
        votes = getattr(features, "traffic_light_region_votes", {}) or {}
        vote = votes.get(owned_side or "", {}) or {}
        owned_color = str(vote.get("color", "unknown") or "unknown").lower()
        owned_confidence = _number(vote.get("confidence"), 0.0)
        owned_green = bool(
            owned_side is not None
            and owned_color == "green"
            and owned_confidence >= _number(
                self.config.get("signal_owned_green_min_confidence"), 0.60
            )
        )
        self.owned_green_frames = (
            self.owned_green_frames + 1 if owned_green else 0
        )
        if self.owned_green_frames >= int(
            self.config.get("signal_owned_green_confirm_frames", 2)
        ):
            self.owned_green_latched = True
        stop_sign = kind in {"stop_sign", "stopsign"}
        hold = bool(stop_sign or (raw_red and not self.owned_green_latched))
        return {
            "hold": hold,
            "raw_red": raw_red,
            "stop_sign": stop_sign,
            "owned_side": owned_side,
            "owned_color": owned_color,
            "owned_confidence": float(owned_confidence),
            "owned_green_frames": int(self.owned_green_frames),
            "owned_green_latched": bool(self.owned_green_latched),
            "far_heading_error_rad": (
                float(far_error) if math.isfinite(far_error) else None
            ),
        }

    @staticmethod
    def _vehicle_track_position(item):
        if not isinstance(item, dict):
            return None
        name = str(item.get("class_name", "") or "").lower()
        if name not in {"car", "van", "truck", "vehicle", "bus"}:
            return None
        score = _number(item.get("score", item.get("confidence", 0.0)))
        if score < 0.70:
            return None
        source = str(item.get("source", "") or "").lower()
        x = _number(item.get("x"), float("nan"))
        y = _number(item.get("y"), float("nan"))
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        forward, lateral = (x, y) if source == "openpcdet" else (y, x)
        if not (0.0 < forward < 30.0 and abs(lateral) < 5.5):
            return None
        return forward, lateral, score

    def _tracked_flow_observation(self, features, timestamp):
        values = (
            features.get("tracked_objects", [])
            if isinstance(features, dict)
            else getattr(features, "tracked_objects", [])
        )
        candidates = []
        active_keys = set()
        for index, item in enumerate(list(values or [])):
            position = self._vehicle_track_position(item)
            if position is None:
                continue
            forward, lateral, score = position
            track_id = item.get("track_id")
            key = (str(item.get("source", "") or ""), track_id)
            if track_id is None:
                key = (key[0], "anonymous-{}".format(index))
            active_keys.add(key)
            measurement = _number(item.get("measurement_timestamp"), timestamp)
            history = self.track_history.setdefault(key, deque(maxlen=16))
            if not history or measurement > history[-1][0] + 1e-6:
                history.append((measurement, forward, lateral))
            recent = [row for row in history if measurement - row[0] <= 0.8]
            if len(recent) < 2:
                continue
            span = measurement - recent[0][0]
            if span < 0.4:
                continue
            closing = (recent[0][1] - forward) / max(span, 1e-3)
            if closing < 1.5:
                continue
            candidates.append({
                "track_id": track_id,
                "forward_m": float(forward),
                "lateral_m": float(lateral),
                "closing_speed_mps": float(closing),
                "eta_s": float(forward / closing),
                "score": float(score),
                "history_span_s": float(span),
            })
        for key in list(self.track_history):
            history = self.track_history[key]
            if key not in active_keys and history and timestamp - history[-1][0] > 1.0:
                del self.track_history[key]
        if not candidates:
            return {"count": 0, "minimum_eta_s": None, "danger": False}
        nearest = min(candidates, key=lambda item: item["eta_s"])
        minimum_eta = float(nearest["eta_s"])
        return {
            "count": len(candidates),
            "minimum_eta_s": minimum_eta,
            "danger": minimum_eta < _number(
                self.config.get("flow_next_min_ttc_s"), 2.8
            ),
            "nearest": nearest,
        }

    def apply(self, control, radar, features):
        result = clone_control(control)
        speed = max(0.0, feature_value(features, "ego_speed", 0.0))
        timestamp = _number(radar.get("_timestamp"), 0.0)
        flow = self._flow_observation(radar, speed)
        tracked_flow = self._tracked_flow_observation(features, timestamp)
        signal = self._signal_observation(radar, features)
        signal_hold = bool(signal["hold"])
        self.wait_frames += 1
        if bool(radar.get("_classifier_flow_pass_seen", False)):
            self.pass_seen = True

        if self.last_timestamp is not None and timestamp > self.last_timestamp:
            dt = min(0.2, timestamp - self.last_timestamp)
            distance_step = speed * dt
            if self.state in (
                self.COMMITTED_ENTRY, self.STAGED_BRAKE,
                self.STAGED_ADVANCE, self.FINAL_WAIT, self.MERGE_ENTRY,
            ):
                self.committed_distance_m += distance_step
            if self.state == self.STAGED_ADVANCE:
                self.staged_advance_distance_m += distance_step
        self.last_timestamp = timestamp

        if (
            self.state == self.WAIT_FOR_PASS
            and flow["count"] > 0
            and flow["distance_m"] <= _number(
                self.config.get("flow_pass_near_m"), 6.0
            )
        ):
            self.pass_seen = True

        minimum_gap_ttc = _number(
            self.config.get("flow_next_min_ttc_s"), 2.8
        )
        clear_candidate = bool(
            self.state == self.WAIT_FOR_PASS
            and self.pass_seen
            and not signal_hold
            and flow["count"] > 0
            and flow["ttc_s"] >= minimum_gap_ttc
            and not tracked_flow["danger"]
        )
        self.clear_frames = self.clear_frames + 1 if clear_candidate else 0
        if self.clear_frames >= int(self.config.get("flow_clear_confirm_frames", 3)):
            self.state = self.COMMITTED_ENTRY
            self.committed_distance_m = 0.0

        if (
            self.state in (self.COMMITTED_ENTRY, self.MERGE_ENTRY)
            and self.committed_distance_m >= _number(
                self.config.get("entry_commit_distance_m"), 30.0
            )
        ):
            self.state = self.ROUTE_FOLLOW

        raw_emergency_flow = bool(
            self.state in (
                self.COMMITTED_ENTRY, self.STAGED_BRAKE,
                self.STAGED_ADVANCE, self.FINAL_WAIT,
            )
            and flow["count"] > 0
            and flow["distance_m"] <= _number(
                self.config.get("committed_emergency_m"), 4.0
            )
        )
        if self.state == self.COMMITTED_ENTRY and raw_emergency_flow:
            # The first stop is known-safe, but v14 proved it is too far from
            # the conflict point for a zero-speed one-shot merge.  Stop here,
            # then use successive observed gaps for bounded low-speed staging
            # advances.  v15 proved that one such advance still leaves the
            # ego too far from the conflict point for a safe zero-speed merge.
            self.state = self.STAGED_BRAKE
            self.staged_near_active = True
            self.staged_near_episode_count = 1
            self.staged_near_clear_frames = 0

        if self.state == self.STAGED_BRAKE:
            if raw_emergency_flow:
                self.staged_near_clear_frames = 0
            elif self.staged_near_active:
                self.staged_near_clear_frames += 1
                if self.staged_near_clear_frames >= int(
                    self.config.get("staged_near_clear_confirm_frames", 2)
                ):
                    self.staged_near_active = False
                    self.staged_near_clear_frames = 0

            stopped = speed <= _number(
                self.config.get("staged_stop_speed_mps"), 0.15
            )
            self.staged_stop_frames = self.staged_stop_frames + 1 if stopped else 0
            advance_candidate = bool(
                not self.staged_near_active
                and self.staged_stop_frames >= int(
                    self.config.get("staged_stop_confirm_frames", 2)
                )
                and not signal_hold
                and flow["count"] > 0
                and flow["ttc_s"] >= _number(
                    self.config.get("staged_advance_min_ttc_s"), 2.2
                )
            )
            if advance_candidate:
                self.state = self.STAGED_ADVANCE
                self.staged_advance_distance_m = 0.0
                self.staged_advance_braking_latched = False
        else:
            advance_candidate = False

        staged_total_cap_m = _number(
            self.config.get("staged_advance_max_total_m"), 5.5
        )
        remaining_advance_m = max(0.0, min(
            _number(self.config.get("staged_advance_distance_m"), 3.0)
            - self.staged_advance_distance_m,
            staged_total_cap_m
            - self.staged_advance_total_m
            - self.staged_advance_distance_m,
        ))
        advance_braking_distance_m = (
            speed * speed
            / (2.0 * max(0.1, _number(
                self.config.get("staged_advance_decel_mps2"), 4.0
            )))
            + _number(self.config.get("staged_advance_brake_margin_m"), 0.5)
        )
        if (
            self.state == self.STAGED_ADVANCE
            and (
                remaining_advance_m <= advance_braking_distance_m
                or raw_emergency_flow
            )
        ):
            self.staged_advance_braking_latched = True
        if (
            self.state == self.STAGED_ADVANCE
            and self.staged_advance_braking_latched
            and speed <= _number(
                self.config.get("staged_stop_speed_mps"), 0.15
            )
        ):
            # Count the actual odometric progress, including an emergency-
            # shortened cycle.  A short cycle must wait for another measured
            # gap; it must never be treated as permission to merge.
            self.staged_advance_total_m += self.staged_advance_distance_m
            self.staged_advance_completed_cycles += 1
            self.state = self.FINAL_WAIT
            self.final_near_active = False
            self.final_near_clear_frames = 0
            self.final_pass_seen = False

        if self.state == self.FINAL_WAIT:
            if raw_emergency_flow:
                if not self.final_near_active:
                    self.final_near_active = True
                    self.staged_near_episode_count += 1
                self.final_near_clear_frames = 0
            elif self.final_near_active:
                self.final_near_clear_frames += 1
                if self.final_near_clear_frames >= int(
                    self.config.get("staged_near_clear_confirm_frames", 2)
                ):
                    self.final_near_active = False
                    self.final_near_clear_frames = 0
                    self.final_pass_seen = True

            # Position alone is never merge authority.  v16 rejected a
            # measured 3.10 s gap while 0.43 m short of its fixed progress
            # gate, then accepted a 2.45 s oncoming vehicle after crossing
            # that gate and became physically wedged.  Convert both facts to
            # one feasibility boundary: farther-back stops demand additional
            # visible time, while even the closest staged stop keeps the same
            # observed-gap floor as the initial junction entry.
            merge_reference_progress_m = _number(
                self.config.get("merge_reference_progress_m"), 4.0
            )
            merge_gap_floor_ttc_s = _number(
                self.config.get("flow_next_min_ttc_s"), 2.8
            )
            merge_gap_extra_ttc_s = max(
                0.0,
                merge_reference_progress_m - self.staged_advance_total_m,
            ) * _number(
                self.config.get("merge_gap_extra_ttc_per_remaining_m"), 0.5
            )
            merge_required_ttc_s = (
                merge_gap_floor_ttc_s + merge_gap_extra_ttc_s
            )
            merge_gap_candidate = bool(
                self.final_pass_seen
                and not self.final_near_active
                and not signal_hold
                and flow["count"] > 0
                and flow["ttc_s"] >= merge_required_ttc_s
                and not tracked_flow["danger"]
            )
            creep_gap_candidate = bool(
                self.final_pass_seen
                and not self.final_near_active
                and not signal_hold
                and flow["count"] > 0
                and flow["ttc_s"] >= _number(
                    self.config.get("staged_advance_min_ttc_s"), 2.2
                )
            )
            maximum_cycles = int(
                self.config.get("staged_advance_max_cycles", 3)
            )
            remaining_total_m = max(
                0.0,
                staged_total_cap_m - self.staged_advance_total_m,
            )
            if merge_gap_candidate:
                self.state = self.MERGE_ENTRY
                self.staged_release_latched = True
            elif creep_gap_candidate:
                if (
                    self.staged_advance_completed_cycles < maximum_cycles
                    and remaining_total_m > 1e-3
                ):
                    # Re-arm the same class-level creep primitive for the next
                    # independently observed actor gap.  This is allowed only
                    # after full-merge feasibility was evaluated and rejected.
                    self.state = self.STAGED_ADVANCE
                    self.staged_advance_distance_m = 0.0
                    self.staged_advance_braking_latched = False
                    self.final_near_active = False
                    self.final_near_clear_frames = 0
                    self.final_pass_seen = False
                else:
                    # No further bounded creep is available and the visible
                    # merge gap is insufficient. Hold rather than guessing.
                    self.staged_advance_exhausted = True
            staged_required_total_m = merge_reference_progress_m
            staged_need_more_advance = bool(
                self.staged_advance_total_m < merge_reference_progress_m
            )
            staged_release_ttc_s = merge_required_ttc_s
            staged_release_candidate = merge_gap_candidate
        else:
            staged_release_candidate = False
            merge_gap_candidate = False
            creep_gap_candidate = False
            staged_required_total_m = _number(
                self.config.get("merge_reference_progress_m"), 4.0
            )
            staged_need_more_advance = bool(
                self.staged_advance_total_m < staged_required_total_m
            )
            staged_release_ttc_s = None
            merge_required_ttc_s = None

        emergency_flow = bool(self.state in (self.STAGED_BRAKE, self.FINAL_WAIT))
        lateral_m = _number(radar.get("_route_lateral_m"), float("nan"))
        dense_handoff_candidate = bool(
            self.state in (
                self.COMMITTED_ENTRY, self.MERGE_ENTRY, self.ROUTE_FOLLOW
            )
            and self.committed_distance_m >= _number(
                self.config.get("dense_handoff_min_distance_m"), 15.0
            )
            and math.isfinite(lateral_m)
            and abs(lateral_m) <= _number(
                self.config.get("dense_handoff_max_lateral_m"), 1.0
            )
        )
        if not self.dense_handoff_latched:
            self.dense_handoff_frames = (
                self.dense_handoff_frames + 1
                if dense_handoff_candidate
                else 0
            )
            if self.dense_handoff_frames >= int(
                self.config.get("dense_handoff_confirm_frames", 3)
            ):
                self.dense_handoff_latched = True
        target_speed = None
        if self.state == self.WAIT_FOR_PASS:
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
        elif self.state in (self.STAGED_BRAKE, self.FINAL_WAIT):
            result.throttle = 0.0
            result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
        elif self.state == self.STAGED_ADVANCE:
            if self.staged_advance_braking_latched:
                result.throttle = 0.0
                result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)
            else:
                result.throttle = _number(
                    self.config.get("staged_advance_throttle"), 0.45
                )
                result.brake = 0.0
        elif self.state in (self.COMMITTED_ENTRY, self.MERGE_ENTRY):
            target_key = (
                "merge_target_speed_mps"
                if self.state == self.MERGE_ENTRY
                else "entry_target_speed_mps"
            )
            target_speed = _number(
                self.config.get(target_key),
                10.0 if self.state == self.MERGE_ENTRY else 7.0,
            )
            if speed < target_speed:
                result.throttle = max(
                    _number(getattr(result, "throttle", 0.0)),
                    _number(self.config.get("entry_throttle_floor"), 0.75),
                )
                result.brake = 0.0
            else:
                result.throttle = min(_number(getattr(result, "throttle", 0.0)), 0.35)
                result.brake = min(_number(getattr(result, "brake", 0.0)), 0.10)

        debug = {
            "source": "ordinary_class_44",
            "state": "CLASS44_{}".format(self.state),
            "pass_seen": bool(self.pass_seen),
            "clear_candidate": clear_candidate,
            "clear_confirm_frames": int(self.clear_frames),
            "wait_frames": int(self.wait_frames),
            "signal_hold": bool(signal_hold),
            "signal_owner": signal,
            "emergency_flow": bool(emergency_flow),
            "raw_emergency_flow": bool(raw_emergency_flow),
            "staged_near_active": bool(self.staged_near_active),
            "staged_near_episode_count": int(self.staged_near_episode_count),
            "staged_near_clear_frames": int(self.staged_near_clear_frames),
            "staged_stop_frames": int(self.staged_stop_frames),
            "staged_advance_candidate": bool(advance_candidate),
            "staged_advance_distance_m": float(self.staged_advance_distance_m),
            "staged_advance_total_m": float(self.staged_advance_total_m),
            "staged_advance_completed_cycles": int(
                self.staged_advance_completed_cycles
            ),
            "staged_advance_required_total_m": float(
                staged_required_total_m
            ),
            "staged_advance_need_more": bool(staged_need_more_advance),
            "staged_advance_exhausted": bool(self.staged_advance_exhausted),
            "staged_release_ttc_s": staged_release_ttc_s,
            "staged_advance_remaining_m": float(remaining_advance_m),
            "staged_advance_braking_distance_m": float(
                advance_braking_distance_m
            ),
            "staged_advance_braking_latched": bool(
                self.staged_advance_braking_latched
            ),
            "final_near_active": bool(self.final_near_active),
            "final_near_clear_frames": int(self.final_near_clear_frames),
            "final_pass_seen": bool(self.final_pass_seen),
            "staged_release_candidate": bool(staged_release_candidate),
            "merge_gap_candidate": bool(merge_gap_candidate),
            "creep_gap_candidate": bool(creep_gap_candidate),
            "merge_required_ttc_s": merge_required_ttc_s,
            "staged_release_latched": bool(self.staged_release_latched),
            "active_target_speed_mps": target_speed,
            "committed_distance_m": float(self.committed_distance_m),
            "route_lateral_m": radar.get("_route_lateral_m"),
            "dense_handoff": {
                "candidate": bool(dense_handoff_candidate),
                "confirm_frames": int(self.dense_handoff_frames),
                "latched": bool(self.dense_handoff_latched),
            },
            "flow": flow,
            "tracked_flow": tracked_flow,
            "minimum_gap_ttc_s": float(minimum_gap_ttc),
            "route_horizon": radar.get("_route_horizon"),
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
            "model_steer_preserved": bool(
                _number(getattr(result, "steer", 0.0))
                == _number(getattr(control, "steer", 0.0))
            ),
        }
        if self.state in (
            self.COMMITTED_ENTRY, self.STAGED_BRAKE,
            self.STAGED_ADVANCE, self.FINAL_WAIT,
            self.MERGE_ENTRY, self.ROUTE_FOLLOW,
        ):
            # The first formal online-waypoint run removed all collisions and
            # completed the route.  A controlled ablation that returned the
            # model at 30 m increased route departure and hit a pole, so keep
            # the last-good online owner through ROUTE_FOLLOW.  The added
            # route-lateral value is telemetry only.
            debug["ordinary_route_waypoint"] = {
                "active": True,
                "phase": "class44_turning_flow_entry",
                "steer_kp": _number(
                    self.config.get("entry_waypoint_steer_kp"), 0.72
                ),
                "model_steer_blend": _number(
                    self.config.get("entry_waypoint_model_blend"), 0.25
                ),
                "steer_limit": _number(
                    self.config.get("entry_waypoint_steer_limit"), 0.65
                ),
                "steer_rate_per_frame": _number(
                    self.config.get("entry_waypoint_steer_rate"), 0.10
                ),
                "high_speed_start_mps": 7.5,
                "high_speed_steer_limit": 0.34,
                "near_switch_distance_m": 2.0,
                "prefer_lower_error_far": True,
                "far_lifecycle_max_distance_m": _number(
                    self.config.get("entry_far_lifecycle_max_distance_m"), 25.0
                ),
                "far_lifecycle_error_margin_rad": _number(
                    self.config.get("entry_far_lifecycle_error_margin_rad"), 0.10
                ),
            }
            if self.dense_handoff_latched and self.state not in (
                self.STAGED_BRAKE, self.STAGED_ADVANCE, self.FINAL_WAIT
            ):
                debug["ordinary_dense_route_pid"] = {
                    "active": True,
                    "phase": "class44_dense_route_follow",
                    "lookahead_base_m": _number(
                        self.config.get("dense_pid_lookahead_base_m"), 5.0
                    ),
                    "lookahead_speed_s": _number(
                        self.config.get("dense_pid_lookahead_speed_s"), 0.45
                    ),
                    "lookahead_min_m": 5.0,
                    "lookahead_max_m": 9.0,
                    "steer_limit": _number(
                        self.config.get("dense_pid_steer_limit"), 0.48
                    ),
                    "steer_rate_per_frame": _number(
                        self.config.get("entry_waypoint_steer_rate"), 0.10
                    ),
                    "pose_filter_frames": int(
                        self.config.get("dense_pid_pose_filter_frames", 7)
                    ),
                }
        return result, debug


def build_rule():
    return Rule44(get_config(44))
