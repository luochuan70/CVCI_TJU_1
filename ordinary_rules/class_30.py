"""Class-30 open-door bypass with a measured opposing-flow gap.

The parked door is a physical lane blockage, not a transient object that can
be cleared by alternating throttle and radar braking.  Reuse the already
validated Class-23 stop/plan/gap/compact-bypass controller, but enter it
through a slow straight approach: the Class-30 silhouette is visible while
the obstacle is still roughly 25--30 m away, so stopping at classification
time would leave too little route time for the manoeuvre.
"""

from __future__ import absolute_import

import math

from .class_23 import Rule23
from .configs import get_config


class Rule30(Rule23):
    """Approach the door, stop at the live hazard, then bypass on the left."""

    HANDOFF_REASONS = frozenset({
        "online_persistent_weak_open_door_vehicle",
        "online_dense_parked_open_door_corridor",
    })
    APPROACH_TARGET_SPEED_MPS = 3.0
    CONFIRMED_CONSTRUCTION_APPROACH_SPEED_MPS = 3.0
    # Unlike Class23's longer roadside blockage, the official Class30 family
    # contains one car-sized parked body.  The live stop-to-centre distance is
    # about 9 m; holding full offset through 13 m clears its rear bumper plus
    # margin, then returns before the 20 m scoring boundary.  The inherited
    # 8.5 m/s construction speed overshot both sides of the curved Town11
    # route, so give the public PID enough time to track this compact path.
    NORMAL_COMPACT_HOLD_END_M = 13.0
    NORMAL_COMPACT_REJOIN_END_M = 19.0
    NORMAL_COMPACT_FINISH_M = 23.0
    BYPASS_TARGET_SPEED_MPS = 4.5
    STOP_SIGN_CONFIRM_FRAMES = 20
    POST_STOP_SIGN_CONFIRM_FRAMES = 20
    POST_STOP_SIGN_CLEAR_FRAMES = 5
    POST_STOP_SIGN_TARGET_SPEED_MPS = 3.0
    MAX_APPROACH_SEGMENTS = 2

    def __init__(self, cfg):
        self._stop_sign_stop_frames = 0
        self._stop_sign_released = False
        self._post_stop_sign_stop_frames = 0
        self._post_stop_sign_released = False
        self._post_stop_sign_clear_frames = 0
        self._dense_corridor_handoff = False
        self._approach_segment_count = 0
        self._approach_total_m = 0.0
        super(Rule30, self).__init__(cfg)

    def reset(self):
        super(Rule30, self).reset()
        self._stop_sign_stop_frames = 0
        self._stop_sign_released = False
        self._post_stop_sign_stop_frames = 0
        self._post_stop_sign_released = False
        self._post_stop_sign_clear_frames = 0
        self._dense_corridor_handoff = False
        self._approach_segment_count = 0
        self._approach_total_m = 0.0

    @staticmethod
    def _map_stop_sign_without_traffic_light(features):
        """Separate a completed stop sign from a signal that needs green.

        The shared adapter deliberately combines map infrastructure with RGB
        colour.  On the target road, the only high-confidence infrastructure
        is a StopSign, while yellow road pixels keep the aggregate state
        active forever.  A real TrafficLight candidate must always veto this
        Class30-private release.
        """
        stop_sign = False
        traffic_light = False
        for item in list(
            getattr(features, "traffic_light_candidates", None) or []
        ):
            if not isinstance(item, dict):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            source = str(item.get("source", "") or "").lower()
            class_name = str(item.get("class_name", "") or "").lower()
            if source == "map":
                values = item.get("box") or []
                try:
                    distance = math.hypot(float(values[0]), float(values[1]))
                except (IndexError, TypeError, ValueError):
                    distance = 999.0
                if score >= 0.35 and distance <= 16.0:
                    stop_sign = stop_sign or class_name == "stopsign"
                    traffic_light = traffic_light or class_name == "trafficlight"
            elif score >= 0.45:
                # This mirrors the shared adapter's minimum confidence for a
                # detector traffic-light candidate.
                traffic_light = True
        return bool(stop_sign and not traffic_light)

    def _cancel_reason(self, features):
        reason = super(Rule30, self)._cancel_reason(features)
        if (
            reason == "immediate_hazard"
            and self._dense_corridor_handoff
            and self.phase == "APPROACH"
            and not self._centered_close_vehicle(features)
        ):
            # The dense online family cue is formed by parked bodies on both
            # road edges.  Their side-only emergency returns are expected
            # during the route-centred approach and are not the opened door.
            # A centred close body still stops immediately; all unrelated
            # cancellation reasons remain authoritative.
            return ""
        if (
            reason == "immediate_hazard"
            and self.maneuver_complete
            and self.phase in {"COOLDOWN", "CRUISE"}
            and not self._centered_close_vehicle(features)
        ):
            # The compact path has already passed and rejoined around the
            # single open-door actor.  Its body can remain less than a metre
            # forward but a full lane to the side for the rest of cooldown;
            # treating that residual side return as a new forward blockage
            # deadlocks the route after a successful manoeuvre.  Centred
            # vehicles remain authoritative, as do the independent raw-radar
            # and rejoin-corridor guards in Rule23.
            return ""
        if reason == "red_light" and self._stop_sign_released:
            return ""
        return reason

    def _build_plan(self, route_lateral_m):
        plan = super(Rule30, self)._build_plan(route_lateral_m)
        plan.update({
            "class_id": 30,
            "path_id": "c30_open_door_compact_left_bypass_v1",
            "route_pid_phase": "ordinary_class30_stop_plan_rejoin",
            "dense_route_pid_phase": "ordinary_class30_dense_bypass",
            "class30_open_door_geometry": True,
            "target_speed_mps": self.BYPASS_TARGET_SPEED_MPS,
        })
        return plan

    def _maybe_rearm_post_completion(self, features, radar):
        """Class30 owns one door actor and therefore one committed bypass."""
        self._post_completion_static_frames = 0
        return False

    def apply(self, control, radar, features):
        handoff_reason = str(radar.get("_classifier_reason", "") or "")
        handoff = handoff_reason in self.HANDOFF_REASONS
        stale_handoff_feedback_discarded = False
        approach_segment_extended = False
        dense_clutter_clear_after_call = False
        # The router transfers ownership while Rule04 is stopped at the early
        # weak-door silhouette.  Resume only a slow route-centred approach;
        # inherited Rule04 logic makes the later close hazard an irreversible
        # stop and then invokes the compact Class23 planner.
        if self.phase == "CRUISE" and handoff and not self.maneuver_complete:
            self._dense_corridor_handoff = bool(
                handoff_reason
                == "online_dense_parked_open_door_corridor"
            )
            if self._dense_corridor_handoff:
                # Make Rule04's existing approach logic treat raw side-only
                # emergency points as roadside context.  Rule30 pins the
                # confirmed-context target speed to 3 m/s above, so this does
                # not inherit Class04's faster construction approach.
                self.construction_context_seen = True
            route_lateral = radar.get("_route_lateral_m")
            self.cycle += 1
            self.approach_plan = self._build_approach_plan(route_lateral)
            self.approach_plan.update({
                "class_id": 30,
                "path_id": "c30_open_door_slow_approach_v1",
                "target_speed_mps": self.APPROACH_TARGET_SPEED_MPS,
                "route_pid_phase": "ordinary_class30_door_approach",
            })
            self.phase = "APPROACH"
            self.phase_frames = 0
            self.approach_progress_m = 0.0
            self._approach_segment_count = 1
            self._approach_total_m = 0.0
            # The handoff happens one frame after Rule04 produced its final
            # action, so the router necessarily carries Rule04's previous
            # controller feedback into this first Rule30 call.  A completed
            # 28 m Class04 staging path must not complete the newly-created
            # Class30 approach before its first waypoint is even executed.
            # Clear only this ownership-boundary sample; the next frame is
            # feedback from the Class30 path emitted below.
            if radar.get("_local_path_feedback"):
                radar = dict(radar)
                radar.pop("_local_path_feedback", None)
                stale_handoff_feedback_discarded = True

        # A route-centred controller endpoint is not evidence that the door
        # is close enough to bypass.  Continue once when the current segment
        # completed without any live hazard/cancellation.  This is bounded:
        # after two segments the inherited stop/plan fallback remains active.
        if self.phase == "APPROACH" and self.approach_plan:
            feedback = radar.get("_local_path_feedback") or {}
            progress = float(
                feedback.get("x_m", self.approach_progress_m) or 0.0
            )
            reached_endpoint = bool(
                feedback.get("done")
                or progress >= float(self.APPROACH_DISTANCE_M)
            )
            current_cancel = self._cancel_reason(features)
            centered_close = self._centered_close_vehicle(features)
            if (
                reached_endpoint
                and not current_cancel
                and not centered_close
                and self._approach_segment_count
                < self.MAX_APPROACH_SEGMENTS
            ):
                self._approach_total_m += max(0.0, progress)
                self._approach_segment_count += 1
                self.cycle += 1
                self.approach_plan = self._build_approach_plan(
                    radar.get("_route_lateral_m")
                )
                self.approach_plan.update({
                    "class_id": 30,
                    "path_id": "c30_open_door_slow_approach_v1",
                    "target_speed_mps": self.APPROACH_TARGET_SPEED_MPS,
                    "route_pid_phase": "ordinary_class30_door_approach",
                })
                self.phase_frames = 0
                self.approach_progress_m = 0.0
                radar = dict(radar)
                radar.pop("_local_path_feedback", None)
                approach_segment_extended = True
                dense_clutter_clear_after_call = bool(
                    self._dense_corridor_handoff
                )

        stop_sign_only = bool(
            self.phase == "BYPASS"
            and bool(getattr(features, "red_light_active", False))
            and self._map_stop_sign_without_traffic_light(features)
        )
        stopped = abs(float(getattr(features, "ego_speed", 0.0) or 0.0)) <= 0.12
        if stop_sign_only and stopped and not self._stop_sign_released:
            self._stop_sign_stop_frames += 1
            if self._stop_sign_stop_frames >= self.STOP_SIGN_CONFIRM_FRAMES:
                self._stop_sign_released = True
        elif not stop_sign_only and not self._stop_sign_released:
            self._stop_sign_stop_frames = 0

        output, debug = super(Rule30, self).apply(control, radar, features)
        if dense_clutter_clear_after_call:
            # The first dense segment has cleared the roadside row.  Starting
            # with the next observation, every immediate hazard is again
            # authoritative even when the detector centre remains lateral.
            self._dense_corridor_handoff = False

        # The compact bypass can encounter a second StopSign just after its
        # endpoint.  This is a new traffic-control transaction: the release
        # confirmed during BYPASS must not silently carry across it.  Require
        # another full stop, retain all real-light/VRU/hazard vetoes, and only
        # then resume a low-speed route-following command.  This override is
        # Class30-private and does not alter the shared signal adapter.
        post_stop_sign_only = bool(
            self.maneuver_complete
            and self.phase in {"COOLDOWN", "CRUISE"}
            and bool(getattr(features, "red_light_active", False))
            and self._map_stop_sign_without_traffic_light(features)
        )
        post_stopped = abs(
            float(getattr(features, "ego_speed", 0.0) or 0.0)
        ) <= 0.12
        if post_stop_sign_only:
            self._post_stop_sign_clear_frames = 0
            if post_stopped and not self._post_stop_sign_released:
                self._post_stop_sign_stop_frames += 1
                if (
                    self._post_stop_sign_stop_frames
                    >= self.POST_STOP_SIGN_CONFIRM_FRAMES
                ):
                    self._post_stop_sign_released = True
        else:
            self._post_stop_sign_clear_frames += 1
            if (
                self._post_stop_sign_clear_frames
                >= self.POST_STOP_SIGN_CLEAR_FRAMES
            ):
                self._post_stop_sign_stop_frames = 0
                self._post_stop_sign_released = False

        post_release_safe = bool(
            post_stop_sign_only
            and self._post_stop_sign_released
            and not bool(getattr(features, "immediate_hazard", False))
            and getattr(features, "front_pedestrian_distance", None) is None
            and not bool(debug.get("class23_rejoin_corridor_hazard"))
            and not bool(debug.get("class23_post_completion_emergency"))
        )
        if post_release_safe:
            speed = abs(float(getattr(features, "ego_speed", 0.0) or 0.0))
            output.steer = float(getattr(control, "steer", 0.0) or 0.0)
            if speed < self.POST_STOP_SIGN_TARGET_SPEED_MPS - 0.5:
                output.throttle = max(
                    0.35, float(getattr(control, "throttle", 0.0) or 0.0)
                )
                output.brake = 0.0
            elif speed > self.POST_STOP_SIGN_TARGET_SPEED_MPS + 0.5:
                output.throttle = 0.0
                output.brake = 0.15
            else:
                output.throttle = max(
                    0.15, float(getattr(control, "throttle", 0.0) or 0.0)
                )
                output.brake = 0.0
            debug["state"] = "CLASS30_POST_STOP_SIGN_RELEASE"
            debug["source"] = "ordinary_class_30_post_stop_sign"
            debug["override_source"] = "class30_post_stop_sign_release"
            debug["output_control"] = {
                "throttle": float(output.throttle),
                "steer": float(output.steer),
                "brake": float(output.brake),
            }
        debug.update({
            "class30_phase": self.phase,
            "class30_online_handoff": bool(handoff),
            "class30_dense_corridor_handoff": bool(
                self._dense_corridor_handoff
            ),
            "class30_open_door_compact_planner": True,
            "class30_stop_sign_only": bool(stop_sign_only),
            "class30_stop_sign_stop_frames": int(
                self._stop_sign_stop_frames
            ),
            "class30_stop_sign_released": bool(
                self._stop_sign_released
            ),
            "class30_stale_handoff_feedback_discarded": bool(
                stale_handoff_feedback_discarded
            ),
            "class30_approach_segment_extended": bool(
                approach_segment_extended
            ),
            "class30_approach_segment_count": int(
                self._approach_segment_count
            ),
            "class30_approach_total_m": float(
                self._approach_total_m
                + (
                    self.approach_progress_m
                    if self.phase == "APPROACH" else 0.0
                )
            ),
            "class30_post_stop_sign_only": bool(post_stop_sign_only),
            "class30_post_stop_sign_stop_frames": int(
                self._post_stop_sign_stop_frames
            ),
            "class30_post_stop_sign_released": bool(
                self._post_stop_sign_released
            ),
            "class30_post_stop_sign_clear_frames": int(
                self._post_stop_sign_clear_frames
            ),
            "class30_post_stop_sign_release_safe": bool(post_release_safe),
        })
        if str(debug.get("state", "")).startswith("CLASS23_"):
            debug["state"] = "CLASS30_" + str(debug["state"])[8:]
        if str(debug.get("override_source", "")).startswith("class23_"):
            debug["override_source"] = "class30_" + str(
                debug["override_source"]
            )[8:]
        return output, debug


def build_rule():
    return Rule30(get_config(30))
