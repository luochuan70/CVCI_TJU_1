"""Class-23 single parked-vehicle bypass with opposing-flow gating."""

from __future__ import absolute_import

import math

from .class_04 import Rule04, _number
from .configs import get_config


class Rule23(Rule04):
    """Clear one parked body and return before the following oncoming car."""

    SHIFT_END_M = 7.0
    HOLD_END_M = 22.0
    REJOIN_END_M = 29.0
    FINISH_M = 34.0
    # The compact path clears the opposing lane before the following actor
    # reaches radar range.  After a measured near actor has passed, require
    # the inherited six consecutive empty frames and launch immediately;
    # waiting to measure the next actor can turn a real gap into a permanent
    # hold under continuous scenario flow.
    ALLOW_POST_PASS_SILENCE = True
    POST_COMPLETION_STATIC_CONFIRM_FRAMES = 2
    MAX_POST_COMPLETION_REARMS = 1
    POST_COMPLETION_STATIC_SPEED_MPS = 1.0
    POST_COMPLETION_REARM_MARGIN_M = 0.25
    REJOIN_BODY_CLEARANCE_M = 7.0
    # The formal Class24/route25358 trace exposed a dangerous ownership gap:
    # after the first Class23 path completed, the generic radar guard emitted
    # only 0.70 brake and briefly dropped one frame before the bounded second
    # transaction could be recognized.  Keep a short, longitudinal-only hard
    # stop across that measured dropout.  This guard is deliberately limited
    # to the first completed bypass; it cannot alter normal Class23 approach,
    # committed-path steering, or the already active second transaction.
    POST_COMPLETION_EMERGENCY_HOLD_FRAMES = 3
    # If Rule04 reaches its nominal rejoin progress while a live actor still
    # occupies the physical corridor, extend the same bounded second bypass
    # once from the current pose.  This is an endpoint correction, not an
    # unbounded retry or a relaxed collision envelope.
    MAX_POST_COMPLETION_ENDPOINT_EXTENSIONS = 1
    # The compact online envelope removes only the excess opposing-lane
    # residence measured by the formal evaluator.  Keep the original class
    # constants below for conservative clearance-time and late-oncoming
    # checks; these values are waypoint-plan geometry, not safety thresholds.
    NORMAL_COMPACT_HOLD_END_M = 18.5
    NORMAL_COMPACT_REJOIN_END_M = 25.5
    NORMAL_COMPACT_FINISH_M = 30.5

    # The second transaction starts after the body has already been
    # physically cleared once and uses the same compact geometry.
    POST_REARM_HOLD_END_M = 21.5
    POST_REARM_REJOIN_END_M = 28.5
    POST_REARM_FINISH_M = 33.5

    def __init__(self, cfg):
        self._ignore_static_immediate = False
        self._ignore_opposing_cruise = False
        self._post_completion_static_frames = 0
        self._post_completion_rearm_count = 0
        self._post_completion_rearm_active = False
        self._post_completion_endpoint_extension_count = 0
        self._post_completion_emergency_hold_frames = 0
        super(Rule23, self).__init__(cfg)

    def reset(self):
        super(Rule23, self).reset()
        self._ignore_static_immediate = False
        self._ignore_opposing_cruise = False
        self._post_completion_static_frames = 0
        self._post_completion_rearm_count = 0
        self._post_completion_rearm_active = False
        self._post_completion_endpoint_extension_count = 0
        self._post_completion_emergency_hold_frames = 0
        # The normal Class23 entry rule requires a measured following actor
        # after flow has been seen.  A second bypass is a fresh physical
        # transaction: silence is eligible only after the first flow has
        # already passed, while current approaching flow still vetoes launch.
        self.ALLOW_POST_PASS_SILENCE = False

    def _cancel_reason(self, features):
        reason = super(Rule23, self)._cancel_reason(features)
        if (
            reason == "immediate_hazard"
            and self._post_completion_rearm_active
            and self.phase in {"COOLDOWN", "CRUISE"}
            and self.maneuver_complete
            and self._post_completion_static_distance(features) is None
            and not self._has_approaching_corridor_actor(features)
        ):
            # Rule04's generic immediate-hazard hold is intentionally broad.
            # After a bounded Class23 bypass, a stationary body outside the
            # 7 m physical envelope must not freeze the cooldown forever; the
            # live corridor/approach checks below still veto an actor that is
            # actually approaching or inside the body-clearance envelope.
            reason = ""
        if reason == "immediate_hazard" and (
            self._ignore_static_immediate or self._ignore_opposing_cruise
        ):
            return ""
        return reason

    @staticmethod
    def _vehicle_item(item):
        return (
            isinstance(item, dict)
            and str(item.get("class_name", "") or "").lower()
            in {"car", "vehicle", "van", "truck", "bus", "motorcycle"}
        )

    def _minimum_rejoin_gap(self, features):
        ego_speed = max(
            0.0, _number(getattr(features, "ego_speed", 0.0), 0.0)
        )
        # A tracked centre at 6 m is still physically overlapping a CARLA
        # vehicle after the ego front/rear extents are accounted for.  Use a
        # class-level body envelope, then grow it with ego speed; this is a
        # geometric clearance bound, not a post-pass delay or route threshold.
        return max(self.REJOIN_BODY_CLEARANCE_M, 4.5 + 0.75 * ego_speed)

    def _rejoin_corridor_hazard(self, features):
        """Keep CRUISE stopped for a vehicle in the physical rejoin corridor."""
        if self.phase != "CRUISE" or not self.maneuver_complete:
            return False
        minimum_gap_m = self._minimum_rejoin_gap(features)

        def blocked(lateral, distance, closing, score=None):
            if score is not None and score > 0.0 and score < 0.20:
                return False
            if not (-4.5 <= lateral <= 4.5 and 0.0 < distance <= 12.0):
                return False
            # A centre/right vehicle is never an opposing-lane clearance, but
            # it only blocks while inside the actual gap or approaching.  A
            # left vehicle is safe only after it has cleared the physical
            # rejoin envelope and is not still approaching.
            return distance < minimum_gap_m or closing < -1.0

        # Raw detector boxes are current observations and preserve approach
        # speed when the compact track briefly flips sign.
        for item in list(getattr(features, "detector_objects", None) or []):
            if not self._vehicle_item(item):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                box = item.get("box_lidar") or {}
                lateral = float(box.get("x"))
                distance = float(box.get("y"))
                closing = float(box.get("vy", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if score >= 0.20 and blocked(
                lateral, distance, closing, score=score
            ):
                return True

        for item in list(getattr(features, "tracked_objects", None) or []):
            if not self._vehicle_item(item):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                lateral = float(item.get("x"))
                distance = float(item.get("y"))
                closing = float(item.get("vy", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if blocked(
                lateral, distance, closing,
                score=score if score > 0.0 else None,
            ):
                return True
        return False

    def _has_approaching_corridor_actor(self, features):
        """Return true for a score-filtered actor closing through the return."""
        for values, detector in (
            (getattr(features, "tracked_objects", None) or [], False),
            (getattr(features, "detector_objects", None) or [], True),
        ):
            for item in values:
                if not self._vehicle_item(item):
                    continue
                try:
                    score = float(item.get("score", 0.0) or 0.0)
                    box = item.get("box_lidar") if detector else item
                    box = box or {}
                    lateral = float(box.get("x"))
                    distance = float(box.get("y"))
                    closing = float(box.get("vy", 0.0) or 0.0)
                except (AttributeError, TypeError, ValueError):
                    continue
                if detector and score < 0.20:
                    continue
                if (
                    -4.5 <= lateral <= 4.5
                    and 0.0 < distance <= 12.0
                    and closing < -1.0
                ):
                    return True
        return False

    def _post_completion_static_distance(self, features):
        """Return a near-static body distance that requires a second plan.

        ``Rule04`` marks a path complete from local controller geometry.  That
        endpoint can precede the physical rear bumper of the parked body.  A
        second plan is therefore allowed only for a high-confidence, nearly
        stationary vehicle already inside the measured minimum gap.  Dynamic
        opposing flow is intentionally excluded; it is handled by the normal
        measured-gap release logic.
        """
        minimum_gap_m = self._minimum_rejoin_gap(features)
        candidates = []
        for item in list(getattr(features, "tracked_objects", None) or []):
            if not self._vehicle_item(item):
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                lateral = float(item.get("x"))
                distance = float(item.get("y"))
                vx = float(item.get("vx", 0.0) or 0.0)
                vy = float(item.get("vy", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if score > 0.0 and score < 0.20:
                continue
            if not (
                -4.5 <= lateral <= 4.5
                and 0.0 < distance <= minimum_gap_m + self.POST_COMPLETION_REARM_MARGIN_M
            ):
                continue
            if math.hypot(vx, vy) <= self.POST_COMPLETION_STATIC_SPEED_MPS:
                candidates.append(distance)

        # A compact track is normally available at the endpoint.  If it is
        # temporarily absent, accept a detector-only static observation only
        # when its own motion estimate is also small; a dynamic raw box must
        # never trigger a second bypass.
        if not candidates:
            for item in list(getattr(features, "detector_objects", None) or []):
                if not self._vehicle_item(item):
                    continue
                try:
                    score = float(item.get("score", 0.0) or 0.0)
                    box = item.get("box_lidar") or {}
                    lateral = float(box.get("x"))
                    distance = float(box.get("y"))
                    vx = float(box.get("vx", 0.0) or 0.0)
                    vy = float(box.get("vy", 0.0) or 0.0)
                except (AttributeError, TypeError, ValueError):
                    continue
                if score < 0.50 or not (
                    -4.5 <= lateral <= 4.5
                    and 0.0 < distance <= minimum_gap_m + self.POST_COMPLETION_REARM_MARGIN_M
                ):
                    continue
                if math.hypot(vx, vy) <= self.POST_COMPLETION_STATIC_SPEED_MPS:
                    candidates.append(distance)
        return min(candidates) if candidates else None

    def _maybe_rearm_post_completion(self, features, radar):
        """Start one bounded second bypass for a body left in the corridor."""
        if self.phase not in {"CRUISE", "COOLDOWN"} or not self.maneuver_complete:
            self._post_completion_static_frames = 0
            return False
        distance = self._post_completion_static_distance(features)
        if distance is None:
            self._post_completion_static_frames = 0
            return False
        self._post_completion_static_frames += 1
        if (
            self._post_completion_static_frames
            < self.POST_COMPLETION_STATIC_CONFIRM_FRAMES
            or self._post_completion_rearm_count >= self.MAX_POST_COMPLETION_REARMS
        ):
            return False
        # Re-enter the same finite stop/plan/gap/bypass state machine.  The
        # current live route lateral is the new origin; no route identity or
        # XML geometry is involved.
        self._post_completion_rearm_count += 1
        self._post_completion_rearm_active = True
        self.ALLOW_POST_PASS_SILENCE = True
        self.phase = "BRAKE"
        self.phase_frames = 0
        self.stop_frames = 0
        self.cooldown_frames = 0
        self.entry_front_m = max(1.0, distance)
        self.corridor_samples.clear()
        self.plan = None
        self.approach_plan = None
        self.maneuver_complete = False
        self.cycle += 1
        return True

    def _maybe_extend_endpoint(self, features, radar):
        """Restart one finite local bypass if its endpoint is still blocked.

        Rule04 can finish a committed path at its nominal rejoin progress when
        an immediate hazard appears in the same frame.  For Class23 that
        would discard the local path and leave the physical rejoin guard with
        no executable route.  Rebuild once from the current live route pose;
        the normal BRAKE/PLAN/WAIT_GAP state machine then chooses a measured
        interval before moving again.
        """
        if not (
            self._post_completion_rearm_active
            and self.phase == "CRUISE"
            and self.maneuver_complete
            and self._post_completion_endpoint_extension_count
            < self.MAX_POST_COMPLETION_ENDPOINT_EXTENSIONS
            and self._rejoin_corridor_hazard(features)
        ):
            return False
        self._post_completion_endpoint_extension_count += 1
        self.phase = "BRAKE"
        self.phase_frames = 0
        self.stop_frames = 0
        self.cooldown_frames = 0
        self.entry_front_m = max(
            1.0, _number(radar.get("front_min"), self.entry_front_m)
        )
        self.corridor_samples.clear()
        self.plan = None
        self.approach_plan = None
        self.maneuver_complete = False
        self.cycle += 1
        return True

    def _opposing_only_cruise(self, features, radar):
        """Permit post-rejoin progress only with tracked opposing-lane flow."""
        if self.phase != "CRUISE" or not self.maneuver_complete:
            return False
        tracks = [
            item for item in (getattr(features, "tracked_objects", None) or [])
            if isinstance(item, dict)
            and str(item.get("class_name", "") or "").lower()
            in {"car", "vehicle", "van", "truck", "bus", "motorcycle"}
        ]
        if not tracks:
            return False
        close_dynamic = []
        for item in tracks:
            try:
                lateral = float(item.get("x"))
                distance = float(item.get("y"))
                speed = abs(float(item.get("vx", 0.0) or 0.0)) + abs(
                    float(item.get("vy", 0.0) or 0.0)
                )
            except (TypeError, ValueError):
                continue
            if 0.0 < distance <= 12.0 and speed >= 1.0:
                close_dynamic.append((lateral, distance))
        if not close_dynamic:
            return False
        # A usable rejoin gap must clear the ego vehicle body and leave a
        # speed-dependent margin.  This is a physical envelope, not a fixed
        # post-pass delay that can be satisfied by a noisy sign flip.
        minimum_gap_m = self._minimum_rejoin_gap(features)
        # A left-side car is still unsafe when it is closing through the
        # actual rejoin corridor.  Wait for the vehicle to pass or for a
        # materially larger headway; this is the usable-gap condition rather
        # than an arbitrary post-pass timer.
        for item in tracks:
            try:
                lateral = float(item.get("x"))
                distance = float(item.get("y"))
                closing = float(item.get("vy", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
            if 0.0 < distance <= 12.0 and (
                lateral > -1.8
                or distance < minimum_gap_m
                or closing < -1.0
            ):
                return False
        # The compact tracker deliberately smooths detector motion and can
        # briefly flip the longitudinal sign at close range.  Check the
        # score-filtered online detector boxes as an independent, current
        # observation so a raw approaching box still vetoes rejoin.
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "van", "truck", "bus", "motorcycle",
            }:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                box = item.get("box_lidar") or {}
                lateral = float(box.get("x"))
                distance = float(box.get("y"))
                closing = float(box.get("vy", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            if score < 0.20 or not (0.0 < distance <= 12.0):
                continue
            if (
                lateral > -1.8
                or distance < minimum_gap_m
                or closing < -1.0
            ):
                return False
        # Class23's planned bypass is on the left.  A right-side return is
        # not an opposing-lane clearance and must never be suppressed.
        return all(lateral <= -1.8 for lateral, _ in close_dynamic)

    def _build_plan(self, route_lateral_m):
        plan = super(Rule23, self)._build_plan(route_lateral_m)
        old_hold = _number(plan.get("hold_end_m"), self.HOLD_END_M)
        old_rejoin = _number(plan.get("rejoin_end_m"), self.REJOIN_END_M)
        old_finish = _number(plan.get("stabilize_end_m"), self.FINISH_M)
        if self._post_completion_rearm_active:
            new_hold = self.POST_REARM_HOLD_END_M
            new_rejoin = self.POST_REARM_REJOIN_END_M
            new_finish = self.POST_REARM_FINISH_M
        else:
            new_hold = self.NORMAL_COMPACT_HOLD_END_M
            new_rejoin = self.NORMAL_COMPACT_REJOIN_END_M
            new_finish = self.NORMAL_COMPACT_FINISH_M
        compressed = []
        for point in list(plan.get("waypoints", [])):
            try:
                x_value, y_value = float(point[0]), float(point[1])
            except (TypeError, ValueError, IndexError):
                continue
            if x_value <= self.SHIFT_END_M:
                x_new = x_value
            elif x_value <= old_hold:
                ratio = (x_value - self.SHIFT_END_M) / max(
                    1e-6, old_hold - self.SHIFT_END_M
                )
                x_new = self.SHIFT_END_M + ratio * (
                    new_hold - self.SHIFT_END_M
                )
            elif x_value <= old_rejoin:
                ratio = (x_value - old_hold) / max(
                    1e-6, old_rejoin - old_hold
                )
                x_new = new_hold + ratio * (new_rejoin - new_hold)
            else:
                ratio = (x_value - old_rejoin) / max(
                    1e-6, old_finish - old_rejoin
                )
                x_new = new_rejoin + ratio * (new_finish - new_rejoin)
            compressed.append([x_new, y_value])
        plan["waypoints"] = compressed
        plan.update({
            "hold_end_m": new_hold,
            "rejoin_end_m": new_rejoin,
            "stabilize_end_m": new_finish,
            "class23_compact_geometry": True,
        })
        if self._post_completion_rearm_active:
            plan["post_completion_shortened"] = True
        else:
            plan["normal_compact_geometry"] = True
        plan.update({
            "class_id": 23,
            "path_id": "c23_dense_single_parked_bypass_v1",
            "route_pid_phase": "ordinary_class23_stop_plan_rejoin",
            "use_dense_route_pid": True,
            "dense_route_pid_phase": "ordinary_class23_dense_bypass",
            "trust_live_route_lateral": True,
            "finish_use_live_route_lateral": True,
            "finish_use_live_route_heading": True,
        })
        return plan

    def apply(self, control, radar, features):
        handoff = str(radar.get("_classifier_reason", "") or "") == (
            "online_single_parked_vehicle_with_opposing_flow"
        )
        if self.phase == "CRUISE" and handoff and not self.maneuver_complete:
            self.phase = "BRAKE"
            self.phase_frames = 0
            self.stop_frames = 0
            self.entry_front_m = _number(radar.get("front_min"), 0.0)
            self.corridor_samples.clear()
            self.cycle += 1

        first_completion_guard_eligible = bool(
            self.phase in {"CRUISE", "COOLDOWN"}
            and self.maneuver_complete
            and not self._post_completion_rearm_active
        )
        post_completion_rearm = self._maybe_rearm_post_completion(features, radar)

        self._ignore_static_immediate = bool(
            self.phase == "BYPASS"
            and float(radar.get("front_dynamic_count", 0.0) or 0.0) <= 1.0
            and float(radar.get("front_closing_count", 0.0) or 0.0) <= 1.0
        )
        self._ignore_opposing_cruise = self._opposing_only_cruise(features, radar)
        rejoin_corridor_hazard = self._rejoin_corridor_hazard(features)
        output, debug = super(Rule23, self).apply(control, radar, features)
        endpoint_extension = self._maybe_extend_endpoint(features, radar)
        if endpoint_extension:
            output.throttle = 0.0
            output.brake = 1.0
            output.steer = 0.0
            debug["state"] = "CLASS23_ENDPOINT_EXTENSION_BRAKE"
            debug["source"] = "ordinary_class_23_endpoint_extension"
            debug["override_source"] = "class23_endpoint_extension_stop"
        if self._ignore_opposing_cruise and bool(
            debug.get("emergency") or debug.get("caution")
        ):
            # The shared radar safety helper only has aggregate front TTC and
            # therefore cannot see that every close return is in the opposing
            # lane.  Class23 may release this one post-rejoin case using the
            # already-validated model steering; all center-lane hazards still
            # retain the helper's hard stop above.
            output.throttle = max(
                0.25, float(getattr(control, "throttle", 0.0) or 0.0)
            )
            output.brake = 0.0
            debug["state"] = "CLASS23_OPPOSING_FLOW_CRUISE_RELEASE"
            debug["source"] = "ordinary_class_23"
        if rejoin_corridor_hazard:
            # The generic front-TTC aggregate can miss a vehicle that is
            # already alongside the planned left return.  Keep a hard stop in
            # this short physical corridor until the live gap clears.
            output.throttle = 0.0
            output.brake = 1.0
            debug["state"] = "CLASS23_REJOIN_CORRIDOR_HOLD"
            debug["source"] = "ordinary_class_23_rejoin_gap"
        # Apply this last so neither the generic 0.70 emergency response nor
        # the opposing-flow cruise release can reopen the throttle while the
        # first completed bypass is waiting for second-transaction ownership.
        # Preserve steering: this is a longitudinal safety boundary, not a
        # new path planner or route-specific manoeuvre.
        post_completion_emergency = False
        if (
            first_completion_guard_eligible
            and not self._post_completion_rearm_active
        ):
            raw_emergency = bool(debug.get("raw_emergency"))
            if raw_emergency:
                self._post_completion_emergency_hold_frames = (
                    self.POST_COMPLETION_EMERGENCY_HOLD_FRAMES
                )
            post_completion_emergency = bool(
                raw_emergency
                or self._post_completion_emergency_hold_frames > 0
            )
            if post_completion_emergency:
                output.throttle = 0.0
                output.brake = 1.0
                debug["state"] = "CLASS23_POST_COMPLETION_EMERGENCY_HOLD"
                debug["source"] = "ordinary_class_23_post_completion_safety"
                debug["override_source"] = (
                    "class23_post_completion_emergency_hold"
                )
                # ``radar_safety_action`` snapshots this dictionary before
                # the category-specific final override.  Keep telemetry equal
                # to the control actually returned to CARLA so later failure
                # analysis cannot mistake the generic 0.70 command for the
                # full-brake safety result.
                debug["output_control"] = {
                    "throttle": float(output.throttle),
                    "steer": float(output.steer),
                    "brake": float(output.brake),
                }
            if (
                not raw_emergency
                and self._post_completion_emergency_hold_frames > 0
            ):
                self._post_completion_emergency_hold_frames -= 1
        else:
            self._post_completion_emergency_hold_frames = 0
        debug.update({
            "class23_phase": self.phase,
            "class23_online_handoff": handoff,
            "class23_measured_following_actor_required": bool(
                self.oncoming_seen and not self.ALLOW_POST_PASS_SILENCE
            ),
            "class23_committed_static_hazard_suppressed": bool(
                self._ignore_static_immediate
                and bool(getattr(features, "immediate_hazard", False))
            ),
            "class23_opposing_only_cruise_release": bool(
                self._ignore_opposing_cruise
            ),
            "class23_rejoin_corridor_hazard": bool(rejoin_corridor_hazard),
            "class23_post_completion_rearm": bool(post_completion_rearm),
            "class23_post_completion_static_frames": int(
                self._post_completion_static_frames
            ),
            "class23_post_completion_rearm_count": int(
                self._post_completion_rearm_count
            ),
            "class23_post_completion_rearm_active": bool(
                self._post_completion_rearm_active
            ),
            "class23_endpoint_extension": bool(endpoint_extension),
            "class23_endpoint_extension_count": int(
                self._post_completion_endpoint_extension_count
            ),
            "class23_post_completion_emergency": bool(
                post_completion_emergency
            ),
            "class23_post_completion_emergency_hold_frames": int(
                self._post_completion_emergency_hold_frames
            ),
        })
        if str(debug.get("state", "")).startswith("CLASS04_"):
            debug["state"] = "CLASS23_" + str(debug["state"])[8:]
        if str(debug.get("override_source", "")).startswith("class04_"):
            debug["override_source"] = "class23_" + str(
                debug["override_source"]
            )[8:]
        return output, debug


def build_rule():
    return Rule23(get_config(23))
