"""Class-05 accident bypass with model-first, hazard-triggered ownership."""

from __future__ import absolute_import

import math
from collections import deque

from .class_04 import Rule04, _number
from .common import clone_control
from .configs import get_config


class Rule05(Rule04):
    """Reuse Class04's proven gap/bypass lifecycle without its early staging.

    The accident category contains several stopped vehicles but not every
    route needs an override: one fixed development route already scores 100
    under the learned controller.  Consequently this rule remains transparent
    until online perception reports a *close stationary* hazard.  It then
    stops, plans once, observes the opposing lane, and treats the stationary
    accident vehicle as the object the plan is intended to clear rather than
    as a reason to cancel that plan forever.
    """

    STATIC_CONFIRM_FRAMES = 2
    # The three accident vehicles begin at the stopped ego's close hazard and
    # span about 22 m.  Clear their left edge, not the centre of the opposing
    # lane, then start returning immediately after the common vehicle train.
    # Accident actors are placed 0.6 half-lane-width toward the road edge,
    # leaving a class-common pass corridor inside the left side of the ego
    # lane.  Use that space instead of driving the ego centre down the middle
    # of the opposing lane for the complete 60 m train footprint.
    LEFT_CORRIDOR_CANDIDATES_M = (-1.45, -1.55, -1.65)
    NOMINAL_LEFT_CORRIDOR_M = -1.55
    # The plan is created only after the ego has stopped about five metres
    # behind the first accident vehicle.  Distances are therefore relative to
    # that stopped pose, not to the scenario trigger tens of metres earlier.
    # Acquire the opposing corridor immediately, retain it beyond the shared
    # 10 m + 6 m three-vehicle train, then let the live-route PID rejoin.
    SHIFT_START_M = 0.0
    SHIFT_END_M = 6.0
    # The old plan started at the route origin and therefore needed a 54 m
    # hold.  Class05 now keeps normal route following until a detector-backed
    # lane vehicle is close, so the local plan starts at the stopped obstacle
    # pose.  The official class geometry has two following accident vehicles
    # at +10 m and +16 m; 30 m clears their bodies with margin before a smooth
    # ten-metre return.
    HOLD_END_M = 30.0
    REJOIN_END_M = 40.0
    FINISH_M = 46.0
    HANDOFF_MIN_PROGRESS_M = 38.0
    HANDOFF_ROUTE_LATERAL_M = 0.45
    MOVING_HOLD_CLEAR_FRAMES = 5
    MOVING_STABLE_CONFIRM_FRAMES = 12
    MOVING_STABLE_POSITION_SPAN_M = 0.55
    SIDE_ROUTE_OFFSET_M = 0.9
    SIDE_ROUTE_MAX_OFFSET_M = 3.2
    # Category corridor used by the original finite Class05 plan that scored
    # 100 on route2643.  The 3.2 m cap remains available to adaptive generic
    # side avoidance, but must not be hard-coded into the static train route
    # now that the direct waypoint PID preserves positional offset correctly.
    # AccidentTwoWays requires a left pass, but the static-latched profile has
    # formal 100-point evidence with this narrow corridor and fast clearance.
    # The 3.2 m cap remains only for adaptive/recovery observations.
    STATIC_DENSE_CORRIDOR_M = abs(NOMINAL_LEFT_CORRIDOR_M)
    # A narrow corridor is used only when the stopped-pose radar directly
    # measures enough physical-left clearance.  If the curved/occluded launch
    # view cannot establish that clearance, reserve the older collision-free
    # 2.5 m corridor instead of assuming the accident body is farther right.
    STATIC_DENSE_CONSERVATIVE_CORRIDOR_M = 2.5
    STATIC_DENSE_NARROW_CLEARANCE_M = 2.6
    STATIC_DENSE_NARROW_MIN_POINTS = 3
    # A position-stable detector track can still carry noisy non-zero velocity
    # and use the adaptive generic corridor.  It represents the same three-car
    # accident train, so a temporary radar dropout must not start returning
    # before the common train envelope has been traversed.
    SIDE_ROUTE_MIN_PASS_M = 24.0
    SIDE_ROUTE_REQUIRED_CLEARANCE_M = 2.6
    SIDE_ROUTE_REJOIN_DISTANCE_M = 8.0
    # The historical finite Class05 plan that formally scored 100 returned
    # over ten metres (30 -> 40 m).  The replacement dense route accidentally
    # compressed that same lateral motion into eight metres.  At 8 m/s this
    # removed most of the left clearance beside the second accident body.
    # Preserve the shorter distance for generic moving-side avoidance, but
    # reproduce the proven category trajectory for the stopped train.
    STATIC_DENSE_REJOIN_DISTANCE_M = 10.0
    # Distance completion alone can lead the commanded cosine return by an
    # entire lane on tight curves.  Keep dense PID ownership until the online
    # public-route lateral estimate confirms that the physical return was
    # actually executed.
    SIDE_ROUTE_HANDOFF_LATERAL_M = 0.65
    SIDE_ROUTE_REJOIN_SETTLE_FRAMES = 5
    SIDE_ROUTE_PASS_FORWARD_M = 3.0
    # The dense stopped-train route runs several metres to the left of the
    # accident bodies.  On the official curved route those bodies remain
    # 3.7--5.0 m forward when they are abreast of the ego corridor, so the
    # generic three-metre gate never latched them.  It then latched the next
    # oncoming car instead and kept the ego in the opposing lane until impact.
    # Use the wider gate only for this category's static-train mode.
    STATIC_DENSE_PASS_FORWARD_M = 6.0
    SIDE_ROUTE_CLEAR_FRAMES = 5
    SIDE_GAP_CLEAR_FRAMES = 6
    # A detector-backed static accident is a three-vehicle train rather than
    # the single short-lived ambient actor handled by the original dense
    # route.  Keep the live-route corridor through the class-common train
    # envelope before allowing the measured radar-clear rejoin.
    # Scenario source defines three right-displaced bodies at 0/10/16 m, but
    # the curved live-radar view can briefly lose the final body before its
    # swept envelope is behind the ego.  The failed 36-point route began its
    # return at 24.46 m and struck the same Impala twice; the formal-100 trace
    # held the corridor until 30.16 m.  Treat the already established 30 m
    # full-train boundary as the earliest return point, while still requiring
    # the measured five-frame disappearance.  This changes no launch/gap or
    # speed policy and remains common to every Class05 route.
    STATIC_DENSE_MIN_PASS_M = 30.0
    # Residual static returns must not leave the ego in opposing flow forever.
    # The formal-100 trajectory begins its ten-metre return at about 30 m.
    STATIC_DENSE_MAX_HOLD_M = 30.0
    # A usable gap is finite.  The formal-100 static trajectory clears the
    # train at about 8.1 m/s rather than remaining exposed at 3.5 m/s.
    STATIC_DENSE_COMMITTED_SPEED_MPS = 8.5
    # Preserve the exact stable dense-route controller used by that trajectory.
    STATIC_DENSE_LOOKAHEAD_M = 4.5
    STATIC_DENSE_TRACK_LOOKAHEAD_M = 4.5
    STATIC_DENSE_TRACK_POSE_FILTER_FRAMES = 5
    STATIC_DENSE_STEER_LIMIT = 0.42
    STATIC_DENSE_TRACK_STEER_LIMIT = 0.42
    STATIC_DENSE_ACQUIRED_ERROR_M = 0.65
    STATIC_OBSERVED_GAP_CLEAR_FRAMES = 3
    # The finite stopped-pose plan is the default: it has a formal 100-point
    # result.  On a curved road, however, its local x/y frame can move the ego
    # to the opposite side of the requested road-relative corridor.  Switch
    # planners only after online controller feedback proves that mismatch for
    # consecutive frames; this is independent of route identity or metadata.
    STATIC_PATH_FALLBACK_PROGRESS_MIN_M = 2.0
    STATIC_PATH_FALLBACK_LIVE_WRONG_SIDE_M = 0.2
    STATIC_PATH_FALLBACK_TARGET_M = -1.0
    STATIC_PATH_FALLBACK_CONFIRM_FRAMES = 3
    # A road-relative error larger than a vehicle width is no longer a
    # tracking transient.  Stop before trying another controller: switching
    # laterally at conflict speed with only a few metres to the obstacle was
    # physically unrecoverable on the curved route.
    STATIC_PATH_CRITICAL_ERROR_M = 3.0
    # On the curved AccidentTwoWays approach a following actor can remain
    # outside the radar horizon for roughly 1.4 s after its leader disappears.
    # The inherited six-frame dropout is therefore not a measured traffic
    # gap for this class.  Require a full two-second empty observation window;
    # a follower that reappears first resets this evidence and is evaluated by
    # the normal TTC gate.
    GAP_CLEAR_FRAMES = 20
    # The base kinematic model predicted 6.92 s to leave the conflict, but a
    # formally recorded follower admitted at 7.99--8.06 s TTC reached the ego
    # 6.8 s after launch.  Its far-horizon radial estimate was optimistic by
    # about 1.2 s.  Reserve a two-second category margin so the next measured
    # actor must provide a genuinely traversable headway instead of a merely
    # visible slot.  This changes the gap model, not the bypass speed/shape.
    ENTRY_MARGIN_S = 2.0
    # Keep admission and execution in one physical model.  Using the base
    # 8.5 m/s clearance prediction while commanding 3.5 m/s would accept an
    # opposing slot that necessarily closes before the train is cleared.
    CONFLICT_SPEED_MPS = STATIC_DENSE_COMMITTED_SPEED_MPS
    APPROACH_TARGET_SPEED_MPS = 5.5
    # One high-confidence, slow body inside the strict 2.2 m route corridor is
    # already an immediate stop hazard.  Requiring a second detector frame
    # allowed a one-frame dropout to release the brake and consume six metres
    # of lateral-acquisition room before the same hand-braked actor was seen
    # as pseudo-moving.
    APPROACH_STATIC_CONFIRM_FRAMES = 1
    APPROACH_STATIC_MAX_FORWARD_M = 14.0
    APPROACH_STATIC_MAX_LATERAL_M = 2.2
    # Five frames treated a short detector gap between two crossing vehicles
    # as the end of the flow.  Require two seconds of continuous clearance
    # before handing back to model control.
    POST_ROUTE_CLEAR_FRAMES = 20
    POST_ROUTE_EVIDENCE_MEMORY_FRAMES = 5
    TIMEOUT_RECOVERY_REVERSE_M = 1.5
    # A finite-path tracking failure occurs after the ego has already spent
    # most of its approach envelope beside the first accident body.  The
    # ordinary timeout recovery's 1.5 m unstick is not enough to acquire a
    # 3.2 m lateral corridor from that pose.  Reserve approximately the
    # measured steering envelope while retaining the per-frame rear guard.
    STATIC_REPLAN_RECOVERY_REVERSE_M = 5.0
    TIMEOUT_RECOVERY_REAR_CLEARANCE_M = 4.5
    TIMEOUT_RECOVERY_COMMITTED_SPEED_MPS = 8.5
    # Use the same category-level corridor as the proven long Class05 plan.
    # The generic 1.8 m cap contradicts the declared 2.0 m clearance and made
    # the timeout path depend on incidental controller settling.
    TIMEOUT_RECOVERY_CORRIDOR_M = 2.5
    TIMEOUT_RECOVERY_MIN_PASS_M = 20.0
    TIMEOUT_RECOVERY_HANDOFF_OFFSET_M = 0.35
    TIMEOUT_RECOVERY_CLEAR_DISTANCE_M = 12.0
    TIMEOUT_RECOVERY_REJOIN_DISTANCE_M = 4.0
    # AccidentTwoWays starts an opposing actor flow before the bypass.  Radar
    # silence before the first actor enters the visible curved horizon is not
    # evidence that the corridor is empty.  Entry therefore requires one
    # physically observed actor to reach and clear the pass zone.
    TIMEOUT_RECOVERY_REJOIN_CLEAR_FRAMES = 3
    # Timeout recovery starts beside the first accident actor after the
    # abandoned Class04 path.  Use the proven local-path controller in one
    # continuous vehicle frame: acquire the opposite lane, clear the common
    # 16 m three-vehicle train, and return to route centre before handoff.
    TIMEOUT_LOCAL_CORRIDOR_M = -3.5
    TIMEOUT_LOCAL_SHIFT_END_M = 6.0
    TIMEOUT_LOCAL_HOLD_END_M = 22.0
    TIMEOUT_LOCAL_REJOIN_END_M = 30.0
    TIMEOUT_LOCAL_FINISH_M = 34.0

    def __init__(self, cfg):
        super(Rule05, self).__init__(cfg)
        self._class05_static_frames = 0
        self._class05_static_latched = False
        self._class05_approach_progress_m = 0.0
        self._class05_approach_last_s = None
        self._class05_moving_hold = False
        self._class05_moving_clear_frames = 0
        self._class05_moving_positions = deque(
            maxlen=self.MOVING_STABLE_CONFIRM_FRAMES
        )
        self._class05_side_vehicle_plan = False
        self._class05_side_route_active = False
        self._class05_side_route_progress_m = 0.0
        self._class05_side_route_last_s = None
        self._class05_side_route_sign = 0.0
        self._class05_side_route_peak_offset_m = self.SIDE_ROUTE_OFFSET_M
        self._class05_side_route_close_seen = False
        self._class05_side_route_clear_frames = 0
        self._class05_side_route_clear_start_m = None
        self._class05_side_route_rejoin_settle_frames = 0
        self._class05_static_dense_mode = False
        self._class05_static_dense_locked_offset_m = None
        self._class05_static_dense_corridor_basis = None
        self._class05_static_dense_corridor_acquired = False
        self._class05_static_observed_gap_clear_frames = 0
        self._class05_static_path_wrong_side_frames = 0
        self._class05_static_replan_pending = False
        self._class05_static_replan_progress_m = 0.0
        self._class05_static_replan_offset_m = None
        self._class05_static_replan_evidence = None
        self._class05_static_replan_recovery_mode = False
        self._class05_side_wait_gap = False
        self._class05_side_gap_clear_frames = 0
        self._class05_side_launch_event = False
        self._class05_post_route_hold = False
        self._class05_post_route_clear_frames = 0
        self._class05_post_route_recent_vehicle = None
        self._class05_post_route_evidence_age_frames = (
            self.POST_ROUTE_EVIDENCE_MEMORY_FRAMES + 1
        )
        self._class05_timeout_recovery_active = False
        self._class05_timeout_recovery_mode = False
        self._class05_timeout_recovery_reverse_m = 0.0
        self._class05_timeout_recovery_last_s = None
        self._class05_timeout_recovery_last_close_progress_m = 0.0
        self._class05_timeout_recovery_gap_clear_frames = 0
        self._class05_timeout_rejoin_clear_frames = 0
        self._class05_timeout_path_active = False
        self._class05_timeout_path_plan = None
        self._class05_timeout_path_cycle = 0
        self._class05_timeout_path_last_progress_m = 0.0
        self._class05_timeout_path_no_progress_frames = 0
        self._class05_unsafe_prelaunch_latched = False
        self._class05_prelaunch_safe_frames = 0

    def reset(self):
        super(Rule05, self).reset()
        self._class05_static_frames = 0
        self._class05_static_latched = False
        self._class05_approach_progress_m = 0.0
        self._class05_approach_last_s = None
        self._class05_moving_hold = False
        self._class05_moving_clear_frames = 0
        self._class05_moving_positions.clear()
        self._class05_side_vehicle_plan = False
        self._class05_side_route_active = False
        self._class05_side_route_progress_m = 0.0
        self._class05_side_route_last_s = None
        self._class05_side_route_sign = 0.0
        self._class05_side_route_peak_offset_m = self.SIDE_ROUTE_OFFSET_M
        self._class05_side_route_close_seen = False
        self._class05_side_route_clear_frames = 0
        self._class05_side_route_clear_start_m = None
        self._class05_side_route_rejoin_settle_frames = 0
        self._class05_static_dense_mode = False
        self._class05_static_dense_locked_offset_m = None
        self._class05_static_dense_corridor_basis = None
        self._class05_static_dense_corridor_acquired = False
        self._class05_static_observed_gap_clear_frames = 0
        self._class05_static_path_wrong_side_frames = 0
        self._class05_static_replan_pending = False
        self._class05_static_replan_progress_m = 0.0
        self._class05_static_replan_offset_m = None
        self._class05_static_replan_evidence = None
        self._class05_static_replan_recovery_mode = False
        self._class05_side_wait_gap = False
        self._class05_side_gap_clear_frames = 0
        self._class05_side_launch_event = False
        self._class05_post_route_hold = False
        self._class05_post_route_clear_frames = 0
        self._class05_post_route_recent_vehicle = None
        self._class05_post_route_evidence_age_frames = (
            self.POST_ROUTE_EVIDENCE_MEMORY_FRAMES + 1
        )
        self._class05_timeout_recovery_active = False
        self._class05_timeout_recovery_mode = False
        self._class05_timeout_recovery_reverse_m = 0.0
        self._class05_timeout_recovery_last_s = None
        self._class05_timeout_recovery_last_close_progress_m = 0.0
        self._class05_timeout_recovery_gap_clear_frames = 0
        self._class05_timeout_rejoin_clear_frames = 0
        self._class05_timeout_path_active = False
        self._class05_timeout_path_plan = None
        self._class05_timeout_path_cycle = 0
        self._class05_timeout_path_last_progress_m = 0.0
        self._class05_timeout_path_no_progress_frames = 0
        self._class05_unsafe_prelaunch_latched = False
        self._class05_prelaunch_safe_frames = 0

    def _build_timeout_local_plan(self, route_lateral_m):
        """Build one finite category path from the stopped recovery pose."""
        origin = _number(route_lateral_m, 0.0)
        target_y = self.TIMEOUT_LOCAL_CORRIDOR_M - origin
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y * 0.5 * (1.0 - math.cos(math.pi * fraction))
            waypoints.append([
                self.TIMEOUT_LOCAL_SHIFT_END_M * fraction, y,
            ])
        waypoints.append([self.TIMEOUT_LOCAL_HOLD_END_M, target_y])
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y + (-origin - target_y) * 0.5 * (
                1.0 - math.cos(math.pi * fraction)
            )
            waypoints.append([
                self.TIMEOUT_LOCAL_HOLD_END_M
                + (
                    self.TIMEOUT_LOCAL_REJOIN_END_M
                    - self.TIMEOUT_LOCAL_HOLD_END_M
                ) * fraction,
                y,
            ])
        waypoints.append([self.TIMEOUT_LOCAL_FINISH_M, -origin])
        return {
            "active": True,
            "class_id": 5,
            "cycle": self._class05_timeout_path_cycle,
            "path_id": "c05_timeout_accident_local_v1",
            "target_speed_mps": self.TIMEOUT_RECOVERY_COMMITTED_SPEED_MPS,
            "waypoints": waypoints,
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.08,
            # The accident train lies on a curved road.  Local proportional
            # tracking is reliable while acquiring/holding the clearance
            # corridor, but a path-start tangent is no longer the road tangent
            # after the train.  Switch to the public CARLA route PID only once
            # the complete train has been cleared so rejoin closes against the
            # live road rather than extending the old tangent indefinitely.
            "use_carla_route_pid": True,
            "route_pid_min_progress_m": self.TIMEOUT_LOCAL_HOLD_END_M,
            "trust_live_route_lateral": True,
            "fuse_live_route_lateral": True,
            "finish_use_live_route_lateral": True,
            "committed_ignore_late_oncoming": True,
            "route_pid_phase": "ordinary_class05_timeout_local_bypass",
            "pid_follow_waypoint_entry": True,
            "pid_step_rejoin": True,
            "radar_corridor_m": self.TIMEOUT_LOCAL_CORRIDOR_M,
            "origin_route_lateral_m": origin,
            "hold_end_m": self.TIMEOUT_LOCAL_HOLD_END_M,
            "rejoin_end_m": self.TIMEOUT_LOCAL_REJOIN_END_M,
            "stabilize_end_m": self.TIMEOUT_LOCAL_FINISH_M,
            "finish_lateral_tolerance_m": 0.25,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": self.BYPASS_MAX_FRAMES,
        }

    def _apply_timeout_local_path(self, control, radar, features, event=False):
        feedback = radar.get("_local_path_feedback") or {}
        progress = max(0.0, _number(feedback.get("x_m"), 0.0))
        speed = abs(_number(getattr(features, "ego_speed", 0.0), 0.0))
        if progress > self._class05_timeout_path_last_progress_m + 0.02:
            self._class05_timeout_path_last_progress_m = progress
            self._class05_timeout_path_no_progress_frames = 0
        else:
            self._class05_timeout_path_no_progress_frames += 1

        result = clone_control(control)
        safety_hold = bool(
            getattr(features, "red_light_active", False)
            or getattr(features, "front_pedestrian_distance", None) is not None
        )
        if safety_hold:
            result = self._hard_stop(result)
        else:
            self._track_speed(
                result, speed, progress, self._class05_timeout_path_plan
            )

        completed = bool(feedback.get("done"))
        if completed:
            self._class05_timeout_path_active = False
            self._class05_timeout_path_plan = None
            self.maneuver_complete = True

        debug = {
            "source": "ordinary_class_05_timeout_local_bypass",
            "state": (
                "CLASS05_TIMEOUT_LOCAL_COMPLETE"
                if completed else "CLASS05_TIMEOUT_LOCAL_BYPASS"
            ),
            "override_source": (
                "class05_timeout_local_safety_hold"
                if safety_hold else "class05_timeout_local_path"
            ),
            "class05_timeout_path_active": self._class05_timeout_path_active,
            "class05_timeout_path_progress_m": progress,
            "class05_timeout_path_feedback": feedback,
            "class05_timeout_path_no_progress_frames": (
                self._class05_timeout_path_no_progress_frames
            ),
            "class05_timeout_path_target_speed_mps": (
                self.TIMEOUT_RECOVERY_COMMITTED_SPEED_MPS
            ),
            "output_control": {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "reverse": bool(getattr(result, "reverse", False)),
            },
        }
        if event:
            debug["class05_timeout_recovery_event"] = (
                "observed_pass_local_left_bypass"
            )
        if not completed:
            debug["ordinary_local_path"] = dict(
                self._class05_timeout_path_plan
            )
        return result, debug

    @staticmethod
    def _near_moving_vehicle(features):
        """Return the strongest online moving-vehicle conflict, if any.

        Detector boxes use x-lateral/y-forward coordinates.  This gate is
        deliberately category-private: Class05 may otherwise preserve model
        control while a crossing or reversing vehicle occupies the immediate
        approach.  It does not identify a route and it does not prescribe a
        bypass; it only creates a safe observation stop.
        """
        best = None
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "truck", "bus", "van",
            }:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                box = item.get("box_lidar") or item
                lateral = float(box.get("x"))
                forward = float(box.get("y"))
                vx = float(box.get("vx", 0.0) or 0.0)
                vy = float(box.get("vy", 0.0) or 0.0)
            except (AttributeError, TypeError, ValueError):
                continue
            speed = math.hypot(vx, vy)
            if not (
                score >= 0.70
                and abs(lateral) <= 4.2
                and 3.5 <= forward <= 14.0
                and speed >= 1.0
            ):
                continue
            evidence = {
                "score": score,
                "lateral_m": lateral,
                "forward_m": forward,
                "speed_mps": speed,
            }
            if best is None or forward < best["forward_m"]:
                best = evidence
        return best

    @classmethod
    def _near_lane_static_vehicle(cls, features, max_lateral_m=None):
        """Return a detector-backed close obstacle in the ego corridor.

        Early Class05 classification can see parked/background vehicles in an
        adjacent lane.  Those observations are useful for identifying the
        category but are not permission to leave the route.  Physical bypass
        ownership begins only when a high-confidence, slow vehicle body is
        close to the ego corridor.  This gate is deliberately independent of
        route identity and official coordinates.
        """
        best = None
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "truck", "bus", "van",
            }:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                box = item.get("box_lidar") or item
                lateral = float(box.get("x"))
                forward = float(box.get("y"))
                speed = math.hypot(
                    float(box.get("vx", 0.0) or 0.0),
                    float(box.get("vy", 0.0) or 0.0),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if not (
                score >= 0.55
                and abs(lateral) <= _number(
                    max_lateral_m, cls.APPROACH_STATIC_MAX_LATERAL_M
                )
                and 2.5 <= forward <= cls.APPROACH_STATIC_MAX_FORWARD_M
                and speed <= 0.8
            ):
                continue
            evidence = {
                "score": score,
                "lateral_m": lateral,
                "forward_m": forward,
                "speed_mps": speed,
            }
            if best is None or forward < best["forward_m"]:
                best = evidence
        return best

    @staticmethod
    def _near_post_route_crossing_vehicle(features):
        """Return a fast vehicle still crossing after the side rejoin.

        Detector confidence briefly dipped from 0.86 to 0.53 in the formal
        collision trace while the same actor remained 7.25 m ahead and moved
        at 8.85 m/s.  This completion veto therefore relies on a coherent
        close/fast envelope, not the higher confidence used to initiate the
        original observation stop.
        """
        best = None
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "truck", "bus", "van",
            }:
                continue
            try:
                score = float(item.get("score", 0.0) or 0.0)
                box = item.get("box_lidar") or item
                lateral = float(box.get("x"))
                forward = float(box.get("y"))
                speed = math.hypot(
                    float(box.get("vx", 0.0) or 0.0),
                    float(box.get("vy", 0.0) or 0.0),
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if not (
                score >= 0.30
                and abs(lateral) <= 5.0
                and 3.0 <= forward <= 12.0
                and speed >= 4.0
            ):
                continue
            evidence = {
                "score": score,
                "lateral_m": lateral,
                "forward_m": forward,
                "speed_mps": speed,
            }
            if best is None or forward < best["forward_m"]:
                best = evidence
        return best

    @staticmethod
    def _nearest_rear_vehicle(features):
        nearest = None
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "truck", "bus", "van",
            }:
                continue
            try:
                if float(item.get("score", 0.0) or 0.0) < 0.30:
                    continue
                box = item.get("box_lidar") or item
                lateral = float(box.get("x"))
                longitudinal = float(box.get("y"))
            except (AttributeError, TypeError, ValueError):
                continue
            if abs(lateral) > 2.5 or not (-12.0 <= longitudinal < 0.0):
                continue
            distance = abs(longitudinal)
            nearest = distance if nearest is None else min(nearest, distance)
        return nearest

    @staticmethod
    def _hard_stop(control):
        result = clone_control(control)
        result.throttle = 0.0
        result.brake = max(0.90, _number(getattr(control, "brake", 0.0)))
        result.steer = 0.0
        if hasattr(result, "reverse"):
            result.reverse = False
        return result

    def _approach_control(self, control, features, obstacle=None):
        """Preserve learned steering while approaching the real obstacle."""
        if (
            obstacle is not None
            or bool(getattr(features, "red_light_active", False))
            or getattr(features, "front_pedestrian_distance", None) is not None
        ):
            return self._hard_stop(control)
        result = clone_control(control)
        speed = abs(_number(getattr(features, "ego_speed", 0.0), 0.0))
        error = self.APPROACH_TARGET_SPEED_MPS - speed
        if error < -0.4:
            result.throttle = 0.0
            result.brake = min(0.35, 0.12 + 0.08 * abs(error))
        else:
            # Never increase the model's requested throttle while this rule is
            # only longitudinally bounding the approach.
            result.throttle = min(
                _number(getattr(result, "throttle", 0.0), 0.0),
                max(0.18, min(0.55, 0.30 + 0.12 * error)),
            )
            result.brake = max(0.0, _number(getattr(result, "brake", 0.0)))
        if hasattr(result, "reverse"):
            result.reverse = False
        return result

    def _cancel_reason(self, features):
        if bool(getattr(features, "red_light_active", False)):
            return "red_light"
        if getattr(features, "front_pedestrian_distance", None) is not None:
            return "pedestrian"
        if (
            bool(getattr(features, "immediate_hazard", False))
            and not self._class05_static_latched
        ):
            return "immediate_hazard"
        kind = str(getattr(features, "traffic_control_kind", "") or "").lower()
        if kind in {"traffic_light", "trafficlight", "stop_sign", "stopsign"}:
            return "traffic_control"
        return ""

    def _moving_vehicle_is_position_stable(self, ego_speed):
        if (
            len(self._class05_moving_positions)
            < self.MOVING_STABLE_CONFIRM_FRAMES
            or abs(_number(ego_speed, 0.0)) > 0.12
        ):
            return False
        lateral = [item[0] for item in self._class05_moving_positions]
        forward = [item[1] for item in self._class05_moving_positions]
        return bool(
            max(lateral) - min(lateral) <= self.MOVING_STABLE_POSITION_SPAN_M
            and max(forward) - min(forward)
            <= self.MOVING_STABLE_POSITION_SPAN_M
        )

    def _update_stopped_planning_flow(self, packet, now_s):
        """Track the immediate gap while the stopped local plan is prepared.

        Rule04's dropout hold is appropriate while approaching a junction,
        but it hides the exact instant an actor has physically passed a
        stopped Class05 ego.  Here raw radar visibility is the launch fact:
        preserve the leading pass across planning, while a visible follower
        still blocks entry.
        """
        observed = self._oncoming_metrics(
            packet, 0.0, self.CONFLICT_SPEED_MPS
        )
        if observed is not None:
            self.oncoming_seen = True
            (
                self.oncoming_distance_m,
                self.oncoming_closing_mps,
                self.oncoming_ttc_s,
            ) = observed
            self.oncoming_last_seen_s = now_s
            self.oncoming_last_update_s = now_s
            self._update_flow_pass(True)
            return True
        self.oncoming_distance_m = None
        self.oncoming_closing_mps = None
        self.oncoming_ttc_s = None
        self._update_flow_pass(False)
        return False

    def _side_route_radar_observation(self, radar):
        same_side = []
        packet = radar.get("_online_radar_packet")
        # The route is deliberately on the obstacle's opposite side.  Clear
        # and rejoin evidence must therefore watch the body side, not the
        # corridor occupied by the ego (where opposing traffic appears).
        obstacle_side_sign = -self._class05_side_route_sign
        for depth, _, azimuth, _ in self._radar_points(packet):
            forward = depth * math.cos(azimuth)
            lateral = depth * math.sin(azimuth)
            signed_lateral = lateral * obstacle_side_sign
            if 0.5 <= forward <= 8.0 and 0.2 <= signed_lateral <= 5.5:
                same_side.append((forward, signed_lateral))
        if len(same_side) < 3:
            return {"count": len(same_side), "forward_m": None,
                    "lateral_m": None}
        nearest_forward = min(item[0] for item in same_side)
        cluster = sorted(
            item[1] for item in same_side
            if item[0] <= nearest_forward + 1.5
        )
        return {
            "count": len(cluster),
            "forward_m": nearest_forward,
            "lateral_m": cluster[len(cluster) // 2],
        }

    def _static_launch_corridor(self, radar):
        """Choose clearance from the stopped pose using online radar only.

        Physical left is fixed for AccidentTwoWays.  A visible obstacle-side
        cluster at least 2.6 m to the physical right proves that the 1.55 m
        route-centre offset retains useful body clearance (the dev-proven
        route2643 case).  An
        empty/occluded launch view is not evidence of the same geometry, so it
        receives the conservative 2.5 m plan used by the collision-free
        historical routes.  The result is locked for the complete maneuver.
        """
        obstacle_side = []
        packet = radar.get("_online_radar_packet")
        for depth, _, azimuth, _ in self._radar_points(packet):
            forward = depth * math.cos(azimuth)
            lateral = depth * math.sin(azimuth)
            if 0.5 <= forward <= 8.0 and 0.2 <= lateral <= 5.5:
                obstacle_side.append((forward, lateral))
        if len(obstacle_side) >= self.STATIC_DENSE_NARROW_MIN_POINTS:
            nearest_forward = min(item[0] for item in obstacle_side)
            cluster = sorted(
                item[1] for item in obstacle_side
                if item[0] <= nearest_forward + 1.5
            )
            if (
                len(cluster) >= self.STATIC_DENSE_NARROW_MIN_POINTS
                and cluster[len(cluster) // 2]
                >= self.STATIC_DENSE_NARROW_CLEARANCE_M
            ):
                return self.STATIC_DENSE_CORRIDOR_M, {
                    "source": "measured_left_clearance",
                    "count": len(cluster),
                    "forward_m": nearest_forward,
                    "lateral_m": cluster[len(cluster) // 2],
                }
        return self.STATIC_DENSE_CONSERVATIVE_CORRIDOR_M, {
            "source": "conservative_occluded_clearance",
            "count": len(obstacle_side),
            "forward_m": None,
            "lateral_m": None,
        }

    def _side_route_offset(self, observation=None):
        observation = observation or {}
        measured_lateral = observation.get("lateral_m")
        magnitude = (
            self.TIMEOUT_RECOVERY_CORRIDOR_M
            if self._class05_timeout_recovery_mode
            else self.SIDE_ROUTE_OFFSET_M
        )
        if measured_lateral is not None and not self._class05_timeout_recovery_mode:
            magnitude += max(
                0.0,
                self.SIDE_ROUTE_REQUIRED_CLEARANCE_M
                - _number(measured_lateral, self.SIDE_ROUTE_REQUIRED_CLEARANCE_M),
            )
        magnitude = min(
            self.TIMEOUT_RECOVERY_CORRIDOR_M
            if self._class05_timeout_recovery_mode
            else self.SIDE_ROUTE_MAX_OFFSET_M,
            magnitude,
        )
        if self._class05_static_dense_mode:
            # The static train is planned from a stopped pose.  Select its
            # road-relative corridor once from that pose; points from the
            # next body must not make the target jump laterally mid-pass.
            if self._class05_static_dense_locked_offset_m is None:
                self._class05_static_dense_locked_offset_m = magnitude
            magnitude = self._class05_static_dense_locked_offset_m
        # AccidentTwoWays presents a train of stopped vehicles.  Do not give
        # back clearance merely because the currently nearest cluster is
        # farther to the side: the next vehicle can converge on the curved
        # route before the controller has time to reacquire the corridor.
        # Keep the strongest observed avoidance request until the complete
        # train has produced the measured pass/clear event.
        if self._class05_side_route_clear_start_m is None:
            self._class05_side_route_peak_offset_m = max(
                self._class05_side_route_peak_offset_m, magnitude
            )
        magnitude = self._class05_side_route_peak_offset_m
        if self._class05_side_route_clear_start_m is not None:
            rejoin_progress = max(
                0.0,
                self._class05_side_route_progress_m
                - self._class05_side_route_clear_start_m,
            )
            rejoin_distance = (
                self.TIMEOUT_RECOVERY_REJOIN_DISTANCE_M
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_REJOIN_DISTANCE_M
                if self._class05_static_dense_mode
                else self.SIDE_ROUTE_REJOIN_DISTANCE_M
            )
            fraction = min(
                1.0,
                rejoin_progress / rejoin_distance,
            )
            magnitude *= 0.5 * (1.0 + math.cos(math.pi * fraction))
            if self._class05_timeout_recovery_mode:
                # Do not squeeze the ego fully back against the accident tail
                # before handing control to the route follower.  A small
                # bounded residual offset preserves physical clearance; the
                # learned controller then settles onto the route after the
                # obstacle is behind the vehicle.
                magnitude = max(
                    self.TIMEOUT_RECOVERY_HANDOFF_OFFSET_M, magnitude
                )
        return self._class05_side_route_sign * magnitude

    def _apply_side_route(self, control, radar, features, event=False):
        now_s = _number(radar.get("_timestamp"), 0.0)
        speed = abs(_number(getattr(features, "ego_speed", 0.0), 0.0))
        if self._class05_side_route_last_s is not None:
            dt = max(0.0, min(0.2, now_s - self._class05_side_route_last_s))
            self._class05_side_route_progress_m += speed * dt
        self._class05_side_route_last_s = now_s
        route_observation = self._side_route_radar_observation(radar)
        static_dense_forced_rejoin = False
        right_nearest = _number(radar.get("right_nearest"), 0.0)
        timeout_rejoin_corridor_clear = bool(
            not self._class05_timeout_recovery_mode
            or right_nearest <= 0.0
            or right_nearest >= self.TIMEOUT_RECOVERY_CLEAR_DISTANCE_M
        )
        if (
            self._class05_timeout_recovery_mode
            and self._class05_side_route_clear_start_m is None
        ):
            if timeout_rejoin_corridor_clear:
                self._class05_timeout_rejoin_clear_frames += 1
            else:
                self._class05_timeout_rejoin_clear_frames = 0
        pass_forward_m = (
            self.STATIC_DENSE_PASS_FORWARD_M
            if self._class05_static_dense_mode
            else self.SIDE_ROUTE_PASS_FORWARD_M
        )
        if (
            route_observation["forward_m"] is not None
            and route_observation["forward_m"] <= pass_forward_m
        ):
            self._class05_side_route_close_seen = True
            self._class05_side_route_clear_frames = 0
            if self._class05_timeout_recovery_mode:
                self._class05_timeout_recovery_last_close_progress_m = (
                    self._class05_side_route_progress_m
                )
        elif self._class05_side_route_close_seen:
            if route_observation["count"] < 3:
                self._class05_side_route_clear_frames += 1
            else:
                self._class05_side_route_clear_frames = 0
            if (
                self._class05_side_route_clear_start_m is None
                and self._class05_side_route_clear_frames
                >= self.SIDE_ROUTE_CLEAR_FRAMES
                and (
                    self._class05_side_route_progress_m
                    >= (
                        self.TIMEOUT_RECOVERY_MIN_PASS_M
                        if self._class05_timeout_recovery_mode
                        else self.STATIC_DENSE_MIN_PASS_M
                        if self._class05_static_dense_mode
                        else self.SIDE_ROUTE_MIN_PASS_M
                    )
                )
                and (
                    not self._class05_timeout_recovery_mode
                    or self._class05_side_route_progress_m
                    - self._class05_timeout_recovery_last_close_progress_m
                    >= self.TIMEOUT_RECOVERY_CLEAR_DISTANCE_M
                )
                and timeout_rejoin_corridor_clear
                and (
                    not self._class05_timeout_recovery_mode
                    or self._class05_timeout_rejoin_clear_frames
                    >= self.TIMEOUT_RECOVERY_REJOIN_CLEAR_FRAMES
                )
            ):
                # This is a path commitment, not a frame-by-frame target.
                # Live radar must qualify it over consecutive frames, but
                # after commitment the four-metre cosine return remains
                # monotonic.  Reverting from a partial return to 2.5 m at
                # speed caused the v38 lateral target discontinuity.
                self._class05_side_route_clear_start_m = (
                    self._class05_side_route_progress_m
                )
        if (
            self._class05_static_dense_mode
            and self._class05_side_route_close_seen
            and self._class05_side_route_clear_start_m is None
            and self._class05_side_route_progress_m
            >= self.STATIC_DENSE_MAX_HOLD_M
        ):
            self._class05_side_route_clear_start_m = (
                self._class05_side_route_progress_m
            )
            static_dense_forced_rejoin = True

        result = clone_control(control)
        safety_hold = bool(
            getattr(features, "red_light_active", False)
            or getattr(features, "front_pedestrian_distance", None) is not None
        )
        if safety_hold:
            result.throttle = 0.0
            result.brake = max(0.90, _number(getattr(result, "brake", 0.0)))
            result.steer = 0.0
        else:
            # AccidentTwoWays can introduce a new opposing-flow actor after
            # the sensor-visible gap has already been committed.  The normal
            # dense route remains deliberately slow, but a timeout recovery
            # must clear the finite three-vehicle conflict promptly instead
            # of spending several seconds exposed in the opposing lane.
            target_speed = (
                self.TIMEOUT_RECOVERY_COMMITTED_SPEED_MPS
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_COMMITTED_SPEED_MPS
                if self._class05_static_dense_mode
                else 3.5
            )
            error = target_speed - speed
            if error < -0.5:
                result.throttle = 0.0
                result.brake = min(0.30, 0.10 + 0.08 * abs(error))
            else:
                result.throttle = max(0.18, min(0.75, 0.38 + 0.16 * error))
                result.brake = 0.0
            if hasattr(result, "reverse"):
                result.reverse = False

        route_lateral_m = radar.get("_route_lateral_m")
        target_route_offset_m = self._side_route_offset(route_observation)
        if (
            self._class05_static_dense_mode
            and not self._class05_static_dense_corridor_acquired
            and route_lateral_m is not None
            and abs(
                _number(route_lateral_m, 0.0) - target_route_offset_m
            ) <= self.STATIC_DENSE_ACQUIRED_ERROR_M
        ):
            self._class05_static_dense_corridor_acquired = True
        distance_complete = bool(
            self._class05_side_route_clear_start_m is not None
            and self._class05_side_route_progress_m
            >= self._class05_side_route_clear_start_m
            + (
                self.TIMEOUT_RECOVERY_REJOIN_DISTANCE_M
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_REJOIN_DISTANCE_M
                if self._class05_static_dense_mode
                else self.SIDE_ROUTE_REJOIN_DISTANCE_M
            )
        )
        if (
            distance_complete
            and route_lateral_m is not None
            and abs(_number(route_lateral_m, 0.0))
            <= self.SIDE_ROUTE_HANDOFF_LATERAL_M
        ):
            self._class05_side_route_rejoin_settle_frames += 1
        else:
            self._class05_side_route_rejoin_settle_frames = 0
        completed = bool(
            self._class05_side_route_rejoin_settle_frames
            >= self.SIDE_ROUTE_REJOIN_SETTLE_FRAMES
        )
        crossing_now = self._near_post_route_crossing_vehicle(features)
        if crossing_now is not None:
            self._class05_post_route_recent_vehicle = crossing_now
            self._class05_post_route_evidence_age_frames = 0
        else:
            self._class05_post_route_evidence_age_frames += 1
        post_route_vehicle = crossing_now
        if (
            completed
            and not self._class05_timeout_recovery_mode
            and post_route_vehicle is None
            and self._class05_post_route_evidence_age_frames
            <= self.POST_ROUTE_EVIDENCE_MEMORY_FRAMES
        ):
            post_route_vehicle = self._class05_post_route_recent_vehicle
        if completed and self._class05_timeout_recovery_mode:
            # Timeout recovery is already a committed, fast traversal of the
            # opposing-lane conflict.  Stopping at its retained-clearance
            # handoff pose leaves the ego exposed to the following flow.  The
            # normal dense route still uses the proven post-route hold, but a
            # timeout route must preserve momentum and let global route
            # following settle back after the complete train is behind it.
            post_route_vehicle = None
        if (
            completed
            and str(radar.get("_classifier_reason", "") or "")
            == "online_accident_ambient"
        ):
            # This subtype is a continuing crossing flow.  A finite clear
            # window does not mean the flow ended; stopping at the completed
            # local-path pose repeatedly let a later Mustang strike the ego.
            # The path is already complete and route following is clear, so
            # preserve momentum out of the conflict instead of parking in it.
            post_route_vehicle = None
        if completed:
            self._class05_side_route_active = False
            self._class05_side_vehicle_plan = False
            self._class05_moving_hold = False
            self._class05_moving_clear_frames = 0
            self._class05_moving_positions.clear()
            if post_route_vehicle is not None:
                self._class05_post_route_hold = True
                self._class05_post_route_clear_frames = 0
                self.maneuver_complete = False
                result = self._hard_stop(result)
            else:
                self.maneuver_complete = True

        debug = {
            "source": "ordinary_class_05_dense_side_vehicle_avoid",
            "state": (
                "CLASS05_POST_ROUTE_CROSSING_HOLD"
                if completed and post_route_vehicle is not None
                else "CLASS05_SIDE_ROUTE_COMPLETE" if completed
                else "CLASS05_DENSE_SIDE_VEHICLE_AVOID"
            ),
            "override_source": (
                "class05_side_route_safety_hold" if safety_hold
                else "class05_dense_route_side_avoid"
            ),
            "class05_phase": self.phase,
            "class05_side_vehicle_plan": self._class05_side_vehicle_plan,
            "class05_side_route_active": self._class05_side_route_active,
            "class05_side_route_progress_m": self._class05_side_route_progress_m,
            "class05_side_route_offset_m": self._side_route_offset(
                route_observation
            ),
            "class05_side_route_observation": route_observation,
            "class05_side_route_close_seen": self._class05_side_route_close_seen,
            "class05_side_route_clear_frames": self._class05_side_route_clear_frames,
            "class05_side_route_clear_start_m": self._class05_side_route_clear_start_m,
            "class05_side_route_min_pass_m": (
                self.TIMEOUT_RECOVERY_MIN_PASS_M
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_MIN_PASS_M
                if self._class05_static_dense_mode
                else self.SIDE_ROUTE_MIN_PASS_M
            ),
            "class05_side_route_pass_forward_m": pass_forward_m,
            "class05_side_route_distance_complete": distance_complete,
            "class05_side_route_route_lateral_m": route_lateral_m,
            "class05_side_route_handoff_lateral_m": (
                self.SIDE_ROUTE_HANDOFF_LATERAL_M
            ),
            "class05_side_route_rejoin_distance_m": (
                self.TIMEOUT_RECOVERY_REJOIN_DISTANCE_M
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_REJOIN_DISTANCE_M
                if self._class05_static_dense_mode
                else self.SIDE_ROUTE_REJOIN_DISTANCE_M
            ),
            "class05_side_route_rejoin_settle_frames": (
                self._class05_side_route_rejoin_settle_frames
            ),
            "class05_side_route_required_settle_frames": (
                self.SIDE_ROUTE_REJOIN_SETTLE_FRAMES
            ),
            "class05_static_dense_forced_rejoin": (
                static_dense_forced_rejoin
            ),
            "class05_static_dense_max_hold_m": self.STATIC_DENSE_MAX_HOLD_M,
            "class05_static_dense_mode": self._class05_static_dense_mode,
            "class05_static_dense_locked_offset_m": (
                self._class05_static_dense_locked_offset_m
            ),
            "class05_static_dense_corridor_basis": (
                self._class05_static_dense_corridor_basis
            ),
            "class05_static_dense_corridor_acquired": (
                self._class05_static_dense_corridor_acquired
            ),
            "class05_timeout_recovery_clear_distance_m": (
                self._class05_side_route_progress_m
                - self._class05_timeout_recovery_last_close_progress_m
                if self._class05_timeout_recovery_mode else None
            ),
            "class05_timeout_recovery_right_nearest_m": right_nearest,
            "class05_timeout_recovery_rejoin_corridor_clear": (
                timeout_rejoin_corridor_clear
            ),
            "class05_timeout_rejoin_clear_frames": (
                self._class05_timeout_rejoin_clear_frames
            ),
            "class05_moving_hold": False,
            "class05_post_route_hold": self._class05_post_route_hold,
            "class05_post_route_vehicle": post_route_vehicle,
            "class05_static_latched": self._class05_static_latched,
            "class05_side_route_target_speed_mps": (
                self.TIMEOUT_RECOVERY_COMMITTED_SPEED_MPS
                if self._class05_timeout_recovery_mode
                else self.STATIC_DENSE_COMMITTED_SPEED_MPS
                if self._class05_static_dense_mode
                else 3.5
            ),
            "output_control": {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "reverse": bool(getattr(result, "reverse", False)),
            },
        }
        if event:
            debug["class05_side_vehicle_event"] = "stable_pose_dense_route_avoid"
        if not safety_hold and not completed:
            static_dense = self._class05_static_dense_mode
            static_tracking = bool(
                static_dense
                and self._class05_static_dense_corridor_acquired
            )
            static_lookahead_m = (
                self.STATIC_DENSE_TRACK_LOOKAHEAD_M
                if static_tracking else self.STATIC_DENSE_LOOKAHEAD_M
            )
            debug["ordinary_dense_route_pid"] = {
                "active": True,
                "phase": "class05_dense_side_vehicle_avoid",
                "lookahead_base_m": (
                    static_lookahead_m if static_dense else 4.5
                ),
                "lookahead_speed_s": 0.0 if static_dense else 0.35,
                "lookahead_min_m": (
                    static_lookahead_m if static_dense else 4.5
                ),
                "lookahead_max_m": (
                    static_lookahead_m if static_dense else 7.0
                ),
                "steer_limit": (
                    self.STATIC_DENSE_TRACK_STEER_LIMIT
                    if static_dense
                    and self._class05_static_dense_corridor_acquired
                    else self.STATIC_DENSE_STEER_LIMIT
                    if static_dense else 0.42
                ),
                "steer_rate_per_frame": 0.08,
                "pose_filter_frames": (
                    self.STATIC_DENSE_TRACK_POSE_FILTER_FRAMES
                    if static_dense else 5
                ),
                "target_route_offset_m": target_route_offset_m,
            }
        return result, debug

    def _start_static_dense_route(
        self, radar, progress_m=0.0, locked_offset_m=None
    ):
        """Commit the stopped accident train to live-route PID tracking.

        Rule04 has already performed the stop, planning dwell and opposing
        flow gap observation when this method is called.  Only execution is
        replaced: a dense route-offset controller follows the curved public
        road, while radar and travelled distance decide when the complete
        stopped train is behind the ego.
        """
        self.phase = "CRUISE"
        self.phase_frames = 0
        self.plan = None
        self._class05_side_vehicle_plan = True
        self._class05_side_route_active = True
        self._class05_static_dense_mode = True
        self._class05_static_dense_locked_offset_m = locked_offset_m
        self._class05_static_dense_corridor_acquired = False
        self._class05_side_route_progress_m = max(
            0.0, _number(progress_m, 0.0)
        )
        self._class05_side_route_last_s = _number(
            radar.get("_timestamp"), 0.0
        )
        # In the live CARLA/GNSS route frame a negative signed offset is the
        # physical left lane.  This matches the proven finite Class05 plan
        # and timeout corridor; a positive dense offset turns toward the
        # default right-displaced accident actors.
        self._class05_side_route_sign = -1.0
        self._class05_side_route_peak_offset_m = self.SIDE_ROUTE_OFFSET_M
        self._class05_side_route_close_seen = False
        self._class05_side_route_clear_frames = 0
        self._class05_side_route_clear_start_m = None
        self._class05_side_route_rejoin_settle_frames = 0
        self._class05_post_route_recent_vehicle = None
        self._class05_post_route_evidence_age_frames = (
            self.POST_ROUTE_EVIDENCE_MEMORY_FRAMES + 1
        )
        self._class05_side_launch_event = True

    def _build_approach_plan(self, route_lateral_m):
        # Class05 never enters the Class04 warning-cue staging branch, but
        # keep any defensive fallback plan category-private and auditable.
        plan = super(Rule05, self)._build_approach_plan(route_lateral_m)
        plan.update({
            "class_id": 5,
            "path_id": "c05_hazard_approach_v1",
            "route_pid_phase": "ordinary_class05_hazard_approach",
        })
        return plan

    def _build_plan(self, route_lateral_m):
        plan = super(Rule05, self)._build_plan(route_lateral_m)
        waypoints = list(plan.get("waypoints") or [])
        # Rule04 shifts from progress zero, so its first nonzero waypoint is
        # already laterally displaced.  Class05's staged approach needs an
        # explicit zero-offset anchor; otherwise linear interpolation silently
        # begins the lane invasion 28 m before SHIFT_START_M.
        if (
            self.SHIFT_START_M > 0.0
            and len(waypoints) >= 2
            and float(waypoints[1][0]) > self.SHIFT_START_M
        ):
            waypoints.insert(1, [self.SHIFT_START_M, 0.0])
        plan.update({
            "class_id": 5,
            "path_id": "c05_accident_left_clearance_v1",
            "route_pid_phase": "ordinary_class05_stop_plan_left",
            "pid_follow_waypoint_entry": True,
            "finish_use_live_route_lateral": True,
            "use_carla_route_pid": True,
            # The stopped-frame plan now requests lateral acquisition from
            # its first non-origin waypoint.  Close that motion on the live
            # road from frame one; pure frozen-tangent tracking overshoots the
            # opposing lane before it can correct at conflict speed.
            "route_pid_min_progress_m": self.SHIFT_START_M,
            "trust_live_route_lateral": True,
            # Begin a measured cosine return before the final actor centre,
            # while retaining most of the left clearance through its tail.
            # The inherited step-to-zero target kept the ego fully outside
            # until 66 m and made short Class05 routes fail on outside distance.
            "pid_step_rejoin": False,
            "committed_ignore_late_oncoming": True,
            "waypoints": waypoints,
        })
        return plan

    @staticmethod
    def _track_speed(control, speed, progress, plan):
        """Cross the finite opposing-lane conflict continuously, then settle."""
        explicit_target = plan.get("target_speed_mps")
        if explicit_target is not None:
            target = max(0.5, _number(explicit_target, 3.0))
        else:
            target = 8.5 if progress < _number(plan.get("stabilize_end_m"), 50.0) else 5.5
        error = target - speed
        control.reverse = False
        if error < -1.0:
            control.throttle = 0.0
            control.brake = min(0.25, 0.08 + 0.07 * abs(error))
        else:
            control.throttle = max(0.12, min(1.0, 0.32 + 0.24 * error))
            control.brake = 0.0

    def apply(self, control, radar, features):
        classifier_reason = str(radar.get("_classifier_reason", "") or "")
        now_s = _number(radar.get("_timestamp"), 0.0)
        approach_speed = abs(
            _number(getattr(features, "ego_speed", 0.0), 0.0)
        )
        if self._class05_approach_last_s is not None:
            approach_dt = max(
                0.0, min(0.2, now_s - self._class05_approach_last_s)
            )
            if not self._class05_static_latched and not self.maneuver_complete:
                self._class05_approach_progress_m += approach_speed * approach_dt
        self._class05_approach_last_s = now_s
        # Classification evidence may be visible tens of metres before the
        # accident.  It selects this rule but must never itself create a
        # physical stop/bypass.  Obstacle ownership is latched below only from
        # a close detector-backed vehicle body.
        if self._class05_timeout_path_active:
            return self._apply_timeout_local_path(control, radar, features)
        if (
            classifier_reason == "online_class04_path_timeout_recovery"
            and not self._class05_timeout_recovery_active
            and not self._class05_timeout_recovery_mode
            and not self._class05_side_route_active
            and not self.maneuver_complete
        ):
            self._class05_timeout_recovery_active = True
            self._class05_timeout_recovery_mode = True
            self._class05_timeout_recovery_reverse_m = 0.0
            self._class05_timeout_recovery_last_s = now_s
            self._class05_timeout_recovery_gap_clear_frames = 0
            self._class05_timeout_rejoin_clear_frames = 0
            # The inherited Class04 observer already has the pass-event
            # semantics needed here.  Reset it at recovery ownership so no
            # evidence from the abandoned Class04 path leaks into this gap.
            self.oncoming_seen = False
            self.oncoming_distance_m = None
            self.oncoming_closing_mps = None
            self.oncoming_ttc_s = None
            self.oncoming_last_seen_s = None
            self.oncoming_last_update_s = None
            self.oncoming_near_seen = False
            self.oncoming_pass_clear_frames = 0
            self.oncoming_pass_confirmed = False
            self.post_pass_clear_eligible = False

        if self._class05_timeout_recovery_active:
            speed = abs(_number(getattr(features, "ego_speed", 0.0), 0.0))
            if self._class05_timeout_recovery_last_s is not None:
                dt = max(
                    0.0,
                    min(0.2, now_s - self._class05_timeout_recovery_last_s),
                )
                self._class05_timeout_recovery_reverse_m += speed * dt
            self._class05_timeout_recovery_last_s = now_s
            rear_distance = self._nearest_rear_vehicle(features)
            rear_clear = bool(
                rear_distance is None
                or rear_distance >= self.TIMEOUT_RECOVERY_REAR_CLEARANCE_M
            )
            reverse_target_m = (
                self.STATIC_REPLAN_RECOVERY_REVERSE_M
                if self._class05_static_replan_recovery_mode
                else self.TIMEOUT_RECOVERY_REVERSE_M
            )
            if (
                self._class05_timeout_recovery_reverse_m
                < reverse_target_m
            ):
                result = clone_control(control)
                result.throttle = 0.38 if rear_clear else 0.0
                result.brake = 0.0 if rear_clear else 0.90
                result.steer = 0.0
                if hasattr(result, "reverse"):
                    result.reverse = bool(rear_clear)
                return result, {
                    "source": "ordinary_class_05_timeout_recovery",
                    "state": (
                        "CLASS05_TIMEOUT_REVERSE"
                        if rear_clear else "CLASS05_TIMEOUT_REAR_HOLD"
                    ),
                    "override_source": (
                        "class05_timeout_reverse_unstick"
                        if rear_clear else "class05_timeout_rear_safety_hold"
                    ),
                    "class05_phase": self.phase,
                    "class05_timeout_recovery_active": True,
                    "class05_timeout_recovery_reverse_m": (
                        self._class05_timeout_recovery_reverse_m
                    ),
                    "class05_timeout_recovery_reverse_target_m": (
                        reverse_target_m
                    ),
                    "class05_timeout_recovery_rear_m": rear_distance,
                }

            oncoming_now = self._update_oncoming(
                radar.get("_online_radar_packet"), 0.0, now_s
            )
            self._update_flow_pass(oncoming_now)
            oncoming = (
                None
                if self.oncoming_ttc_s is None
                else (
                    self.oncoming_distance_m,
                    self.oncoming_closing_mps,
                    self.oncoming_ttc_s,
                )
            )
            if oncoming_now:
                self._class05_timeout_recovery_gap_clear_frames = 0
            elif not self.oncoming_seen:
                self._class05_timeout_recovery_gap_clear_frames += 1
            # A five-frame disappearance is not a usable opening on these
            # curved two-way routes.  The next actor in the distance-spaced
            # flow can still be outside the radar horizon and catch the ego
            # before the 30 m train is cleared.  Reuse Class05's measured
            # two-second post-pass horizon qualification here as well.
            gap_ready = bool(
                self.oncoming_pass_confirmed
                and self.oncoming_pass_clear_frames >= self.GAP_CLEAR_FRAMES
            )
            if not gap_ready:
                result = self._hard_stop(control)
                return result, {
                    "source": "ordinary_class_05_timeout_recovery",
                    "state": "CLASS05_TIMEOUT_WAIT_GAP",
                    "override_source": "class05_timeout_oncoming_gap_hold",
                    "class05_phase": self.phase,
                    "class05_timeout_recovery_active": True,
                    "class05_timeout_recovery_gap_clear_frames": (
                        self._class05_timeout_recovery_gap_clear_frames
                    ),
                    "class05_timeout_recovery_oncoming": oncoming,
                    "class05_timeout_recovery_oncoming_seen": (
                        self.oncoming_seen
                    ),
                    "class05_timeout_recovery_pass_confirmed": (
                        self.oncoming_pass_confirmed
                    ),
                    "class05_timeout_recovery_pass_clear_frames": (
                        self.oncoming_pass_clear_frames
                    ),
                }

            if self._class05_static_replan_recovery_mode:
                # The abandoned finite track consumed its original approach
                # envelope.  After the bounded rear-safe reverse and a fresh
                # observed traffic gap, use the full category-safe corridor;
                # the inherited 1.45 m request already failed on the second
                # accident body.
                self._class05_timeout_recovery_active = False
                self._class05_timeout_recovery_mode = False
                self._class05_static_replan_recovery_mode = False
                replan_progress_m = max(
                    0.0,
                    self._class05_static_replan_progress_m
                    - self._class05_timeout_recovery_reverse_m,
                )
                self._start_static_dense_route(
                    radar,
                    progress_m=replan_progress_m,
                    locked_offset_m=self.STATIC_DENSE_CORRIDOR_M,
                )
                self._class05_side_launch_event = False
                dense_result, dense_debug = self._apply_side_route(
                    control, radar, features, event=True
                )
                dense_debug["class05_static_path_fallback"] = True
                dense_debug["class05_static_replan_from_reverse"] = True
                dense_debug["class05_static_path_fallback_evidence"] = (
                    self._class05_static_replan_evidence
                )
                return dense_result, dense_debug

            self._class05_timeout_recovery_active = False
            self._class05_timeout_path_active = True
            self._class05_timeout_path_cycle += 1
            self._class05_timeout_path_last_progress_m = 0.0
            self._class05_timeout_path_no_progress_frames = 0
            self._class05_timeout_path_plan = self._build_timeout_local_plan(
                radar.get("_route_lateral_m")
            )
            self._class05_static_latched = True
            return self._apply_timeout_local_path(
                control, radar, features, event=True
            )

        if self._class05_post_route_hold:
            post_route_vehicle = self._near_post_route_crossing_vehicle(features)
            if post_route_vehicle is None:
                self._class05_post_route_clear_frames += 1
            else:
                self._class05_post_route_clear_frames = 0
            if (
                self._class05_post_route_clear_frames
                < self.POST_ROUTE_CLEAR_FRAMES
            ):
                return self._hard_stop(control), {
                    "source": "ordinary_class_05_dense_side_vehicle_avoid",
                    "state": "CLASS05_POST_ROUTE_CROSSING_HOLD",
                    "override_source": "class05_post_route_crossing_stop",
                    "class05_phase": self.phase,
                    "class05_post_route_hold": True,
                    "class05_post_route_clear_frames": (
                        self._class05_post_route_clear_frames
                    ),
                    "class05_post_route_vehicle": post_route_vehicle,
                    "class05_side_vehicle_plan": False,
                    "class05_moving_hold": False,
                }
            self._class05_post_route_hold = False
            self._class05_post_route_clear_frames = 0
            self.maneuver_complete = True

        feedback = radar.get("_local_path_feedback") or {}
        raw_live_route_lateral = feedback.get("live_route_lateral_m")
        live_route_lateral = feedback.get("filtered_live_route_lateral_m")
        if live_route_lateral is None:
            live_route_lateral = raw_live_route_lateral
        handoff_route_lateral = (
            _number(live_route_lateral, 99.0)
            if live_route_lateral is not None
            else _number(feedback.get("route_lateral_m"), 99.0)
        )
        if self._class05_static_replan_pending:
            if approach_speed > 0.12:
                return self._hard_stop(control), {
                    "source": "ordinary_class_05_accident_stop_plan_left",
                    "state": "CLASS05_STATIC_REPLAN_BRAKE",
                    "override_source": "class05_static_replan_brake",
                    "class05_phase": self.phase,
                    "class05_static_latched": self._class05_static_latched,
                    "class05_static_replan_pending": True,
                    "class05_static_replan_evidence": (
                        self._class05_static_replan_evidence
                    ),
                }
            # Stopping alone still left the ego beside the next train body.
            # Arm the existing rear-aware reverse and fresh-gap lifecycle so
            # the new path starts with physical steering room and current
            # opposing-flow evidence.
            self._class05_static_replan_pending = False
            self._class05_static_replan_recovery_mode = True
            self._class05_timeout_recovery_active = True
            self._class05_timeout_recovery_mode = True
            self._class05_timeout_recovery_reverse_m = 0.0
            self._class05_timeout_recovery_last_s = now_s
            self._class05_timeout_recovery_gap_clear_frames = 0
            self.oncoming_seen = False
            self.oncoming_distance_m = None
            self.oncoming_closing_mps = None
            self.oncoming_ttc_s = None
            self.oncoming_last_seen_s = None
            self.oncoming_last_update_s = None
            self.oncoming_near_seen = False
            self.oncoming_pass_clear_frames = 0
            self.oncoming_pass_confirmed = False
            self.post_pass_clear_eligible = False
            return self._hard_stop(control), {
                "source": "ordinary_class_05_accident_stop_plan_left",
                "state": "CLASS05_STATIC_REPLAN_REVERSE_ARM",
                "override_source": "class05_static_replan_reverse_arm",
                "class05_phase": self.phase,
                "class05_static_latched": True,
                "class05_static_replan_recovery_mode": True,
                "class05_static_replan_evidence": (
                    self._class05_static_replan_evidence
                ),
            }
        handoff_to_model = bool(
            self.phase == "BYPASS"
            and _number(feedback.get("x_m"), 0.0) >= self.HANDOFF_MIN_PROGRESS_M
            and abs(handoff_route_lateral) <= self.HANDOFF_ROUTE_LATERAL_M
        )
        if handoff_to_model:
            self.phase = "COOLDOWN"
            self.phase_frames = 0
            self.cooldown_frames = self.COOLDOWN_FRAMES
            self.plan = None
            self.maneuver_complete = True

        dynamic = _number(radar.get("front_dynamic_count"), 0.0)
        closing = _number(radar.get("front_closing_count"), 0.0)
        immediate = bool(getattr(features, "immediate_hazard", False))
        # Keep this a lane-obstacle gate for the complete approach.  Town12
        # contains dense curbside parking; widening to 3.8 m after 20 m made
        # one of those legal parked cars look like the accident and moved the
        # stopped-pose origin roughly 30 m before the actual police/vehicle
        # train.  The official accident bodies are only about one metre from
        # route centre and remain inside the strict 2.2 m gate.
        approach_lateral_gate = self.APPROACH_STATIC_MAX_LATERAL_M
        lane_static_vehicle = self._near_lane_static_vehicle(
            features, max_lateral_m=approach_lateral_gate
        )
        stationary_now = bool(lane_static_vehicle is not None)
        static_latched_before_observation = self._class05_static_latched
        self._class05_static_frames = (
            self._class05_static_frames + 1 if stationary_now else 0
        )
        if (
            self._class05_static_frames
            >= self.APPROACH_STATIC_CONFIRM_FRAMES
        ):
            self._class05_static_latched = True
        if (
            self._class05_static_latched
            and not static_latched_before_observation
        ):
            # Flow evidence gathered while the ego was still approaching is
            # not a departure gap from the final stopped planning pose.  In
            # particular, a vehicle that passed during the model-first
            # approach used to leave ``oncoming_pass_confirmed`` latched; the
            # inherited planner then launched one frame after becoming ready,
            # just before the next actor entered radar range.  Begin a fresh
            # traffic-observation transaction at the physical obstacle stop.
            # A persistently empty corridor will still launch after the normal
            # bounded WAIT_GAP confirmation, while visible flow must produce a
            # new measured pass/gap.
            self.oncoming_seen = False
            self.oncoming_distance_m = None
            self.oncoming_closing_mps = None
            self.oncoming_ttc_s = None
            self.oncoming_last_seen_s = None
            self.oncoming_last_update_s = None
            self.oncoming_near_seen = False
            self.oncoming_pass_clear_frames = 0
            self.oncoming_pass_confirmed = False
            self.post_pass_clear_eligible = False
            self.gap_clear_frames = 0
            self.entry_safe_frames = 0
            self._class05_static_observed_gap_clear_frames = 0
            self._class05_unsafe_prelaunch_latched = False
            self._class05_prelaunch_safe_frames = 0

        moving_vehicle = None
        if not self._class05_static_latched and not self.maneuver_complete:
            moving_vehicle = self._near_moving_vehicle(features)
        if moving_vehicle is not None:
            self._class05_moving_hold = True
            self._class05_moving_clear_frames = 0
            self._class05_moving_positions.append((
                moving_vehicle["lateral_m"], moving_vehicle["forward_m"]
            ))
        elif self._class05_moving_hold:
            self._class05_moving_clear_frames += 1
            if self._class05_moving_clear_frames >= self.MOVING_HOLD_CLEAR_FRAMES:
                self._class05_moving_hold = False
                self._class05_moving_clear_frames = 0
                self._class05_moving_positions.clear()
        if self._class05_static_latched:
            # A now-stationary close hazard belongs to the normal Class05
            # stop/plan/gap lifecycle, not this temporary observation hold.
            self._class05_moving_hold = False
            self._class05_moving_clear_frames = 0
            self._class05_moving_positions.clear()

        if (
            self._class05_moving_hold
            and not self._class05_side_wait_gap
            and not self._class05_side_route_active
        ):
            self._update_stopped_planning_flow(
                radar.get("_online_radar_packet"), now_s
            )

        stable_vehicle = bool(
            moving_vehicle is not None
            and self._moving_vehicle_is_position_stable(
                getattr(features, "ego_speed", 0.0)
            )
        )
        stable_side_vehicle = bool(
            stable_vehicle and abs(moving_vehicle["lateral_m"]) >= 1.2
        )
        stable_center_vehicle = bool(
            stable_vehicle and abs(moving_vehicle["lateral_m"]) <= 0.6
        )
        stable_bypass_vehicle = bool(
            (stable_side_vehicle or stable_center_vehicle)
            and not self._class05_side_wait_gap
            and not self._class05_side_route_active
            and not self._class05_side_vehicle_plan
        )
        if stable_bypass_vehicle:
            self._class05_moving_hold = False
            self._class05_moving_clear_frames = 0
            self._class05_side_vehicle_plan = True
            self._class05_side_route_active = False
            # Preserve the online distinction that produced formal 100-point
            # results on both fixed development geometries.  A position-stable
            # detector track with non-zero reported velocity uses an adaptive
            # corridor; only the independently latched static train uses the
            # fixed fast corridor.  Both share the train minimum-pass guard.
            self._class05_static_dense_mode = False
            self._class05_static_dense_locked_offset_m = None
            self._class05_static_dense_corridor_acquired = False
            self._class05_side_wait_gap = True
            self._class05_side_gap_clear_frames = 0
            self._class05_side_route_progress_m = 0.0
            self._class05_side_route_last_s = _number(
                radar.get("_timestamp"), 0.0
            )
            # AccidentTwoWays requires a physical-left invasion of the
            # opposing lane.  Detector lateral estimates jitter around the
            # partly occluded accident body and are valid stop evidence, but
            # must not randomly mirror the committed category plan toward the
            # road edge/accident train.
            self._class05_side_route_sign = -1.0
            self._class05_side_route_peak_offset_m = self.SIDE_ROUTE_OFFSET_M
            self._class05_side_route_close_seen = False
            self._class05_side_route_clear_frames = 0
            self._class05_side_route_clear_start_m = None
            self._class05_post_route_recent_vehicle = None
            self._class05_post_route_evidence_age_frames = (
                self.POST_ROUTE_EVIDENCE_MEMORY_FRAMES + 1
            )
            self._class05_moving_positions.clear()
            # Planning happens while fully stopped, so its observed leading
            # pass is temporally local and must remain usable.  Clearing it
            # here made the ego miss the immediate empty interval and wait
            # until the following actor approached.

        if self._class05_side_wait_gap:
            oncoming_now = self._update_stopped_planning_flow(
                radar.get("_online_radar_packet"), now_s
            )
            if oncoming_now:
                self._class05_side_gap_clear_frames = 0
            elif not self.oncoming_seen:
                self._class05_side_gap_clear_frames += 1
            ready = bool(
                (self.oncoming_pass_confirmed and not oncoming_now)
                or self._class05_side_gap_clear_frames
                >= self.SIDE_GAP_CLEAR_FRAMES
            )
            if ready:
                self._class05_side_wait_gap = False
                self._class05_side_route_active = True
                self._class05_side_route_last_s = now_s
                self._class05_side_launch_event = True
            else:
                return self._hard_stop(control), {
                    "source": "ordinary_class_05_dense_side_vehicle_avoid",
                    "state": "CLASS05_SIDE_WAIT_GAP",
                    "override_source": "class05_side_oncoming_gap_hold",
                    "class05_phase": self.phase,
                    "class05_side_wait_gap": True,
                    "class05_side_gap_clear_frames": (
                        self._class05_side_gap_clear_frames
                    ),
                    "class05_side_oncoming_seen": self.oncoming_seen,
                    "class05_side_oncoming_ttc_s": self.oncoming_ttc_s,
                    "class05_side_oncoming_pass_confirmed": (
                        self.oncoming_pass_confirmed
                    ),
                }

        if self._class05_side_route_active:
            launch_event = bool(
                stable_bypass_vehicle or self._class05_side_launch_event
            )
            self._class05_side_launch_event = False
            return self._apply_side_route(
                control, radar, features, event=launch_event
            )

        if self._class05_moving_hold:
            result = clone_control(control)
            result.throttle = 0.0
            result.brake = max(0.90, _number(getattr(control, "brake", 0.0)))
            result.steer = 0.0
            if hasattr(result, "reverse"):
                result.reverse = False
            return result, {
                "source": "ordinary_class_05_accident_stop_plan_left",
                "state": "CLASS05_MOVING_VEHICLE_HOLD",
                "override_source": "class05_near_moving_vehicle_stop",
                "class05_phase": self.phase,
                "class05_static_frames": self._class05_static_frames,
                "class05_static_latched": self._class05_static_latched,
                "class05_model_first": True,
                "class05_model_handoff": handoff_to_model,
                "class05_moving_hold": True,
                "class05_moving_clear_frames": self._class05_moving_clear_frames,
                "class05_moving_position_frames": len(
                    self._class05_moving_positions
                ),
                "class05_moving_vehicle": moving_vehicle,
            }

        if not self._class05_static_latched and not self.maneuver_complete:
            # Keep learned lateral route following until a real close vehicle
            # body is confirmed.  A single first frame already commands a
            # stop; the second frame latches the obstacle and transfers into
            # the normal stop/plan/gap lifecycle.
            result = self._approach_control(
                control, features, obstacle=lane_static_vehicle
            )
            return result, {
                "source": "ordinary_class_05_accident_approach",
                "state": (
                    "CLASS05_OBSTACLE_BRAKE_ACQUIRE"
                    if lane_static_vehicle is not None
                    else "CLASS05_OBSTACLE_APPROACH"
                ),
                "override_source": (
                    "class05_detector_vehicle_stop"
                    if lane_static_vehicle is not None
                    else "class05_model_route_approach"
                ),
                "class05_phase": self.phase,
                "class05_static_frames": self._class05_static_frames,
                "class05_static_latched": False,
                "class05_model_first": True,
                "class05_moving_hold": False,
                "class05_lane_static_vehicle": lane_static_vehicle,
                "class05_approach_progress_m": (
                    self._class05_approach_progress_m
                ),
                "class05_approach_lateral_gate_m": approach_lateral_gate,
                "class05_approach_target_speed_mps": (
                    self.APPROACH_TARGET_SPEED_MPS
                ),
            }

        local_radar = dict(radar)
        # The construction warning cue is shared with Class04 but does not
        # identify the accident's physical blockage.  Before the close static
        # hazard is confirmed, preserve the learned controller byte-for-byte.
        local_radar["_class04_startup_cue_seen"] = False
        if str(local_radar.get("_classifier_reason", "")) == (
            "semantic_static_straight_obstacle_start"
        ):
            local_radar["_classifier_reason"] = "class05_model_first"
        if not self._class05_static_latched:
            local_radar["_prelock_obstacle_hold"] = False
            local_radar["front_min"] = max(
                50.0, _number(local_radar.get("front_min"), 50.0)
            )
            local_radar["front_close_count"] = 0.0
            local_radar["front_dynamic_count"] = 0.0
            local_radar["front_closing_count"] = 0.0
        else:
            local_radar["_prelock_obstacle_hold"] = True

        unsafe_prelaunch_oncoming = False
        if (
            self._class05_static_latched
            and self.phase in {"PLAN", "WAIT_GAP"}
            and not self._class05_side_route_active
        ):
            raw_oncoming = self._oncoming_metrics(
                radar.get("_online_radar_packet"),
                0.0,
                self.CONFLICT_SPEED_MPS,
            )
            recent_ttc = (
                raw_oncoming[2]
                if raw_oncoming is not None
                else self.oncoming_ttc_s
            )
            recent_age_s = (
                max(0.0, now_s - self.oncoming_last_seen_s)
                if self.oncoming_last_seen_s is not None
                else None
            )
            current_unsafe_oncoming = bool(
                recent_ttc is not None
                and recent_ttc < self._required_clearance_time_s()
                and (
                    raw_oncoming is not None
                    or (
                        recent_age_s is not None
                        and recent_age_s <= self.ONCOMING_DROPOUT_HOLD_S
                    )
                )
            )
            if current_unsafe_oncoming:
                self._class05_unsafe_prelaunch_latched = True
                self._class05_prelaunch_safe_frames = 0
            elif (
                self._class05_unsafe_prelaunch_latched
                and raw_oncoming is not None
                and recent_ttc is not None
                and recent_ttc >= self._required_clearance_time_s()
            ):
                self._class05_prelaunch_safe_frames += 1
                if (
                    self._class05_prelaunch_safe_frames
                    >= self.ENTRY_SAFE_CONFIRM_FRAMES
                ):
                    self._class05_unsafe_prelaunch_latched = False
                    self.oncoming_pass_confirmed = True
            elif self._class05_unsafe_prelaunch_latched:
                self._class05_prelaunch_safe_frames = 0
            unsafe_prelaunch_oncoming = bool(
                self._class05_unsafe_prelaunch_latched
            )
            if unsafe_prelaunch_oncoming:
                # An empty corridor observed while the local plan was being
                # built is no longer a usable gap once a new actor appears.
                # Revoke the stale confirmation before Rule04 can consume a
                # one-frame radar dropout as an immediate launch signal.
                self.oncoming_pass_confirmed = False
                self.post_pass_clear_eligible = False
                self.oncoming_pass_clear_frames = 0
                self.gap_clear_frames = 0
                self.entry_safe_frames = 0
            if raw_oncoming is not None:
                self._class05_static_observed_gap_clear_frames = 0
            elif self.oncoming_seen:
                self._class05_static_observed_gap_clear_frames += 1
            if (
                self._class05_static_observed_gap_clear_frames
                >= self.STATIC_OBSERVED_GAP_CLEAR_FRAMES
                and (
                    not self._class05_unsafe_prelaunch_latched
                    or self.oncoming_near_seen
                )
            ):
                # The stopped ego saw a leading actor and raw radar now shows
                # a short continuous empty interval.  Preserve that physical
                # pass even if the generic observer missed its TTC<=0.4 frame;
                # otherwise Class05 can wait forever after the actor vanished.
                self.oncoming_pass_confirmed = True
                self.post_pass_clear_eligible = True
                self.oncoming_pass_clear_frames = self.GAP_CLEAR_FRAMES
                self.oncoming_seen = False
                self.oncoming_last_seen_s = None
                self.oncoming_last_update_s = None
                self.oncoming_distance_m = None
                self.oncoming_closing_mps = None
                self.oncoming_ttc_s = None
                self.gap_clear_frames = self.GAP_CLEAR_FRAMES
                self._class05_unsafe_prelaunch_latched = False
                self._class05_prelaunch_safe_frames = 0

        phase_before_base = self.phase
        result, debug = super(Rule05, self).apply(
            control, local_radar, features
        )
        if (
            self._class05_static_latched
            and phase_before_base != "BYPASS"
            and self.phase == "BYPASS"
            and not self._class05_side_route_active
            and not self.maneuver_complete
        ):
            if unsafe_prelaunch_oncoming:
                self.phase = "WAIT_GAP"
                self.phase_frames = 0
                result = self._hard_stop(result)
                debug = dict(debug)
                debug["state"] = "CLASS05_WAIT_GAP"
                debug["override_source"] = (
                    "class05_recent_unsafe_oncoming_hold"
                )
                debug["class05_unsafe_prelaunch_oncoming"] = True
                debug["class05_unsafe_prelaunch_ttc_s"] = recent_ttc
                return result, debug
            # Start road-relative planning from the original stopped pose.
            # Letting the finite frozen-tangent path move first can put a
            # curved route several metres onto the obstacle side; no bounded
            # recovery can then acquire the opposite corridor before contact.
            launch_corridor_m, launch_basis = self._static_launch_corridor(
                radar
            )
            self._class05_static_dense_corridor_basis = launch_basis
            self._start_static_dense_route(
                radar,
                progress_m=0.0,
                locked_offset_m=launch_corridor_m,
            )
            dense_result, dense_debug = self._apply_side_route(
                control, radar, features, event=True
            )
            dense_debug["class05_static_initial_dense_handoff"] = True
            return dense_result, dense_debug
        if (
            self._class05_static_latched
            and self.phase == "BYPASS"
            and not self._class05_side_route_active
            and not self._class05_timeout_recovery_mode
            and not self.maneuver_complete
        ):
            path_x_m = _number(feedback.get("x_m"), 0.0)
            route_pid_target_m = feedback.get("route_pid_target_m")
            # Three consecutive raw samples already provide the temporal
            # debounce.  Reusing the five-frame median here delayed the
            # curved-route correction until the ego was physically wedged.
            wrong_side_route_lateral = (
                raw_live_route_lateral
                if raw_live_route_lateral is not None
                else live_route_lateral
            )
            path_wrong_side = bool(
                path_x_m >= self.STATIC_PATH_FALLBACK_PROGRESS_MIN_M
                and wrong_side_route_lateral is not None
                and route_pid_target_m is not None
                and _number(wrong_side_route_lateral, 0.0)
                > self.STATIC_PATH_FALLBACK_LIVE_WRONG_SIDE_M
                and _number(route_pid_target_m, 0.0)
                < self.STATIC_PATH_FALLBACK_TARGET_M
            )
            tracking_error_m = (
                _number(wrong_side_route_lateral, 0.0)
                - _number(route_pid_target_m, 0.0)
                if wrong_side_route_lateral is not None
                and route_pid_target_m is not None
                else 0.0
            )
            critical_tracking_error = bool(
                path_x_m >= self.STATIC_PATH_FALLBACK_PROGRESS_MIN_M
                and tracking_error_m >= self.STATIC_PATH_CRITICAL_ERROR_M
            )
            if path_wrong_side:
                self._class05_static_path_wrong_side_frames += 1
            else:
                self._class05_static_path_wrong_side_frames = 0
            if critical_tracking_error or (
                self._class05_static_path_wrong_side_frames
                >= self.STATIC_PATH_FALLBACK_CONFIRM_FRAMES
            ):
                # A failed finite track is an obstacle hazard.  Brake first,
                # then rebuild the remaining route from the stopped pose;
                # changing steering ownership at conflict speed was too late.
                locked_offset_m = abs(
                    _number(
                        route_pid_target_m,
                        self.NOMINAL_LEFT_CORRIDOR_M,
                    )
                )
                self._class05_static_replan_pending = True
                self._class05_static_replan_progress_m = path_x_m
                self._class05_static_replan_offset_m = locked_offset_m
                self._class05_static_replan_evidence = {
                    "x_m": path_x_m,
                    "live_route_lateral_m": _number(
                        wrong_side_route_lateral, 0.0
                    ),
                    "filtered_live_route_lateral_m": _number(
                        live_route_lateral, 0.0
                    ),
                    "route_pid_target_m": _number(
                        route_pid_target_m, 0.0
                    ),
                    "confirm_frames": (
                        self._class05_static_path_wrong_side_frames
                    ),
                    "tracking_error_m": tracking_error_m,
                    "critical_tracking_error": critical_tracking_error,
                }
                return self._hard_stop(control), {
                    "source": "ordinary_class_05_accident_stop_plan_left",
                    "state": "CLASS05_STATIC_REPLAN_BRAKE",
                    "override_source": "class05_static_replan_brake",
                    "class05_phase": self.phase,
                    "class05_static_latched": True,
                    "class05_static_replan_pending": True,
                    "class05_static_replan_evidence": (
                        self._class05_static_replan_evidence
                    ),
                }
        debug["source"] = "ordinary_class_05_accident_stop_plan_left"
        debug["class05_phase"] = self.phase
        debug["class05_static_frames"] = self._class05_static_frames
        debug["class05_static_latched"] = self._class05_static_latched
        debug["class05_static_path_wrong_side_frames"] = (
            self._class05_static_path_wrong_side_frames
        )
        debug["class05_model_first"] = not self._class05_static_latched
        debug["class05_model_handoff"] = handoff_to_model
        debug["class05_moving_hold"] = self._class05_moving_hold
        debug["class05_moving_clear_frames"] = self._class05_moving_clear_frames
        debug["class05_moving_vehicle"] = moving_vehicle
        debug["class05_side_vehicle_plan"] = self._class05_side_vehicle_plan
        if stable_bypass_vehicle:
            debug["class05_side_vehicle_event"] = (
                "stable_center_pose_plan_left"
                if stable_center_vehicle
                else "stable_relative_pose_plan_left"
            )
        if str(debug.get("state", "")).startswith("CLASS04_"):
            debug["state"] = "CLASS05_" + str(debug["state"])[8:]
        override = str(debug.get("override_source", "") or "")
        if override.startswith("class04_"):
            debug["override_source"] = "class05_" + override[8:]
        if handoff_to_model:
            debug["override_source"] = "class05_model_handoff_after_rejoin"
            debug["state"] = "CLASS05_REJOINED_MODEL_CONTROL"
        return result, debug


def build_rule():
    return Rule05(get_config(5))
