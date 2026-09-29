"""Class-04 construction bypass built around stop, plan, gap, execute."""

from __future__ import absolute_import

import math
from collections import deque

from .common import CategoryRule, radar_safety_action
from .configs import get_config


def _number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


class Rule04(CategoryRule):
    """Stop before the closure and execute one online left-bypass plan.

    The rule deliberately has no reverse manoeuvre and no route-specific
    geometry. XML was used offline only to establish the common topology:
    the ego lane is closed and the legal escape is through the opposing lane.
    Runtime decisions use radar, ego state and route-relative feedback only.
    """

    PHASES = {
        "CRUISE", "APPROACH", "BRAKE", "PLAN", "WAIT_GAP",
        "BYPASS", "COOLDOWN",
    }
    OBSTACLE_ACQUIRE_FRAMES = 2
    OBSTACLE_MAX_M = 12.0
    BRAKE_CONFIRM_FRAMES = 3
    BRAKE_TIMEOUT_FRAMES = 80
    # All five official class definitions put the actual construction 40--45m
    # beyond the online warning cue, with that cue 10--13m into the route.
    # The scenario enables opposing flow 30m before the construction and then
    # waits five seconds.  Approach the common activation envelope on the live
    # route before stopping; never treat the warning cue itself as the lane
    # blockage.
    APPROACH_DISTANCE_M = 28.0
    APPROACH_MAX_FRAMES = 500
    PLAN_OBSERVE_FRAMES = 50
    PLAN_MIN_GAP_OBSERVE_FRAMES = 40
    GAP_CLEAR_FRAMES = 6
    GAP_MAX_FRAMES = 900
    ENTRY_SAFE_CONFIRM_FRAMES = 3
    FLOW_PASS_CLEAR_FRAMES = 5
    BYPASS_MAX_FRAMES = 900
    BYPASS_NO_PROGRESS_FRAMES = 100
    COOLDOWN_FRAMES = 100

    # From the staging point the construction is 25--30m ahead.  Shift well
    # before the first transverse warning mesh, remain fully left through the
    # complete cone/debris footprint, then rejoin.  These are class-level
    # bounds learned offline from all five definitions, not route identities.
    # Stop at the staging point and shift immediately.  Route1825 proved that
    # moving another 10 m in-lane leaves only about 3 m to the first warning
    # prop, which is insufficient for a vehicle-width lateral displacement.
    # Relative to the original stop, the common construction footprint ends
    # around 28 m; rejoin only after that envelope.
    SHIFT_START_M = 0.0
    SHIFT_END_M = 9.0
    HOLD_END_M = 42.0
    REJOIN_END_M = 52.0
    FINISH_M = 58.0

    LEFT_CORRIDOR_CANDIDATES_M = (-2.8, -3.0, -3.2, -3.4)
    NOMINAL_LEFT_CORRIDOR_M = -3.2
    BODY_CLEARANCE_M = 1.15
    CONFLICT_SPEED_MPS = 8.5
    ENTRY_MARGIN_S = 0.8
    ENTRY_ACCEL_MPS2 = 3.0
    CONFIRMED_CONSTRUCTION_APPROACH_SPEED_MPS = 8.5
    ONCOMING_DROPOUT_HOLD_S = 0.45
    CRITICAL_TTC_S = 3.0
    # Construction traffic can expose a usable slot immediately behind a
    # measured leading actor.  Categories whose following actor is commonly
    # occluded may disable this and require that actor's TTC to be measured.
    ALLOW_POST_PASS_SILENCE = True

    def __init__(self, cfg):
        super(Rule04, self).__init__(cfg)
        self._reset_private()

    def _reset_private(self):
        self.phase = "CRUISE"
        self.phase_frames = 0
        self.cycle = 0
        self.obstacle_frames = 0
        self.stop_frames = 0
        self.entry_front_m = None
        self.plan = None
        self.corridor_samples = deque(maxlen=9)
        self.oncoming_seen = False
        self.oncoming_distance_m = None
        self.oncoming_closing_mps = None
        self.oncoming_ttc_s = None
        self.oncoming_last_seen_s = None
        self.oncoming_last_update_s = None
        self.gap_clear_frames = 0
        self.entry_safe_frames = 0
        self.oncoming_near_seen = False
        self.oncoming_pass_clear_frames = 0
        self.oncoming_pass_confirmed = False
        self.post_pass_clear_eligible = False
        self.last_progress_m = 0.0
        self.no_progress_frames = 0
        self.cooldown_frames = 0
        self.startup_warning_cue = False
        self.approach_plan = None
        self.approach_progress_m = 0.0
        # Class05 and Class04 share an early side-vehicle signature.  The
        # router consumes a genuine Class05 early stop on the following tick;
        # if Rule04 is called again, ownership stayed with construction and
        # the side return must not turn the warning cue into the bypass start.
        self.early_side_stop_pending = False
        self.early_side_recovery_active = False
        self.maneuver_complete = False
        self.construction_context_seen = False

    def reset(self):
        super(Rule04, self).reset()
        self._reset_private()

    @staticmethod
    def _stop(control):
        control.throttle = 0.0
        control.brake = max(_number(getattr(control, "brake", 0.0)), 0.85)
        control.steer = 0.0
        control.reverse = False
        if hasattr(control, "manual_gear_shift"):
            control.manual_gear_shift = False

    @staticmethod
    def _cancel_reason(features):
        if bool(getattr(features, "red_light_active", False)):
            return "red_light"
        if getattr(features, "front_pedestrian_distance", None) is not None:
            return "pedestrian"
        # A category decision is allowed to mature after the physical hazard
        # appears.  Never keep executing the construction staging trajectory
        # through a close object merely because class-04 is still latched.
        # Holding the current plan is safe: once the hazard clears (or the
        # classifier promotes to the real class), normal execution resumes.
        if bool(getattr(features, "immediate_hazard", False)):
            return "immediate_hazard"
        kind = str(getattr(features, "traffic_control_kind", "") or "").lower()
        if kind in {"traffic_light", "trafficlight", "stop_sign", "stopsign"}:
            return "traffic_control"
        return ""

    @staticmethod
    def _construction_context(features):
        if _number(
            getattr(features, "construction_object_count", 0), 0.0
        ) >= 1.0:
            return True
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() != "traffic_cone":
                continue
            if _number(item.get("score", 0.0), 0.0) >= 0.15:
                return True
        return False

    @staticmethod
    def _centered_close_vehicle(features):
        """Return a semantic vehicle whose centre is in the ego corridor."""
        for item in list(getattr(features, "detector_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "vehicle", "van", "truck", "bus",
            }:
                continue
            if _number(item.get("score", 0.0), 0.0) < 0.30:
                continue
            box = item.get("box_lidar") or item
            lateral = _number(box.get("x"), 99.0)
            forward = _number(box.get("y"), -1.0)
            if 0.5 <= forward <= 10.0 and abs(lateral) <= 1.5:
                return True
        return False

    @staticmethod
    def _radar_points(packet):
        if isinstance(packet, (tuple, list)) and len(packet) == 2:
            packet = packet[1]
        try:
            rows = packet.tolist()
        except AttributeError:
            rows = packet or []
        except Exception:
            return []
        points = []
        try:
            for row in rows:
                if len(row) < 4:
                    continue
                depth, altitude, azimuth, velocity = map(float, row[:4])
                if not all(math.isfinite(v) for v in (depth, altitude, azimuth, velocity)):
                    continue
                if depth >= 1.0 and -0.22 <= altitude <= 0.12:
                    points.append((depth, altitude, azimuth, velocity))
        except (TypeError, ValueError):
            return []
        return points

    def _observe_left_corridor(self, packet, route_lateral_m):
        """Select the least occupied opposing-lane centre from live radar."""
        route_lateral = _number(route_lateral_m, 0.0)
        static_points = []
        for depth, _, azimuth, velocity in self._radar_points(packet):
            forward = depth * math.cos(azimuth)
            lateral = depth * math.sin(azimuth)
            if 0.5 <= forward <= 22.0 and abs(velocity) <= 0.6:
                static_points.append((forward, lateral))
        if len(static_points) < 2:
            return
        scored = []
        for candidate in self.LEFT_CORRIDOR_CANDIDATES_M:
            local_y = candidate - route_lateral
            penetration = sum(
                max(0.0, self.BODY_CLEARANCE_M - abs(y - local_y))
                for _, y in static_points
            )
            scored.append((penetration, abs(candidate - self.NOMINAL_LEFT_CORRIDOR_M), candidate))
        self.corridor_samples.append(min(scored)[2])

    def _selected_corridor(self):
        if not self.corridor_samples:
            return self.NOMINAL_LEFT_CORRIDOR_M
        values = sorted(self.corridor_samples)
        return values[len(values) // 2]

    @classmethod
    def _oncoming_metrics(cls, packet, ego_speed, planned_speed):
        """Return nearest approaching actor in the forward opposing corridor."""
        candidates = []
        for depth, _, azimuth, velocity in cls._radar_points(packet):
            if abs(azimuth) > math.radians(32.0):
                continue
            compensated = velocity + abs(ego_speed) * math.cos(azimuth)
            if compensated <= -1.0:
                candidates.append((depth, -compensated))
        if not candidates:
            return None
        depth, closing = min(candidates)
        ttc = depth / max(0.1, closing + max(0.0, planned_speed))
        return depth, closing, ttc

    def _update_oncoming(self, packet, speed, now_s):
        observed = self._oncoming_metrics(packet, speed, self.CONFLICT_SPEED_MPS)
        if observed is not None:
            self.oncoming_seen = True
            self.oncoming_distance_m, self.oncoming_closing_mps, self.oncoming_ttc_s = observed
            self.oncoming_last_seen_s = now_s
            self.oncoming_last_update_s = now_s
            return True
        if (
            self.oncoming_last_seen_s is not None
            and 0.0 <= now_s - self.oncoming_last_seen_s <= self.ONCOMING_DROPOUT_HOLD_S
            and self.oncoming_ttc_s is not None
        ):
            dt = max(0.0, now_s - _number(self.oncoming_last_update_s, now_s))
            self.oncoming_ttc_s = max(0.0, self.oncoming_ttc_s - dt)
            self.oncoming_last_update_s = now_s
            return True
        self.oncoming_distance_m = None
        self.oncoming_closing_mps = None
        self.oncoming_ttc_s = None
        return False

    def _update_flow_pass(self, oncoming_now):
        """Confirm a leading oncoming actor passed before using a trailing gap."""
        if oncoming_now:
            self.oncoming_pass_clear_frames = 0
            if (
                self.oncoming_ttc_s is not None
                and self.oncoming_ttc_s <= 0.4
            ):
                self.oncoming_near_seen = True
                self.post_pass_clear_eligible = True
            elif (
                self.post_pass_clear_eligible
                and self.oncoming_ttc_s is not None
                and self.oncoming_ttc_s >= 2.0
            ):
                # A trailing actor is already visible. Any later silence is
                # dropout of that actor, not the immediate gap behind the
                # leading actor that reached the pass zone.
                self.oncoming_pass_confirmed = True
                self.post_pass_clear_eligible = False
            return
        if self.post_pass_clear_eligible:
            self.oncoming_pass_clear_frames += 1
            if self.oncoming_pass_clear_frames >= self.FLOW_PASS_CLEAR_FRAMES:
                self.oncoming_pass_confirmed = True
        else:
            self.oncoming_pass_clear_frames = 0

    def _build_approach_plan(self, route_lateral_m):
        origin = _number(route_lateral_m, 0.0)
        return {
            "active": True,
            "class_id": 4,
            "cycle": self.cycle,
            "path_id": "c04_online_staging_approach_v2",
            # This startup cue is also present before class-16's moving-flow
            # evidence becomes visible. Keep enough stopping distance while
            # that online ambiguity resolves; the construction bypass itself
            # still uses the normal controller below.
            "target_speed_mps": 3.0,
            "waypoints": [[0.0, 0.0], [14.0, 0.0], [self.APPROACH_DISTANCE_M, 0.0]],
            "lookahead_m": 3.5,
            "steer_kp": 0.9,
            "steer_kd": 0.04,
            "steer_limit": 0.35,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": True,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class04_staging_approach",
            "pid_step_rejoin": True,
            "radar_corridor_m": 0.0,
            "origin_route_lateral_m": origin,
            "hold_end_m": self.APPROACH_DISTANCE_M,
            "rejoin_end_m": self.APPROACH_DISTANCE_M,
            "stabilize_end_m": self.APPROACH_DISTANCE_M,
            "finish_lateral_tolerance_m": 0.35,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": self.APPROACH_MAX_FRAMES,
        }

    def _build_plan(self, route_lateral_m):
        origin = _number(route_lateral_m, 0.0)
        corridor = self._selected_corridor()
        obstacle = max(5.0, _number(self.entry_front_m, 27.5))
        shift_end = self.SHIFT_END_M
        hold_end = self.HOLD_END_M
        rejoin_end = self.REJOIN_END_M
        finish = self.FINISH_M
        target_y = corridor - origin
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y * 0.5 * (1.0 - math.cos(math.pi * fraction))
            waypoints.append([
                self.SHIFT_START_M
                + (shift_end - self.SHIFT_START_M) * fraction,
                y,
            ])
        waypoints.append([hold_end, target_y])
        for fraction in (0.25, 0.5, 0.75, 1.0):
            y = target_y + (-origin - target_y) * 0.5 * (
                1.0 - math.cos(math.pi * fraction)
            )
            waypoints.append([hold_end + (rejoin_end - hold_end) * fraction, y])
        waypoints.append([finish, -origin])
        return {
            "active": True,
            "class_id": 4,
            "cycle": self.cycle,
            "path_id": "c04_staged_left_clearance_v2",
            "waypoints": waypoints,
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.42,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": True,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class04_stop_plan_left",
            "pid_step_rejoin": True,
            "radar_corridor_m": corridor,
            "origin_route_lateral_m": origin,
            "obstacle_distance_m": obstacle,
            "shift_start_m": self.SHIFT_START_M,
            "shift_end_m": shift_end,
            "hold_end_m": hold_end,
            "rejoin_end_m": rejoin_end,
            "stabilize_end_m": finish,
            "finish_lateral_tolerance_m": 0.25,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": self.BYPASS_MAX_FRAMES,
        }


    @classmethod
    def _required_clearance_time_s(cls):
        """Predict stopped-start time until the ego has cleared oncoming lane."""
        speed = cls.CONFLICT_SPEED_MPS
        accel = cls.ENTRY_ACCEL_MPS2
        accel_time = speed / accel
        accel_distance = 0.5 * accel * accel_time * accel_time
        clearance_distance = cls.REJOIN_END_M
        cruise_distance = max(0.0, clearance_distance - accel_distance)
        return accel_time + cruise_distance / speed + cls.ENTRY_MARGIN_S

    @staticmethod
    def _track_speed(control, speed, progress, plan):
        control.reverse = False
        hold_end = _number(plan.get("hold_end_m"), 30.0)
        rejoin_end = _number(plan.get("rejoin_end_m"), hold_end + 12.0)
        explicit_target = plan.get("target_speed_mps")
        if explicit_target is not None:
            target = max(0.5, _number(explicit_target, 3.0))
        else:
            target = 8.5 if progress < hold_end else (7.0 if progress < rejoin_end else 5.5)
        error = target - speed
        if error < -1.0:
            control.throttle = 0.0
            control.brake = min(0.25, 0.08 + 0.07 * abs(error))
        else:
            control.throttle = max(0.12, min(1.0, 0.32 + 0.24 * error))
            control.brake = 0.0

    def apply(self, control, radar, features):
        approach = self.approach_gate.observe(radar)
        result, debug = radar_safety_action(
            control, radar, features, self.config, "ordinary_class_04_stop_plan_left",
            approach_gate=self.approach_gate, approach_state=approach,
            static_side_pass_active=self.phase == "BYPASS",
        )
        start_phase = self.phase
        self.phase_frames += 1
        speed = abs(_number(getattr(features, "ego_speed", 0.0)))
        front_min = _number(radar.get("front_min"), 0.0)
        front_close = _number(radar.get("front_close_count"), 0.0)
        dynamic = _number(radar.get("front_dynamic_count"), 0.0)
        closing = _number(radar.get("front_closing_count"), 0.0)
        packet = radar.get("_online_radar_packet")
        route_lateral = radar.get("_route_lateral_m")
        feedback = radar.get("_local_path_feedback") or {}
        now_s = _number(radar.get("_timestamp"), 0.0)
        oncoming_now = self._update_oncoming(packet, speed, now_s)
        self._update_flow_pass(oncoming_now)
        cancel = self._cancel_reason(features)
        override = "shared_radar_safety"
        radar_only_construction_warning = False
        construction_staging_side_vehicle = False
        centered_close_vehicle = self._centered_close_vehicle(features)
        self.construction_context_seen = bool(
            self.construction_context_seen
            or bool(radar.get("_class04_construction_context_seen", False))
            or self._construction_context(features)
        )

        static_obstacle = bool(
            not cancel and (
                bool(radar.get("_prelock_obstacle_hold", False))
                or (
                    0.0 < front_min <= self.OBSTACLE_MAX_M
                    and front_close >= 2.0
                    and dynamic <= 1.0
                    and closing <= 1.0
                )
            )
        )

        required_ttc = self._required_clearance_time_s()
        if self.phase == "CRUISE":
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
            reason = str(radar.get("_classifier_reason", "") or "")
            startup_cue = bool(
                not cancel
                and (
                    reason == "semantic_static_straight_obstacle_start"
                    or bool(radar.get("_class04_startup_cue_seen", False))
                )
            )
            if self.maneuver_complete:
                self.obstacle_frames = 0
            elif startup_cue:
                self.startup_warning_cue = True
                self.cycle += 1
                self.approach_plan = self._build_approach_plan(route_lateral)
                if self.construction_context_seen:
                    self.approach_plan["target_speed_mps"] = (
                        self.CONFIRMED_CONSTRUCTION_APPROACH_SPEED_MPS
                    )
                self.phase = "APPROACH"
                self.phase_frames = 0
                self.approach_progress_m = 0.0
                self._track_speed(result, speed, 0.0, self.approach_plan)
                override = "class04_online_staging_approach"
            elif radar.get("_prelock_obstacle_hold", False):
                self.obstacle_frames = self.OBSTACLE_ACQUIRE_FRAMES
            else:
                self.obstacle_frames = self.obstacle_frames + 1 if static_obstacle else 0
            if (
                self.phase == "CRUISE"
                and not self.maneuver_complete
                and self.cooldown_frames == 0
                and self.obstacle_frames >= self.OBSTACLE_ACQUIRE_FRAMES
            ):
                self.phase = "BRAKE"
                self.phase_frames = 0
                self.stop_frames = 0
                self.entry_front_m = front_min
                self.corridor_samples.clear()
                self.cycle += 1
                self._stop(result)
                override = "class04_obstacle_stop"
        elif self.phase == "APPROACH":
            progress = _number(feedback.get("x_m"), self.approach_progress_m)
            self.approach_progress_m = max(self.approach_progress_m, progress)
            if self.approach_plan and self.construction_context_seen:
                self.approach_plan["target_speed_mps"] = (
                    self.CONFIRMED_CONSTRUCTION_APPROACH_SPEED_MPS
                )
            # The Class04 warning area itself produces dense static radar
            # returns before the common 28 m staging point.  Those returns
            # are not a body in the ego corridor: the accepted construction
            # traces keep approaching, whereas a real vehicle is exposed by
            # the semantic immediate-hazard signal.  Do not let the generic
            # radar emergency shortcut the construction staging lifecycle.
            radar_only_construction_warning = bool(
                self.construction_context_seen
                and bool(debug.get("emergency", False))
                and cancel != "immediate_hazard"
            )
            construction_staging_side_vehicle = bool(
                self.construction_context_seen
                and (
                    progress > 3.0
                    or self.early_side_recovery_active
                )
                and cancel == "immediate_hazard"
                and not centered_close_vehicle
            )
            obstacle_stop_trigger = bool(
                (
                    cancel == "immediate_hazard"
                    and not construction_staging_side_vehicle
                )
                or (
                    bool(debug.get("emergency", False))
                    and not radar_only_construction_warning
                    and not construction_staging_side_vehicle
                )
            )
            if obstacle_stop_trigger:
                # The startup signature is shared by construction and the
                # right-turn merge scenario.  Once a physical vehicle hazard
                # appears, resuming the straight staging path after it clears
                # is unsafe and also prevents the stopped ego from producing
                # the motion evidence needed for a class-16 relock.  Make the
                # stop irreversible and continue through the existing online
                # left-clearance/gap planner from the current pose.
                self.phase = "BRAKE"
                self.phase_frames = 0
                self.stop_frames = 0
                self.entry_front_m = front_min
                self.corridor_samples.clear()
                self.approach_plan = None
                self.early_side_stop_pending = bool(
                    self.__class__ is Rule04
                    and self.construction_context_seen
                    and progress <= 3.0
                    and cancel == "immediate_hazard"
                    and not centered_close_vehicle
                )
                self._stop(result)
                override = "class04_staging_hazard_stop_plan"
            elif cancel and not construction_staging_side_vehicle:
                # Traffic controls and vulnerable road users are temporary
                # holds, not obstacles to route around. Preserve the staging
                # plan for those signals only.
                self._stop(result)
                self.phase_frames = max(0, self.phase_frames - 1)
                override = "class04_cancel_hold_plan"
            elif bool(feedback.get("done")) or progress >= self.APPROACH_DISTANCE_M:
                self.phase = "BRAKE"
                self.phase_frames = 0
                self.stop_frames = 0
                self.entry_front_m = front_min
                self.corridor_samples.clear()
                self._stop(result)
                override = "class04_stage_before_construction"
            elif self.phase_frames >= self.APPROACH_MAX_FRAMES:
                self.phase = "BRAKE"
                self.phase_frames = 0
                self.stop_frames = 0
                self.entry_front_m = front_min
                self.corridor_samples.clear()
                self._stop(result)
                override = "class04_approach_timeout_stop"
            else:
                self._track_speed(
                    result, speed, progress,
                    self.approach_plan or self._build_approach_plan(route_lateral),
                )
                override = "class04_online_staging_approach"
        elif cancel and not (
            cancel == "immediate_hazard"
            and (
                self.phase in {"BRAKE", "PLAN", "WAIT_GAP"}
                or (
                    self.__class__ is Rule04
                    and
                    self.phase == "BYPASS"
                    and not centered_close_vehicle
                )
            )
        ):
            self._stop(result)
            progress = _number(feedback.get("x_m"), self.last_progress_m)
            if self.phase == "BYPASS" and progress >= self.REJOIN_END_M:
                # A traffic-control stop can arrive just before the local
                # controller's strict heading/distance ``done`` predicate.
                # Once the ego has already cleared the hold envelope and the
                # rejoin segment, the construction manoeuvre is complete.
                # Latch it before yielding longitudinal control so residual
                # construction returns cannot start the same bypass again.
                self.phase = "CRUISE"
                self.phase_frames = 0
                self.plan = None
                self.maneuver_complete = True
                self.cooldown_frames = self.COOLDOWN_FRAMES
                override = "class04_cancel_after_rejoin_complete"
            else:
                # A temporary red light, stop sign or VRU hold is not a state
                # reset.  Keep the online plan and resume it after the safety
                # condition clears.  Freeze the phase timer while stopped.
                self.phase_frames = max(0, self.phase_frames - 1)
                override = "class04_cancel_hold_plan"
        elif self.phase == "BRAKE":
            if self.early_side_stop_pending and not centered_close_vehicle:
                # A Class05 route is no longer executing Rule04 on this tick.
                # Remaining here is therefore online evidence that the early
                # non-centred return belonged to the construction corridor.
                # Rebase the route-centred staging plan at the stopped pose;
                # do not reuse feedback from the discarded plan.
                self.early_side_stop_pending = False
                self.early_side_recovery_active = True
                self.cycle += 1
                self.approach_plan = self._build_approach_plan(route_lateral)
                self.approach_plan["target_speed_mps"] = (
                    self.CONFIRMED_CONSTRUCTION_APPROACH_SPEED_MPS
                )
                self.phase = "APPROACH"
                self.phase_frames = 0
                self.approach_progress_m = 0.0
                self._track_speed(result, speed, 0.0, self.approach_plan)
                override = "class04_resume_after_unowned_early_side_stop"
            else:
                self._stop(result)
                override = "class04_obstacle_stop"
                self._observe_left_corridor(packet, route_lateral)
                self.stop_frames = self.stop_frames + 1 if speed <= 0.12 else 0
                if self.stop_frames >= self.BRAKE_CONFIRM_FRAMES:
                    self.phase = "PLAN"
                    self.phase_frames = 0
                elif self.phase_frames >= self.BRAKE_TIMEOUT_FRAMES:
                    override = "class04_brake_timeout_stop"
        elif self.phase == "PLAN":
            self._stop(result)
            override = "class04_plan_hold"
            self._observe_left_corridor(packet, route_lateral)
            # Gap observation and local-path construction occur concurrently.
            # Do not throw away a measured large slot merely because the plan
            # becomes ready on the following frame.
            safe_far_target = bool(
                self.oncoming_ttc_s is not None
                and self.oncoming_ttc_s >= required_ttc
                and self.oncoming_pass_confirmed
            )
            if oncoming_now:
                self.oncoming_seen = True
                self.gap_clear_frames = 0
                self.entry_safe_frames = (
                    self.entry_safe_frames + 1 if safe_far_target else 0
                )
            elif self.oncoming_seen:
                self.gap_clear_frames = 0
                self.entry_safe_frames = 0
            else:
                self.gap_clear_frames += 1
                self.entry_safe_frames = 0
            if (
                self.phase_frames >= self.PLAN_MIN_GAP_OBSERVE_FRAMES
                and self.entry_safe_frames >= self.ENTRY_SAFE_CONFIRM_FRAMES
            ):
                self.plan = self._build_plan(route_lateral)
                self.phase = "BYPASS"
                self.phase_frames = 0
                self.last_progress_m = 0.0
                self.no_progress_frames = 0
                self._track_speed(result, speed, 0.0, self.plan)
                override = "class04_planned_measured_gap_bypass"
            elif self.phase_frames >= self.PLAN_OBSERVE_FRAMES:
                self.plan = self._build_plan(route_lateral)
                self.phase = "WAIT_GAP"
                self.phase_frames = 0
                self.gap_clear_frames = 0
                self.entry_safe_frames = 0
                override = "class04_gap_hold"
        elif self.phase == "WAIT_GAP":
            self._stop(result)
            override = "class04_gap_hold"
            self._observe_left_corridor(packet, route_lateral)
            if self.plan is None:
                self.plan = self._build_plan(route_lateral)
            safe_far_target = bool(
                self.oncoming_ttc_s is not None
                and self.oncoming_ttc_s >= required_ttc
                and self.oncoming_pass_confirmed
            )
            if oncoming_now:
                self.gap_clear_frames = 0
                self.entry_safe_frames = self.entry_safe_frames + 1 if safe_far_target else 0
            elif self.oncoming_seen:
                # Once flow exists, silence cannot prove the following actor
                # is far enough away. Wait for that actor to be measured.
                self.gap_clear_frames = 0
                self.entry_safe_frames = 0
            else:
                self.gap_clear_frames += 1
                self.entry_safe_frames = 0
            ready = bool(
                self.entry_safe_frames >= self.ENTRY_SAFE_CONFIRM_FRAMES
                or (
                    self.ALLOW_POST_PASS_SILENCE
                    and
                    self.oncoming_pass_confirmed
                    and self.post_pass_clear_eligible
                    and self.oncoming_pass_clear_frames >= self.GAP_CLEAR_FRAMES
                )
                or self.gap_clear_frames >= self.GAP_CLEAR_FRAMES
            )
            if ready:
                if self.plan is None:
                    self.plan = self._build_plan(route_lateral)
                self.phase = "BYPASS"
                self.phase_frames = 0
                self.last_progress_m = 0.0
                self.no_progress_frames = 0
                self._track_speed(result, speed, 0.0, self.plan)
                override = "class04_left_bypass"
            elif self.phase_frames >= self.GAP_MAX_FRAMES:
                self.phase_frames = self.GAP_MAX_FRAMES
                override = "class04_continuous_oncoming_hold"
        elif self.phase == "BYPASS":
            progress = _number(feedback.get("x_m"), self.last_progress_m)
            if progress > self.last_progress_m + 0.02:
                self.last_progress_m = progress
                self.no_progress_frames = 0
            else:
                self.no_progress_frames += 1
            if bool(feedback.get("done")):
                self.phase = "COOLDOWN"
                self.phase_frames = 0
                self.cooldown_frames = self.COOLDOWN_FRAMES
                self.plan = None
                self.maneuver_complete = True
                override = "class04_path_complete"
            elif self.no_progress_frames >= self.BYPASS_NO_PROGRESS_FRAMES or self.phase_frames >= self.BYPASS_MAX_FRAMES:
                self._stop(result)
                override = "class04_path_timeout_stop"
            elif (
                not bool(
                    (self.plan or {}).get("committed_ignore_late_oncoming")
                )
                and
                oncoming_now
                and progress < self.REJOIN_END_M
                and abs(_number(feedback.get("y_m"), 0.0)) < 0.35
                and _number(self.oncoming_ttc_s, 99.0) < self.CRITICAL_TTC_S
            ):
                self.phase = "WAIT_GAP"
                self.phase_frames = 0
                self.gap_clear_frames = 0
                self.entry_safe_frames = 0
                self._stop(result)
                override = "class04_late_oncoming_stop"
            else:
                self._track_speed(result, speed, progress, self.plan)
                override = "class04_left_bypass"
        elif self.phase == "COOLDOWN":
            self.cooldown_frames -= 1
            if self.cooldown_frames <= 0:
                self.phase = "CRUISE"
                self.phase_frames = 0

        debug.update({
            "class04_phase": self.phase,
            "class04_phase_frames": self.phase_frames,
            "class04_cycle": self.cycle,
            "class04_obstacle_frames": self.obstacle_frames,
            "class04_entry_front_m": self.entry_front_m,
            "class04_startup_warning_cue": self.startup_warning_cue,
            "class04_construction_context_seen": bool(
                self.construction_context_seen
            ),
            "class04_approach_progress_m": self.approach_progress_m,
            "class04_early_side_stop_pending": bool(
                self.early_side_stop_pending
            ),
            "class04_early_side_recovery_active": bool(
                self.early_side_recovery_active
            ),
            "class04_radar_only_construction_warning": bool(
                radar_only_construction_warning
            ),
            "class04_construction_staging_side_vehicle": bool(
                construction_staging_side_vehicle
            ),
            "class04_centered_close_vehicle": bool(centered_close_vehicle),
            "class04_maneuver_complete": self.maneuver_complete,
            "class04_corridor_samples": list(self.corridor_samples),
            "class04_selected_corridor_m": self._selected_corridor(),
            "class04_oncoming_seen": self.oncoming_seen,
            "class04_oncoming_observed_now": oncoming_now,
            "class04_oncoming_distance_m": self.oncoming_distance_m,
            "class04_oncoming_closing_mps": self.oncoming_closing_mps,
            "class04_oncoming_ttc_s": self.oncoming_ttc_s,
            "class04_gap_clear_frames": self.gap_clear_frames,
            "class04_entry_safe_frames": self.entry_safe_frames,
            "class04_oncoming_near_seen": self.oncoming_near_seen,
            "class04_oncoming_pass_clear_frames": self.oncoming_pass_clear_frames,
            "class04_oncoming_pass_confirmed": self.oncoming_pass_confirmed,
            "class04_post_pass_clear_eligible": self.post_pass_clear_eligible,
            "class04_required_clearance_time_s": required_ttc,
            "class04_local_feedback": feedback,
            "override_source": override,
        })
        if start_phase != self.phase:
            debug["class04_event"] = {"from": start_phase, "to": self.phase, "reason": override}
        if self.phase == "APPROACH" and self.approach_plan:
            debug["ordinary_local_path"] = dict(self.approach_plan)
            debug["state"] = "CLASS04_ONLINE_STAGING_APPROACH"
        elif self.phase == "BYPASS" and self.plan:
            debug["ordinary_local_path"] = dict(self.plan)
            debug["state"] = "CLASS04_LEFT_BYPASS"
        elif self.phase not in {"CRUISE", "COOLDOWN"}:
            debug["state"] = "CLASS04_{}".format(self.phase)
        debug["output_control"] = {
            "throttle": _number(getattr(result, "throttle", 0.0)),
            "brake": _number(getattr(result, "brake", 0.0)),
            "steer": _number(getattr(result, "steer", 0.0)),
            "reverse": bool(getattr(result, "reverse", False)),
        }
        return result, debug


def build_rule():
    return Rule04(get_config(4))
