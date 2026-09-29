"""Class 35 category rule; runtime uses radar class lock and legal online navigation."""

import math

from .common import CategoryRule, _number, optional_feature_value
from .configs import get_config


class Rule35(CategoryRule):
    """Split early bends from deferred bends using live route geometry."""

    def __init__(self, cfg):
        super(Rule35, self).__init__(cfg)
        self.last_timestamp = None
        self.progress_m = 0.0
        self.green_commit_latched = False
        self.early_curve_frames = 0
        self.early_planner_latched = False
        self.early_reach_decided = False
        self.early_reach_latched = False
        self.early_reach_unknown_stop_frames = 0
        self.early_signal_view = None
        self.early_signal_target_xy = None
        self.early_signal_pending_view = None
        self.early_signal_pending_target_xy = None
        self.early_signal_red_frames = 0
        self.early_signal_green_frames = 0
        self.early_signal_direct_green_candidate = False
        self.early_signal_direct_green_view = None
        self.early_signal_direct_green_target_xy = None
        self.early_signal_direct_green_gap_frames = 0
        self.early_signal_red_armed = False
        self.early_signal_release_latched = False
        self.early_signal_release_progress_m = None
        self.early_post_release_stall_frames = 0
        self.early_post_release_recovery_latched = False
        self.early_post_release_recovery_done = False
        self.early_post_release_recovery_start_progress_m = None
        self.early_post_release_recovery_start_timestamp = None
        self.early_post_release_heading_clear_frames = 0
        self.deferred_initial_green = False
        self.deferred_signal_side = None
        self.deferred_red_frames = 0
        self.deferred_green_frames = 0
        self.deferred_red_armed = False
        self.deferred_release_latched = False
        self.deferred_signal_target_xy = None
        self.deferred_initial_green_seen = False
        self.deferred_side_green_frames = 0
        self.deferred_signal_rebound = False
        self.signal_infrastructure_absent_frames = 0
        self.signal_infrastructure_seen = False
        self.no_signal_passthrough_latched = False

    def reset(self):
        super(Rule35, self).reset()
        self.last_timestamp = None
        self.progress_m = 0.0
        self.green_commit_latched = False
        self.early_curve_frames = 0
        self.early_planner_latched = False
        self.early_reach_decided = False
        self.early_reach_latched = False
        self.early_reach_unknown_stop_frames = 0
        self.early_signal_view = None
        self.early_signal_target_xy = None
        self.early_signal_pending_view = None
        self.early_signal_pending_target_xy = None
        self.early_signal_red_frames = 0
        self.early_signal_green_frames = 0
        self.early_signal_direct_green_candidate = False
        self.early_signal_direct_green_view = None
        self.early_signal_direct_green_target_xy = None
        self.early_signal_direct_green_gap_frames = 0
        self.early_signal_red_armed = False
        self.early_signal_release_latched = False
        self.early_signal_release_progress_m = None
        self.early_post_release_stall_frames = 0
        self.early_post_release_recovery_latched = False
        self.early_post_release_recovery_done = False
        self.early_post_release_recovery_start_progress_m = None
        self.early_post_release_recovery_start_timestamp = None
        self.early_post_release_heading_clear_frames = 0
        self.deferred_initial_green = False
        self.deferred_signal_side = None
        self.deferred_red_frames = 0
        self.deferred_green_frames = 0
        self.deferred_red_armed = False
        self.deferred_release_latched = False
        self.deferred_signal_target_xy = None
        self.deferred_initial_green_seen = False
        self.deferred_side_green_frames = 0
        self.deferred_signal_rebound = False
        self.signal_infrastructure_absent_frames = 0
        self.signal_infrastructure_seen = False
        self.no_signal_passthrough_latched = False

    def _credible_signal_infrastructure(self, features):
        """Require an online signal prerequisite before owning an indefinite wait."""
        map_floor = _number(
            self.config.get("signal_infrastructure_map_min_score"), 0.35
        )
        detector_floor = _number(
            self.config.get("signal_infrastructure_detector_min_score"), 0.45
        )
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
                if class_name == "trafficlight" and score >= map_floor:
                    return True
            elif score >= detector_floor:
                # Non-map entries in traffic_light_candidates are outputs of
                # the dedicated signal detector and do not carry class_name.
                return True
        return False

    def _advance_progress(self, timestamp, speed):
        if self.last_timestamp is not None:
            dt = max(0.0, min(0.20, timestamp - self.last_timestamp))
            self.progress_m += max(0.0, speed) * dt
        self.last_timestamp = timestamp
        if self.progress_m >= _number(self.config.get("green_commit_progress_m"), 18.0):
            self.green_commit_latched = True

    def _green_window_steer(self, steer, speed):
        """Compensate route steering only when longitudinal override raises speed."""
        steer = _number(steer)
        start = _number(self.config.get("green_turn_steer_start_mps"), 6.0)
        if abs(steer) < 0.08 or speed <= start:
            return steer, 1.0
        gain = 1.0 + _number(
            self.config.get("green_turn_steer_gain_per_mps"), 0.08
        ) * (speed - start)
        limit = abs(_number(self.config.get("green_turn_steer_limit"), 0.80))
        return max(-limit, min(limit, steer * gain)), gain

    def _request_route_waypoint(self, debug, deferred=False):
        debug["ordinary_route_waypoint"] = {
            "active": True,
            "phase": "ordinary_class35_live_route",
            "near_switch_distance_m": 2.0,
            "steer_kp": _number(self.config.get(
                "deferred_waypoint_steer_kp" if deferred else "waypoint_steer_kp"
            ), 0.35 if deferred else 0.72),
            "model_steer_blend": _number(
                self.config.get(
                    "deferred_waypoint_model_steer_blend"
                    if deferred else "waypoint_model_steer_blend"
                ), 0.30 if deferred else 0.20
            ),
            "steer_limit": _number(self.config.get(
                "deferred_waypoint_steer_limit" if deferred else "waypoint_steer_limit"
            ), 0.42 if deferred else 0.80),
            "steer_rate_per_frame": _number(
                self.config.get(
                    "deferred_waypoint_steer_rate_per_frame"
                    if deferred else "waypoint_steer_rate_per_frame"
                ), 0.10 if deferred else 0.12
            ),
            "high_speed_start_mps": _number(
                self.config.get("waypoint_high_speed_start_mps"), 7.5
            ),
            "high_speed_steer_limit": _number(
                self.config.get("waypoint_high_speed_steer_limit"), 0.28
            ),
        }

    def _observe_early_route_curve(self, radar):
        observation = radar.get("_route_waypoint_observation") or {}
        near = observation.get("near") or {}
        error = near.get("heading_error_rad")
        threshold = _number(self.config.get("early_curve_signed_max_rad"), -0.10)
        minimum = _number(self.config.get("early_curve_progress_min_m"), 0.0)
        window = _number(self.config.get("early_curve_progress_window_m"), 1.5)
        if (
            not self.early_planner_latched
            and minimum <= self.progress_m <= window
        ):
            if error is not None and _number(error) <= threshold:
                self.early_curve_frames += 1
            else:
                self.early_curve_frames = 0
            if self.early_curve_frames >= int(self.config.get("early_curve_confirm_frames", 3)):
                self.early_planner_latched = True
        elif self.progress_m < minimum:
            # At standstill the nearest navigation point can alternate across
            # the ego origin and briefly invert its signed heading.  Do not
            # carry that pre-motion noise into the topology decision.
            self.early_curve_frames = 0
        return error

    def _observe_deferred_signal(self, features, near_heading_error):
        decision_progress = _number(
            self.config.get("deferred_decision_progress_m"), 1.5
        )
        if (
            not self.early_planner_latched
            and not self.deferred_initial_green
            and self.progress_m > decision_progress
        ):
            self.deferred_initial_green = True
            # Positive live near-point error means the governing turn head is
            # on the left of the image; negative means right.  This is online
            # route geometry, never a route identifier or XML lookup.
            self.deferred_signal_side = (
                "left" if _number(near_heading_error, 0.0) >= 0.0 else "right"
            )
        votes = getattr(features, "traffic_light_region_votes", {}) or {}
        vote = votes.get(self.deferred_signal_side or "", {}) or {}
        raw_color = str(vote.get("color", "unknown"))
        radius = _number(self.config.get("signal_component_track_radius_px"), 20.0)

        def components(name):
            result = []
            for item in vote.get(name + "_components", []) or []:
                try:
                    xy = item.get("centroid")
                    area = int(item.get("area", 0))
                    point = (float(xy[0]), float(xy[1]))
                except (AttributeError, TypeError, ValueError, IndexError):
                    continue
                if area >= 12:
                    result.append((area, point))
            return result

        greens = components("green")
        reds = components("red")
        if self.deferred_initial_green and self.deferred_signal_target_xy is None and greens:
            self.deferred_signal_target_xy = greens[0][1]

        def nearest(items):
            if self.deferred_signal_target_xy is None or not items:
                return None
            tx, ty = self.deferred_signal_target_xy
            return min(items, key=lambda item: math.hypot(item[1][0] - tx, item[1][1] - ty))

        tracked_color = "unknown"
        tracked = None
        red = nearest(reds)
        green = nearest(greens)
        red_distance = None if red is None else math.hypot(
            red[1][0] - self.deferred_signal_target_xy[0],
            red[1][1] - self.deferred_signal_target_xy[1],
        )
        green_distance = None if green is None else math.hypot(
            green[1][0] - self.deferred_signal_target_xy[0],
            green[1][1] - self.deferred_signal_target_xy[1],
        )
        green_same_vertical_head = False
        if green is not None and self.deferred_signal_target_xy is not None:
            green_dx = abs(green[1][0] - self.deferred_signal_target_xy[0])
            green_dy = green[1][1] - self.deferred_signal_target_xy[1]
            green_same_vertical_head = bool(
                self.deferred_signal_rebound
                and green_dx <= _number(
                    self.config.get("signal_component_track_x_radius_px"), 8.0
                )
                and 0.0 <= green_dy <= _number(
                    self.config.get("signal_component_track_down_px"), 28.0
                )
            )
        if red_distance is not None and red_distance <= radius:
            tracked_color, tracked = "red", red
        elif (
            green_distance is not None
            and (green_distance <= radius or green_same_vertical_head)
        ):
            tracked_color, tracked = "green", green
        staging_ready = bool(
            self.deferred_initial_green
            and self.progress_m >= _number(
                self.config.get("deferred_red_arm_min_progress_m"), 5.5
            )
            and _number(getattr(features, "ego_speed", 0.0)) <= 0.5
        )
        if tracked_color == "green" and not staging_ready:
            self.deferred_initial_green_seen = True

        # The turn head used during the approach can leave its original image
        # region before the vehicle settles.  At the stationary line-safe
        # staging point, bind exactly once to the strongest red component in
        # the complete *front* view.  This is the only view for which the
        # formal route-25975 audit observed TL121's stable red-to-green phase;
        # side-camera greens belong to cross traffic and are diagnostic only.
        if (
            staging_ready
            and not self.deferred_red_armed
            and not self.deferred_signal_rebound
            and bool(self.config.get("deferred_front_red_rebind_enabled", False))
            and tracked_color != "red"
        ):
            red_candidates = []
            for region_name, region_vote in votes.items():
                for item in region_vote.get("red_components", []) or []:
                    try:
                        xy = item.get("centroid")
                        area = int(item.get("area", 0))
                        point = (float(xy[0]), float(xy[1]))
                    except (AttributeError, TypeError, ValueError, IndexError):
                        continue
                    if area >= 12:
                        red_candidates.append((area, region_name, point, region_vote))
            if red_candidates:
                # Preserve the online route-geometry ownership decision.  A
                # larger red blob in another region can be a vehicle lamp or
                # reflection (the route-25968 failure bound a bottom-centre
                # blob while the left governing head changed red->green).
                # Only fall back across the full front image when the selected
                # region truly has no red head, as observed on route 25975.
                owned_candidates = [
                    item for item in red_candidates
                    if item[1] == self.deferred_signal_side
                ]
                area, region_name, point, region_vote = max(
                    owned_candidates or red_candidates,
                    key=lambda item: item[0],
                )
                self.deferred_signal_side = region_name
                self.deferred_signal_target_xy = point
                self.deferred_signal_rebound = True
                vote = region_vote
                raw_color = str(region_vote.get("color", "red"))
                tracked_color = "red"
                tracked = (area, point)
        side_camera_name = (
            "CAM_FRONT_LEFT"
            if self.deferred_signal_side == "left"
            else "CAM_FRONT_RIGHT"
        )
        camera_votes = getattr(features, "traffic_light_camera_votes", {}) or {}
        side_camera_vote = camera_votes.get(side_camera_name, {}) or {}
        side_green_components = []
        for region_vote in (side_camera_vote.get("region_votes", {}) or {}).values():
            for item in region_vote.get("green_components", []) or []:
                try:
                    area = int(item.get("area", 0))
                except (AttributeError, TypeError, ValueError):
                    continue
                if area >= 12:
                    side_green_components.append(item)
        if (
            staging_ready
            and self.deferred_initial_green_seen
            and bool(self.config.get("deferred_side_green_release_enabled", True))
        ):
            self.deferred_side_green_frames = (
                self.deferred_side_green_frames + 1
                if side_green_components
                else 0
            )
            if self.deferred_side_green_frames >= int(
                self.config.get("signal_phase_confirm_frames", 2)
            ):
                self.deferred_release_latched = True
                tracked_color = "green"
                raw_color = str(side_camera_vote.get("observed", "green"))

        # Follow the target while the vehicle is still settling before the
        # red phase.  Once red is armed the camera is stationary: freeze the
        # identity anchor so adjacent red heads cannot create a slow centroid
        # walk toward an unrelated green lamp over a long signal cycle.
        if tracked is not None and not self.deferred_red_armed:
            self.deferred_signal_target_xy = tracked[1]
        confirm = int(self.config.get("signal_phase_confirm_frames", 2))
        if (
            self.deferred_initial_green
            and not self.deferred_release_latched
            and not self.deferred_red_armed
            and staging_ready
        ):
            self.deferred_red_frames = self.deferred_red_frames + 1 if tracked_color == "red" else 0
            if self.deferred_red_frames >= confirm:
                self.deferred_red_armed = True
        elif self.deferred_red_armed and not self.deferred_release_latched:
            self.deferred_green_frames = self.deferred_green_frames + 1 if tracked_color == "green" else 0
            if self.deferred_green_frames >= confirm:
                self.deferred_release_latched = True
        return tracked_color, raw_color, vote

    def _observe_early_signal(
        self, features, speed, near_heading_error, model_brake
    ):
        """Bind the stopped early-turn signal and require its red-to-green phase."""
        tracked_color = "unknown"
        if not self.early_planner_latched or self.deferred_initial_green:
            return tracked_color
        if (
            self.green_commit_latched
            and self.early_signal_target_xy is None
            and not self.early_signal_red_armed
        ):
            # The early-turn control zone has already been traversed. A red
            # head first seen after commit belongs to a downstream junction,
            # not to the encounter this category is responsible for. Keep an
            # identity acquired before commit alive through Red-to-Green, but
            # never acquire a new identity after commit.
            return tracked_color

        votes = getattr(features, "traffic_light_region_votes", {}) or {}
        camera_votes = getattr(features, "traffic_light_camera_votes", {}) or {}

        def vote_for_view(view):
            if view == "FRONT_CENTER":
                return votes.get("center", {}) or {}
            if view == "FRONT_LEFT":
                return votes.get("left", {}) or {}
            if view == "FRONT_RIGHT":
                return votes.get("right", {}) or {}
            camera = camera_votes.get(view or "", {}) or {}
            return (camera.get("region_votes", {}) or {}).get("center", {}) or {}

        center = vote_for_view(self.early_signal_view or "FRONT_CENTER")

        def components(vote, name, minimum_area):
            items = []
            for item in vote.get(name + "_components", []) or []:
                try:
                    area = int(item.get("area", 0))
                    xy = item.get("centroid")
                    point = (float(xy[0]), float(xy[1]))
                except (AttributeError, TypeError, ValueError, IndexError):
                    continue
                if area >= minimum_area:
                    items.append((area, point))
            return items

        red_minimum = int(self.config.get("early_signal_red_min_area", 5))
        green_minimum = int(self.config.get("early_signal_green_min_area", 10))
        reds = components(
            center, "red", red_minimum
        )
        greens = components(
            center, "green", green_minimum
        )
        can_bind = bool(
            self.progress_m >= _number(
                self.config.get("early_signal_bind_min_progress_m"), 3.0
            )
            and speed <= _number(
                self.config.get("early_signal_bind_max_speed_mps"), 0.5
            )
        )
        identity_radius = _number(
            self.config.get("early_signal_red_track_radius_px"), 12.0
        )
        principal_x = _number(
            self.config.get("early_signal_principal_x_px"), 320.0
        )
        principal_radius = _number(
            self.config.get("early_signal_principal_radius_px"), 80.0
        )
        center_dim_reds = [
            item for item in components(
                center, "dim_red",
                int(self.config.get("early_signal_dim_min_area", 5)),
            )
            if item[0] <= int(
                self.config.get("early_signal_dim_max_area", 100)
            )
            and abs(item[1][0] - principal_x) <= principal_radius
        ]
        # Dim-only blobs are eligible to acquire braking ownership only after
        # the bounded reach state has proved that the learned model is stuck
        # before a fogged/occluded governing head.  During natural model
        # passthrough they are reflections, not sufficient signal identity:
        # route 14842 otherwise bound a drifting dim blob at 14.44 m and
        # overrode 600 consecutive model go commands until AgentBlocked.
        eligible_dim_reds = (
            center_dim_reds if self.early_reach_latched else []
        )
        direct_green_ready = bool(
            self.progress_m >= _number(
                self.config.get("early_curve_progress_min_m"), 1.5
            )
            and speed <= _number(
                self.config.get("early_direct_green_max_speed_mps"), 0.75
            )
            and (
                self.early_signal_direct_green_candidate
                or model_brake >= _number(
                    self.config.get("early_direct_green_min_model_brake"), 0.5
                )
            )
        )
        if (
            bool(self.config.get("early_direct_green_enabled", False))
            and not self.early_signal_red_armed
            and direct_green_ready
        ):
            # Once a side green has become a temporal candidate, keep
            # observing that exact view.  Tiny centre red specks otherwise
            # alternate with the valid side component and reset identity on
            # every other frame (the v36 closed-loop failure).
            direct_view = (
                self.early_signal_direct_green_view
                or self.early_signal_view
            )
            direct_vote = (
                vote_for_view(direct_view) if direct_view else center
            )
            direct_greens = components(
                direct_vote, "green", green_minimum
            )
            if direct_view is None:
                # A valid centre red always wins over a side green: it is the
                # previously verified governing head for two early routes.
                # Only when the centre contains no red component may the live
                # signed turn direction select one main-front side region.
                if not reds and not direct_greens:
                    direct_view = (
                        "FRONT_LEFT"
                        if _number(near_heading_error, 0.0) >= 0.0
                        else "FRONT_RIGHT"
                    )
                    direct_vote = vote_for_view(direct_view)
                    direct_greens = components(
                        direct_vote, "green", green_minimum
                    )
                elif direct_greens:
                    direct_view = "FRONT_CENTER"
            if (
                (not reds or self.early_signal_direct_green_candidate)
                and direct_view
                and direct_greens
            ):
                selected_green = max(
                    direct_greens, key=lambda item: item[0]
                )
                if (
                    self.early_signal_direct_green_target_xy is None
                    or self.early_signal_direct_green_view != direct_view
                ):
                    self.early_signal_green_frames = 1
                else:
                    tx, ty = self.early_signal_direct_green_target_xy
                    distance = math.hypot(
                        selected_green[1][0] - tx,
                        selected_green[1][1] - ty,
                    )
                    self.early_signal_green_frames = (
                        self.early_signal_green_frames + 1
                        if distance <= identity_radius else 1
                    )
                self.early_signal_direct_green_view = direct_view
                self.early_signal_direct_green_target_xy = selected_green[1]
                self.early_signal_direct_green_gap_frames = 0
                self.early_signal_direct_green_candidate = True
                tracked_color = "green"
                if self.early_signal_green_frames >= int(
                    self.config.get("early_signal_confirm_frames", 2)
                ):
                    # A tentative green must never become the braking target:
                    # doing so changes the vehicle pose and can make the next
                    # camera sample disappear.  Publish the control target only
                    # after temporally stable evidence has authorized release.
                    self.early_signal_view = direct_view
                    self.early_signal_target_xy = selected_green[1]
                    self.early_signal_release_latched = True
                    self.early_signal_release_progress_m = self.progress_m
                return tracked_color
            if self.early_signal_direct_green_candidate:
                # The live adapter can omit one interleaved frame while the
                # same physical lamp remains spatially stable.  Preserve the
                # tentative evidence for one gap, but keep reaching rather
                # than braking.  Two gaps invalidate the candidate.
                self.early_signal_direct_green_gap_frames += 1
                if self.early_signal_direct_green_gap_frames > int(
                    self.config.get("early_direct_green_max_gap_frames", 1)
                ):
                    self.early_signal_green_frames = 0
                    self.early_signal_direct_green_candidate = False
                    self.early_signal_direct_green_view = None
                    self.early_signal_direct_green_target_xy = None
                    self.early_signal_direct_green_gap_frames = 0

        candidate_view = self.early_signal_view or "FRONT_CENTER"
        if (
            not self.early_signal_red_armed
            and can_bind
            and not reds
            and not eligible_dim_reds
            and self.early_signal_view is None
            and self.early_reach_latched
            and bool(self.config.get("early_side_signal_fallback_enabled", True))
        ):
            # Some early-turn approaches stop while the governing head is
            # outside the centre crop.  Select the side front camera from the
            # live signed route geometry, then freeze that camera identity;
            # never search all cameras for any convenient green.
            candidate_view = (
                "CAM_FRONT_LEFT"
                if _number(near_heading_error, 0.0) >= 0.0
                else "CAM_FRONT_RIGHT"
            )
            center = vote_for_view(candidate_view)
            reds = components(center, "red", red_minimum)
            greens = components(center, "green", green_minimum)
        if (
            not self.early_signal_red_armed
            and can_bind
            and (reds or eligible_dim_reds)
        ):
            # The negative early turn leaves its governing overhead head at
            # the right edge of the legal front-centre crop.  Selecting the
            # rightmost *component* rejects the much larger red building
            # reflections in the left crop while remaining route-agnostic.
            if candidate_view == "FRONT_CENTER":
                strict_near = [
                    item for item in reds
                    if abs(item[1][0] - principal_x) <= principal_radius
                ]
                # A strict central lamp remains authoritative.  When fog has
                # desaturated it below the strict ratio, select the compact
                # low-contrast component nearest the optical principal point;
                # this rejects side-facing cross-traffic heads.
                pool = strict_near or eligible_dim_reds or reds
                selected = min(
                    pool, key=lambda item: abs(item[1][0] - principal_x)
                )
            else:
                # A side view can contain several downstream signal heads.
                # Keep the upper lamp among the three strongest components;
                # this rejects tiny reflections while selecting the overhead
                # head whose green lens appears below the frozen red lens.
                strongest = sorted(reds, key=lambda item: item[0], reverse=True)[:3]
                selected = min(strongest, key=lambda item: item[1][1])
            if (
                self.early_signal_pending_target_xy is None
                or self.early_signal_pending_view != candidate_view
            ):
                self.early_signal_red_frames = 1
            else:
                tx, ty = self.early_signal_pending_target_xy
                distance = math.hypot(selected[1][0] - tx, selected[1][1] - ty)
                self.early_signal_red_frames = (
                    self.early_signal_red_frames + 1
                    if distance <= identity_radius else 1
                )
            self.early_signal_pending_target_xy = selected[1]
            self.early_signal_pending_view = candidate_view
            tracked_color = "red"
            if self.early_signal_red_frames >= int(
                self.config.get("early_signal_confirm_frames", 2)
            ):
                # A tentative colour blob cannot own longitudinal control.
                # Publish and freeze the official identity only after the
                # same candidate has survived temporal confirmation.
                self.early_signal_target_xy = selected[1]
                self.early_signal_view = candidate_view
                self.early_signal_red_armed = True
        elif not self.early_signal_red_armed and can_bind:
            # Confirmation is consecutive.  An observation gap terminates the
            # tentative identity instead of allowing two unrelated red blobs
            # to be joined across time.
            self.early_signal_red_frames = 0
            self.early_signal_pending_view = None
            self.early_signal_pending_target_xy = None

        if self.early_signal_red_armed and not self.early_signal_release_latched:
            tracked_color = "red"
            center = vote_for_view(self.early_signal_view)
            greens = components(center, "green", green_minimum)
            if self.early_signal_view == "FRONT_CENTER":
                dim_greens = [
                    item for item in components(
                        center, "dim_green",
                        int(self.config.get("early_signal_dim_min_area", 5)),
                    )
                    if item[0] <= int(
                        self.config.get("early_signal_dim_max_area", 100)
                    )
                ]
                greens = greens + dim_greens
            target = self.early_signal_target_xy
            matching_green = None
            if target is not None:
                tx, ty = target
                candidates = []
                for item in greens:
                    dx = abs(item[1][0] - tx)
                    dy = item[1][1] - ty
                    if (
                        dx <= _number(
                            self.config.get("early_signal_track_x_radius_px"), 8.0
                        )
                        and 0.0 <= dy <= _number(
                            self.config.get("early_signal_track_down_px"), 28.0
                        )
                    ):
                        candidates.append(item)
                if candidates:
                    matching_green = max(candidates, key=lambda item: item[0])
            self.early_signal_green_frames = (
                self.early_signal_green_frames + 1 if matching_green else 0
            )
            if self.early_signal_green_frames >= int(
                self.config.get("early_signal_confirm_frames", 2)
            ):
                self.early_signal_release_latched = True
                self.early_signal_release_progress_m = self.progress_m
                tracked_color = "green"
        elif self.early_signal_release_latched:
            tracked_color = "green"
        return tracked_color

    @staticmethod
    def _hazard(features, radar):
        pedestrian = optional_feature_value(features, "advanced_lidar_front_vru_distance")
        vehicle = optional_feature_value(features, "advanced_lidar_front_vehicle_distance")
        front_min = _number(radar.get("front_min"), 1e9)
        dynamic = _number(radar.get("front_dynamic_count"), 0.0)
        ttc = _number(radar.get("front_ttc"), 0.0)
        return bool(
            (pedestrian is not None and pedestrian <= 12.0)
            or (vehicle is not None and vehicle <= 7.0)
            or (front_min <= 6.0 and dynamic >= 2.0 and 0.0 < ttc <= 2.5)
        )

    def apply(self, control, radar, features):
        result, debug = super(Rule35, self).apply(control, radar, features)
        signal_present = self._credible_signal_infrastructure(features)
        signal_prerequisite_window = bool(
            not self.signal_infrastructure_seen
            and self.progress_m <= _number(
                self.config.get("deferred_decision_progress_m"), 3.0
            )
            and not self.early_signal_red_armed
            and not self.early_signal_release_latched
            and not self.deferred_red_armed
            and not self.deferred_release_latched
        )
        if signal_present:
            self.signal_infrastructure_seen = True
            self.signal_infrastructure_absent_frames = 0
        elif signal_prerequisite_window and not self.no_signal_passthrough_latched:
            self.signal_infrastructure_absent_frames += 1
            if self.signal_infrastructure_absent_frames >= int(
                self.config.get("signal_infrastructure_absent_confirm_frames", 20)
            ):
                self.no_signal_passthrough_latched = True
        debug["signal_infrastructure_present"] = signal_present
        debug["signal_infrastructure_seen"] = self.signal_infrastructure_seen
        debug["signal_prerequisite_window"] = signal_prerequisite_window
        debug["signal_infrastructure_absent_frames"] = (
            self.signal_infrastructure_absent_frames
        )
        debug["no_signal_passthrough_latched"] = (
            self.no_signal_passthrough_latched
        )
        if self.no_signal_passthrough_latched:
            debug["class35_mode"] = "NO_SIGNAL_INFRASTRUCTURE_PASSTHROUGH"
            return result, debug
        timestamp = _number(radar.get("_timestamp", 1e9), 1e9)
        speed = _number(getattr(features, "ego_speed", radar.get("ego_speed", 0.0)))
        self._advance_progress(timestamp, speed)
        near_heading_error = self._observe_early_route_curve(radar)
        route_shadow = radar.get("_route_waypoint_observation") or {}
        deferred_color, deferred_raw_color, deferred_vote = self._observe_deferred_signal(
            features, near_heading_error
        )
        early_signal_color = self._observe_early_signal(
            features, speed, near_heading_error,
            _number(getattr(control, "brake", 0.0)),
        )
        model_brake = _number(getattr(control, "brake", 0.0))
        model_throttle = _number(getattr(control, "throttle", 0.0))
        visible_signal_color = str(
            getattr(features, "traffic_light_color", "unknown") or "unknown"
        ).lower()
        bind_progress = _number(
            self.config.get("early_signal_bind_min_progress_m"), 3.0
        )
        if self.early_planner_latched and not self.early_reach_decided:
            # Control ownership is a temporal reachability decision, never a
            # snapshot of one noisy model frame.  A visible signal means the
            # model may be legally waiting for its phase (the 14842 subtype),
            # while sustained standstill + braking intent + no decisive signal is
            # the camera-reachability deadlock observed on 14909.  Active
            # motion or non-hard braking protects the naturally progressing
            # 14862 subtype when it is still moving.  Once sufficient evidence exists, keep the
            # bounded reach latched so ownership cannot oscillate.
            if self.progress_m >= bind_progress or self.early_signal_target_xy is not None:
                self.early_reach_decided = True
                self.early_reach_latched = False
                self.early_reach_unknown_stop_frames = 0
            elif visible_signal_color in ("red", "green"):
                self.early_reach_unknown_stop_frames = 0
            elif (
                speed <= _number(
                    self.config.get("early_reach_stop_max_speed_mps"), 0.10
                )
                and model_brake > 0.0
                and model_throttle <= 0.0
            ):
                self.early_reach_unknown_stop_frames += 1
                if self.early_reach_unknown_stop_frames >= int(
                    self.config.get("early_reach_unknown_stop_confirm_frames", 3)
                ):
                    self.early_reach_decided = True
                    self.early_reach_latched = True
            else:
                self.early_reach_unknown_stop_frames = 0
        early_launch_end = _number(
            self.early_signal_release_progress_m, self.progress_m
        ) + _number(self.config.get("early_green_launch_distance_m"), 4.0)
        post_release_recovery_distance = 0.0
        post_release_recovery_elapsed = 0.0
        if (
            self.early_signal_release_latched
            and self.progress_m >= early_launch_end
            and not self.early_post_release_recovery_latched
            and not self.early_post_release_recovery_done
        ):
            post_release_stalled = bool(
                speed <= _number(
                    self.config.get("early_post_release_stall_max_speed_mps"), 0.10
                )
                and model_brake > 0.0
                and model_throttle <= 0.0
                and abs(_number(near_heading_error, 0.0)) >= _number(
                    self.config.get("early_post_release_min_heading_error_rad"), 1.0
                )
            )
            self.early_post_release_stall_frames = (
                self.early_post_release_stall_frames + 1
                if post_release_stalled else 0
            )
            if self.early_post_release_stall_frames >= int(
                self.config.get("early_post_release_stall_confirm_frames", 20)
            ):
                self.early_post_release_recovery_latched = True
                self.early_post_release_recovery_start_progress_m = self.progress_m
                self.early_post_release_recovery_start_timestamp = timestamp
        if self.early_post_release_recovery_latched:
            post_release_recovery_distance = self.progress_m - _number(
                self.early_post_release_recovery_start_progress_m, self.progress_m
            )
            post_release_recovery_elapsed = timestamp - _number(
                self.early_post_release_recovery_start_timestamp, timestamp
            )
            heading_clear = bool(
                abs(_number(near_heading_error, 0.0)) <= _number(
                    self.config.get("early_post_release_clear_heading_rad"), 0.35
                )
                and post_release_recovery_distance >= _number(
                    self.config.get("early_post_release_min_recovery_distance_m"), 3.0
                )
            )
            self.early_post_release_heading_clear_frames = (
                self.early_post_release_heading_clear_frames + 1
                if heading_clear else 0
            )
            if (
                self.early_post_release_heading_clear_frames >= int(
                    self.config.get("early_post_release_clear_confirm_frames", 3)
                )
                or post_release_recovery_distance >= _number(
                    self.config.get("early_post_release_max_recovery_distance_m"), 8.0
                )
                or post_release_recovery_elapsed >= _number(
                    self.config.get("early_post_release_recovery_timeout_s"), 12.0
                )
            ):
                self.early_post_release_recovery_latched = False
                self.early_post_release_recovery_done = True
        green_deadline = _number(self.config.get("initial_green_deadline_s"), 5.0)
        commit_guard = _number(self.config.get("green_commit_guard_s"), 4.85)
        hazard = self._hazard(features, radar)
        debug.update({
            "green_deadline_s": green_deadline,
            "green_commit_guard_s": commit_guard,
            "green_progress_m": self.progress_m,
            "green_commit_latched": self.green_commit_latched,
            "class35_semantic_hazard": hazard,
            "near_heading_error_rad": near_heading_error,
            "early_route_shadow": {
                "observation": route_shadow,
                "route_lateral_m": radar.get("_route_lateral_m"),
                "model_steer": _number(getattr(control, "steer", 0.0)),
                "progress_m": self.progress_m,
            },
            "early_curve_frames": self.early_curve_frames,
            "early_planner_latched": self.early_planner_latched,
            "early_reach_decided": self.early_reach_decided,
            "early_reach_latched": self.early_reach_latched,
            "early_reach_unknown_stop_frames": self.early_reach_unknown_stop_frames,
            "early_reach_visible_signal_color": visible_signal_color,
            "early_signal_color": early_signal_color,
            "early_signal_view": self.early_signal_view,
            "early_signal_target_xy": self.early_signal_target_xy,
            "early_signal_pending_view": self.early_signal_pending_view,
            "early_signal_pending_target_xy": self.early_signal_pending_target_xy,
            "early_signal_red_frames": self.early_signal_red_frames,
            "early_signal_green_frames": self.early_signal_green_frames,
            "early_signal_direct_green_candidate": (
                self.early_signal_direct_green_candidate
            ),
            "early_signal_direct_green_view": self.early_signal_direct_green_view,
            "early_signal_direct_green_target_xy": (
                self.early_signal_direct_green_target_xy
            ),
            "early_signal_direct_green_gap_frames": (
                self.early_signal_direct_green_gap_frames
            ),
            "early_signal_red_armed": self.early_signal_red_armed,
            "early_signal_release_latched": self.early_signal_release_latched,
            "early_signal_release_progress_m": self.early_signal_release_progress_m,
            "early_signal_bind_window_closed": bool(
                self.green_commit_latched
                and self.early_signal_target_xy is None
                and not self.early_signal_red_armed
            ),
            "early_post_release_stall_frames": self.early_post_release_stall_frames,
            "early_post_release_recovery_latched": (
                self.early_post_release_recovery_latched
            ),
            "early_post_release_recovery_done": self.early_post_release_recovery_done,
            "early_post_release_recovery_distance_m": post_release_recovery_distance,
            "early_post_release_recovery_elapsed_s": post_release_recovery_elapsed,
            "early_post_release_heading_clear_frames": (
                self.early_post_release_heading_clear_frames
            ),
            "deferred_initial_green": self.deferred_initial_green,
            "deferred_initial_green_seen": self.deferred_initial_green_seen,
            "deferred_signal_side": self.deferred_signal_side,
            "deferred_signal_color": deferred_color,
            "deferred_signal_raw_color": deferred_raw_color,
            "deferred_signal_target_xy": self.deferred_signal_target_xy,
            "deferred_signal_vote": deferred_vote,
            "deferred_red_frames": self.deferred_red_frames,
            "deferred_green_frames": self.deferred_green_frames,
            "deferred_side_green_frames": self.deferred_side_green_frames,
            "deferred_signal_rebound": self.deferred_signal_rebound,
            "deferred_red_armed": self.deferred_red_armed,
            "deferred_staging_ready": bool(
                self.progress_m >= _number(
                    self.config.get("deferred_red_arm_min_progress_m"), 5.5
                ) and speed <= 0.5
            ),
            "deferred_release_latched": self.deferred_release_latched,
            "waypoint_activation_progress_m": _number(
                self.config.get("waypoint_activation_progress_m"), 10.0
            ),
        })

        if hazard:
            result.throttle, result.brake = 0.0, 1.0
            debug["state"] = "CLASS35_SEMANTIC_HAZARD_STOP"
            debug["source"] = "ordinary_class_35"
        elif (
            self.early_planner_latched
            and self.early_reach_latched
            and self.early_signal_target_xy is None
            and self.progress_m < _number(
                self.config.get("early_signal_bind_min_progress_m"), 3.0
            )
        ):
            # Do not hand back to a model that has already stopped before the
            # signal-binding state is observable.  Continue the same bounded
            # geometry probe only to the pre-existing 3 m safety boundary.
            target = _number(
                self.config.get("geometry_probe_target_speed_mps"), 1.5
            )
            result.brake = 0.0
            result.throttle = (
                _number(self.config.get("geometry_probe_throttle"), 0.25)
                if speed < target else 0.0
            )
            debug["geometry_probe_target_speed_mps"] = target
            debug["state"] = "CLASS35_EARLY_REACH_SIGNAL_BIND"
            debug["source"] = "ordinary_class_35"
        elif (
            self.early_planner_latched
            and self.early_signal_target_xy is not None
            and not self.early_signal_release_latched
        ):
            # A red component has been bound at a line-safe standstill.  Do
            # not treat colour dropout as permission to enter the junction.
            result.throttle, result.brake = 0.0, 1.0
            debug["state"] = "CLASS35_EARLY_WAIT_SAME_HEAD_GREEN"
            debug["source"] = "ordinary_class_35"
        elif (
            self.early_planner_latched
            and self.early_signal_release_latched
            and self.progress_m < (
                _number(self.early_signal_release_progress_m, self.progress_m)
                + _number(self.config.get("early_green_launch_distance_m"), 4.0)
            )
        ):
            # The learned steering already completed both recorded early
            # geometries without lane departure.  Supply only enough bounded
            # longitudinal motion to leave the stop line during the verified
            # green phase, then return the whole control loop to the model.
            target = _number(
                self.config.get("early_green_launch_target_speed_mps"), 3.0
            )
            result.brake = 0.0
            result.throttle = (
                _number(self.config.get("early_green_launch_throttle"), 0.65)
                if speed < target else 0.0
            )
            debug["early_green_launch_target_speed_mps"] = target
            debug["state"] = "CLASS35_EARLY_GREEN_BOUNDED_LAUNCH"
            debug["source"] = "ordinary_class_35"
        elif self.early_post_release_recovery_latched:
            # A legal green release has already happened, but the learned
            # controller can occasionally stop forever midway through the
            # turn.  The observed failure is longitudinal (model brake with
            # no throttle), while successful controls retain useful steering.
            # Recover only longitudinal motion and leave lateral ownership
            # with the model; the former near-waypoint override pursued a
            # target behind the vehicle and caused a 41.46% lane departure.
            # Distance/time bounds still return full ownership to the model,
            # and semantic hazards remain higher priority.
            target = _number(
                self.config.get("early_post_release_recovery_target_speed_mps"), 1.5
            )
            result.brake = 0.0
            result.throttle = (
                _number(
                    self.config.get("early_post_release_recovery_throttle"), 0.25
                ) if speed < target else 0.0
            )
            debug["early_post_release_recovery_target_speed_mps"] = target
            debug["state"] = "CLASS35_EARLY_POST_RELEASE_WAYPOINT_RECOVERY"
            debug["source"] = "ordinary_class_35"
        elif self.deferred_initial_green and not self.deferred_release_latched:
            stage_brake = _number(
                self.config.get("deferred_staging_brake_progress_m"), 5.7
            )
            if not self.deferred_red_armed and self.progress_m < stage_brake:
                stage_speed = _number(
                    self.config.get("deferred_staging_target_speed_mps"), 1.5
                )
                result.brake = 0.0
                result.throttle = 0.25 if speed < stage_speed else 0.0
                debug["state"] = "CLASS35_DEFERRED_STAGE_BEFORE_LINE"
            else:
                result.throttle, result.brake = 0.0, 1.0
                debug["state"] = (
                    "CLASS35_DEFERRED_WAIT_NEXT_GREEN"
                    if self.deferred_red_armed
                    else "CLASS35_DEFERRED_WAIT_RED_ARM"
                )
            debug["source"] = "ordinary_class_35"
        elif self.early_planner_latched and not self.deferred_initial_green:
            # The original controller completes the early negative-signed
            # turns without an infraction when allowed to respect its own
            # signal wait and longitudinal braking.  Once that live geometry
            # is confirmed, do not replace either its steering or speed.
            debug["state"] = "CLASS35_EARLY_CURVE_MODEL_PASSTHROUGH"
            debug["source"] = "ordinary_class_35"
        elif not self.deferred_initial_green:
            # Before the turn topology is known, crawl only far enough for the
            # live near waypoint to expose its signed curvature.  The former
            # full-throttle launch fought a model brake and caused 24.38% lane
            # departure on route14842.  This bounded probe is shared by the
            # whole category and never reads route identity or offline XML.
            probe_target = _number(
                self.config.get("geometry_probe_target_speed_mps"), 1.5
            )
            result.brake = 0.0
            result.throttle = (
                _number(self.config.get("geometry_probe_throttle"), 0.25)
                if speed < probe_target else 0.0
            )
            debug["geometry_probe_target_speed_mps"] = probe_target
            debug["state"] = "CLASS35_GEOMETRY_PROBE"
            debug["source"] = "ordinary_class_35"
        elif (
            not self.deferred_release_latched
            and not self.green_commit_latched
            and timestamp >= green_deadline
        ):
            result.throttle, result.brake = 0.0, 1.0
            debug["state"] = "CLASS35_MISSED_GREEN_SAFE_HOLD"
            debug["source"] = "ordinary_class_35"
        else:
            target = _number(
                self.config.get(
                    "deferred_launch_target_speed_mps"
                    if self.deferred_release_latched
                    else "route_follow_target_speed_mps"
                ),
                7.0 if self.deferred_release_latched else 6.5,
            )
            clear_progress = _number(
                self.config.get("intersection_clear_progress_m"), 36.0
            )
            clearing_intersection = (
                not self.deferred_release_latched and self.progress_m < clear_progress
            )
            debug["intersection_clear_progress_m"] = clear_progress
            debug["clearing_intersection"] = clearing_intersection
            result.brake = 0.0
            tail_clear_progress = _number(
                self.config.get("deferred_green_tail_clear_progress_m"), 13.0
            )
            debug["deferred_green_tail_clear_progress_m"] = tail_clear_progress
            if self.deferred_release_latched and self.progress_m < tail_clear_progress:
                result.throttle = 1.0
            elif clearing_intersection:
                result.throttle = 0.25 if speed < 5.0 else 0.0
            elif speed < target - 0.8:
                result.throttle = min(
                    _number(self.config.get("route_follow_throttle_limit"), 0.65),
                    max(0.25, 0.18 + 0.12 * (target - speed)),
                )
            elif speed > target + 1.0:
                result.throttle, result.brake = 0.0, _number(
                    self.config.get("post_commit_overspeed_brake"), 0.80
                )
            else:
                result.throttle = 0.0
            if self.progress_m >= debug["waypoint_activation_progress_m"]:
                self._request_route_waypoint(
                    debug, deferred=self.deferred_release_latched
                )
            elif self.deferred_release_latched:
                debug["deferred_waypoint_planner_pending"] = True
            debug["state"] = (
                "CLASS35_CLEAR_INTERSECTION" if clearing_intersection else
                ("CLASS35_DEFERRED_GREEN_RELEASE" if self.deferred_release_latched else
                 "CLASS35_COMMITTED_ROUTE_FOLLOW")
            )
            debug["source"] = "ordinary_class_35"

        debug["output_control"] = {
            "throttle": _number(getattr(result, "throttle", 0.0)),
            "steer": _number(getattr(result, "steer", 0.0)),
            "brake": _number(getattr(result, "brake", 0.0)),
        }
        return result, debug


def build_rule():
    return Rule35(get_config(35))
