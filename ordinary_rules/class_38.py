from .class_16 import Rule16
from .common import clone_control, _number
from .configs import get_config


class Rule38(Rule16):
    """Highway-exit controller that matches a live rear traffic stream.

    The official scenario requires acceleration into a continuous highway
    flow before following the online route out of it. Lateral control always
    remains with the learned route follower; this class only prevents an
    approaching rear-quarter vehicle from catching an unnecessarily braking
    ego when there is adequate room to the vehicle ahead.
    """

    def reset(self):
        super(Rule38, self).reset()
        self.class38_flow_target_mps = 0.0
        self.class38_flow_target_timestamp = None
        self.class38_exit_turn_frames = 0
        self.class38_exit_turn_latched = False
        self.class38_exit_latch_reason = ""
        self.class38_exit_candidate_source = ""
        self.class38_dense_window_frames = 0
        self.class38_dense_started = False

    def apply(self, control, radar, features):
        result, debug = super(Rule38, self).apply(control, radar, features)
        debug["source"] = "ordinary_class_38_highway_exit_flow"
        timestamp = _number(radar.get("_timestamp"), 0.0)
        ego_speed = _number(debug.get("ego_speed"), 0.0)
        flow_count = _number(radar.get("same_direction_flow_count"), 0.0)
        observed_flow_speed = _number(
            radar.get("same_direction_flow_speed_median_mps"), 0.0
        )
        if (
            flow_count >= _number(
                self.config.get("class38_flow_target_min_points"), 2.0
            )
            and _number(
                self.config.get("class38_flow_target_min_mps"), 8.0
            ) <= observed_flow_speed <= _number(
                self.config.get("class38_flow_target_max_mps"), 16.5
            )
        ):
            self.class38_flow_target_mps = observed_flow_speed
            self.class38_flow_target_timestamp = timestamp

        flow_target_age_s = (
            timestamp - self.class38_flow_target_timestamp
            if self.class38_flow_target_timestamp is not None
            else float("inf")
        )
        retained_target = (
            self.class38_flow_target_mps
            if flow_target_age_s <= _number(
                self.config.get("class38_flow_target_hold_s"), 20.0
            ) else 0.0
        )
        observation = radar.get("_route_waypoint_observation") or {}
        far_route = observation.get("far") or {}
        far_heading_signed = _number(
            far_route.get("heading_error_rad"), 0.0
        )
        far_heading_error = abs(far_heading_signed)
        far_route_distance = _number(far_route.get("distance_m"), 0.0)
        strict_exit_turn_candidate = bool(
            retained_target > 0.0
            and _number(
                self.config.get("class38_exit_far_distance_min_m"), 8.0
            ) <= far_route_distance <= _number(
                self.config.get("class38_exit_far_distance_max_m"), 40.0
            )
            and far_heading_error >= _number(
                self.config.get("class38_exit_far_heading_min_rad"), 0.40
            )
        )
        # A member with a dense highway corridor can already be travelling at
        # flow speed when the learned follower drifts across the short near
        # chord.  Its ego-relative bearing fluctuates below the strict exit
        # angle even though immediate route correction is needed.  Require a
        # second, independent radar-environment signature instead of weakening
        # the strict exit gate used by the sparse members.
        dense_corridor_recovery_candidate = bool(
            retained_target > 0.0
            and ego_speed >= _number(
                self.config.get("class38_corridor_min_ego_speed_mps"), 10.0
            )
            and _number(radar.get("point_count"), 0.0) >= _number(
                self.config.get("class38_corridor_min_radar_points"), 50.0
            )
            and _number(
                self.config.get("class38_corridor_distance_min_m"), 8.0
            ) <= far_route_distance <= _number(
                self.config.get("class38_corridor_distance_max_m"), 15.0
            )
            and far_heading_error >= _number(
                self.config.get("class38_corridor_heading_min_rad"), 0.25
            )
        )
        exit_turn_candidate = bool(
            strict_exit_turn_candidate
            or dense_corridor_recovery_candidate
        )
        if exit_turn_candidate:
            if self.class38_exit_turn_frames == 0:
                self.class38_exit_candidate_source = (
                    "dense_corridor_recovery"
                    if dense_corridor_recovery_candidate else "strict_exit"
                )
            elif dense_corridor_recovery_candidate:
                self.class38_exit_candidate_source = "dense_corridor_recovery"
            self.class38_exit_turn_frames += 1
        else:
            self.class38_exit_turn_frames = 0
            self.class38_exit_candidate_source = ""
        if (
            not self.class38_exit_turn_latched
            and self.class38_exit_turn_frames >= max(
                1, int(self.config.get("class38_exit_confirm_frames", 2))
            )
        ):
            self.class38_exit_turn_latched = True
            self.class38_exit_latch_reason = (
                self.class38_exit_candidate_source or "strict_exit"
            )
            # Window evidence collected before the semantic prelock belongs
            # to the feeder approach, not the confirmed exit.  Start the
            # strict settling clock at the lock transition itself.
            self.class38_dense_window_frames = 0
        rear = debug.get("rear_side_pressure") or {}
        rear_present = bool(
            rear.get("active", False)
            or rear.get("lifecycle_observed", False)
        )
        front = debug.get("front_tracked_vehicle") or {}
        front_distance = _number(front.get("distance_m"), 0.0)
        required_front_room = max(
            _number(self.config.get("class38_min_front_room_m"), 10.0),
            _number(self.config.get("class38_front_headway_s"), 0.65)
            * ego_speed,
        )
        front_room_ready = bool(
            not front.get("active", False)
            or front_distance >= required_front_room
        )
        safety_owns = bool(
            debug.get("state") == "TRAFFIC_HOLD"
            or bool(getattr(features, "immediate_hazard", False))
            or bool(getattr(features, "red_light_active", False))
            or getattr(features, "front_pedestrian_distance", None) is not None
        )

        active_merge_lifecycle = bool(
            debug.get("longitudinal_merge_latched", False)
            or debug.get("merge_gap_wait_active", False)
            or debug.get("merge_gap_committed", False)
            or debug.get("merge_entry_follow_active", False)
            or debug.get("merge_entry_follow_lifecycle_observed", False)
        )
        if (
            not rear_present
            and not active_merge_lifecycle
            and debug.get("state") in {
                "FLOW_MATCH", "FLOW_SPEED_RECOVERY", "FLOW_SPEED_CAP",
                "FLOW_SPEED_BRAKE", "FLOW_REAR_SIDE_SPEED_TRACK",
            }
        ):
            result = clone_control(control)
            debug["state"] = "CLASS38_MODEL_ROUTE_FOLLOW"

        if rear_present and not front_room_ready and not safety_owns:
            # The rear stream does not authorize accelerating into a close
            # lead.  Restore the learned headway/braking decision that the
            # generic Class16 FLOW_MATCH branch would otherwise cancel.
            result = clone_control(control)
            debug["state"] = "CLASS38_FRONT_ROOM_MODEL_HOLD"

        speed_error = retained_target - ego_speed
        rear_flow_boost = bool(
            rear_present
            and retained_target > 0.0
            and front_room_ready
            and not safety_owns
            and speed_error >= _number(
                self.config.get("class38_flow_boost_min_error_mps"), 0.5
            )
        )
        if rear_flow_boost:
            result = clone_control(control)
            result.throttle = max(
                _number(getattr(result, "throttle", 0.0)),
                min(
                    1.0,
                    _number(
                        self.config.get("class38_flow_boost_base"), 0.78
                    )
                    + _number(
                        self.config.get("class38_flow_boost_gain"), 0.10
                    ) * max(0.0, speed_error),
                ),
            )
            result.brake = 0.0
            debug["state"] = "CLASS38_REAR_FLOW_SPEED_MATCH"

        # The learned controller can negotiate the live highway stream, but
        # the long straight approach in one member leaves the exit turn until
        # it is physically too late.  Once the legal online route horizon
        # shows a sustained, substantial turn after live flow was observed,
        # reuse the dense CARLA waypoint PID already validated by Class39.
        # This directive changes only lateral control; longitudinal safety and
        # the measured-flow speed policy above remain authoritative.
        # Route-horizon resampling can move the far waypoint back outside the
        # activation window after the vehicle has already reached it.  The
        # physical entry event is therefore one-way for the remainder of the
        # route: do not hand lateral control back and forth as the sampled
        # waypoint distance changes.
        inside_dense_window = bool(
            far_route_distance > 0.0
            and far_route_distance <= _number(
                self.config.get(
                    "class38_dense_activation_distance_max_m"), 28.0
            )
        )
        if inside_dense_window and not self.class38_dense_started:
            self.class38_dense_window_frames += 1
        elif not self.class38_dense_started:
            self.class38_dense_window_frames = 0
        required_window_frames = (
            1 if self.class38_exit_latch_reason == "dense_corridor_recovery"
            else max(1, int(self.config.get(
                "class38_strict_window_confirm_frames", 3
            )))
        )
        if (
            self.class38_exit_turn_latched
            and inside_dense_window
            and self.class38_dense_window_frames >= required_window_frames
        ):
            self.class38_dense_started = True
        dense_exit_window_ready = bool(self.class38_dense_started)
        if dense_exit_window_ready:
            debug["ordinary_dense_route_pid"] = {
                "active": True,
                "phase": "ordinary_class38_highway_exit_turn",
                "target_route_offset_m": 0.0,
                "lookahead_base_m": _number(
                    self.config.get("class38_dense_lookahead_base_m"), 5.0
                ),
                "lookahead_speed_s": _number(
                    self.config.get("class38_dense_lookahead_speed_s"), 0.40
                ),
                "lookahead_min_m": _number(
                    self.config.get("class38_dense_lookahead_min_m"), 4.5
                ),
                "lookahead_max_m": _number(
                    self.config.get("class38_dense_lookahead_max_m"), 8.0
                ),
                "steer_limit": _number(
                    self.config.get("class38_dense_steer_limit"), 0.40
                ),
                "steer_rate_per_frame": _number(
                    self.config.get("class38_dense_steer_rate"), 0.08
                ),
                "pose_filter_frames": int(
                    self.config.get("class38_dense_pose_filter_frames", 5)
                ),
            }

        debug["class38_observed_flow_speed_mps"] = float(observed_flow_speed)
        debug["class38_retained_flow_target_mps"] = float(retained_target)
        debug["class38_flow_target_age_s"] = (
            float(flow_target_age_s)
            if flow_target_age_s != float("inf") else None
        )
        debug["class38_rear_present"] = bool(rear_present)
        debug["class38_front_room_ready"] = bool(front_room_ready)
        debug["class38_required_front_room_m"] = float(required_front_room)
        debug["class38_rear_flow_boost"] = bool(rear_flow_boost)
        debug["class38_exit_turn_candidate"] = bool(exit_turn_candidate)
        debug["class38_strict_exit_turn_candidate"] = bool(
            strict_exit_turn_candidate
        )
        debug["class38_dense_corridor_recovery_candidate"] = bool(
            dense_corridor_recovery_candidate
        )
        debug["class38_exit_turn_frames"] = int(self.class38_exit_turn_frames)
        debug["class38_exit_turn_latched"] = bool(
            self.class38_exit_turn_latched
        )
        debug["class38_exit_latch_reason"] = str(
            self.class38_exit_latch_reason
        )
        debug["class38_dense_window_frames"] = int(
            self.class38_dense_window_frames
        )
        debug["class38_dense_required_window_frames"] = int(
            required_window_frames
        )
        debug["class38_dense_exit_window_ready"] = bool(
            dense_exit_window_ready
        )
        debug["class38_exit_far_heading_error_rad"] = float(
            far_heading_error
        )
        debug["class38_exit_far_heading_signed_rad"] = float(
            far_heading_signed
        )
        debug["class38_exit_far_distance_m"] = float(far_route_distance)
        debug["class38_route_turn_observation"] = (
            radar.get("_route_turn_observation") or {}
        )
        debug["output_control"] = {
            "throttle": _number(getattr(result, "throttle", 0.0)),
            "steer": _number(getattr(result, "steer", 0.0)),
            "brake": _number(getattr(result, "brake", 0.0)),
        }
        return result, debug


def build_rule():
    return Rule38(get_config(38))
