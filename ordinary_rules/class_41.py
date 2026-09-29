from .common import CategoryRule, _number
from .configs import get_config


class Rule41(CategoryRule):
    """Cross one interurban flow with the model, then rejoin the live route."""

    def __init__(self, cfg):
        super(Rule41, self).__init__(cfg)
        self._turn_seen = False
        self._turn_seen_frames = 0
        self._rejoin_aligned_frames = 0
        self._rejoin_active = False
        self._cross_track_frames = 0
        self._cross_track_clear_frames = 0
        self._cross_track_active = False

    def reset(self):
        super(Rule41, self).reset()
        self._turn_seen = False
        self._turn_seen_frames = 0
        self._rejoin_aligned_frames = 0
        self._rejoin_active = False
        self._cross_track_frames = 0
        self._cross_track_clear_frames = 0
        self._cross_track_active = False

    def apply(self, control, radar, features):
        output, debug = super(Rule41, self).apply(control, radar, features)
        observation = radar.get("_route_waypoint_observation") or {}
        near = observation.get("near") or {}
        far = observation.get("far") or {}
        near_error = abs(_number(near.get("heading_error_rad"), 99.0))
        far_error = abs(_number(far.get("heading_error_rad"), 99.0))
        far_distance = _number(far.get("distance_m"), 0.0)
        route_lateral = _number(radar.get("_route_lateral_m"), 99.0)
        timestamp = _number(radar.get("_timestamp"), 0.0)

        # The route is initially aligned, so alignment by itself cannot prove
        # progress.  Transfer lateral control only after two live frames show
        # that the vehicle has physically entered the category's common turn.
        turning_now = bool(
            timestamp >= _number(self.config.get("turn_seen_min_time_s"), 3.0)
            and max(near_error, far_error) >= _number(
                self.config.get("turn_seen_heading_min_rad"), 0.30
            )
            and far_distance >= _number(
                self.config.get("turn_seen_far_distance_min_m"), 12.0
            )
        )
        self._turn_seen_frames = self._turn_seen_frames + 1 if turning_now else 0
        if self._turn_seen_frames >= max(
            1, int(self.config.get("turn_seen_confirm_frames", 2))
        ):
            self._turn_seen = True
        aligned = bool(
            self._turn_seen
            and near_error <= _number(
                self.config.get("rejoin_near_heading_max_rad"), 0.22
            )
            and far_error <= _number(
                self.config.get("rejoin_far_heading_max_rad"), 0.22
            )
            and far_distance >= _number(
                self.config.get("rejoin_far_distance_min_m"), 12.0
            )
        )
        self._rejoin_aligned_frames = (
            self._rejoin_aligned_frames + 1 if aligned else 0
        )
        if self._rejoin_aligned_frames >= max(
            1, int(self.config.get("rejoin_align_confirm_frames", 3))
        ):
            self._rejoin_active = True

        lateral_valid = abs(route_lateral) < 20.0
        cross_track = bool(
            self._turn_seen
            and lateral_valid
            and abs(route_lateral) >= _number(
                self.config.get("cross_track_enter_m"), 3.0
            )
            # A large lateral value during the normal tight bend is not by
            # itself a departure.  Route 23901 repeatedly completes at 100
            # while the near chord is sharp but the far route remains aligned
            # (about 0.16 rad).  The failed 23910 trajectory instead misses the
            # whole turn: its far error is already about 0.60 rad at entry.
            # Only the latter needs the discontinuous signed-waypoint recovery.
            and far_error >= _number(
                self.config.get("cross_track_far_heading_min_rad"), 0.35
            )
        )
        if not self._cross_track_active:
            self._cross_track_frames = (
                self._cross_track_frames + 1 if cross_track else 0
            )
            if self._cross_track_frames >= max(
                1, int(self.config.get("cross_track_confirm_frames", 2))
            ):
                self._cross_track_active = True
                self._cross_track_clear_frames = 0
        else:
            cross_track_clear = bool(
                lateral_valid
                and abs(route_lateral) <= _number(
                    self.config.get("cross_track_exit_m"), 1.5
                )
                and near_error <= _number(
                    self.config.get("cross_track_exit_near_rad"), 0.35
                )
                and far_error <= _number(
                    self.config.get("cross_track_exit_far_rad"), 0.25
                )
            )
            self._cross_track_clear_frames = (
                self._cross_track_clear_frames + 1
                if cross_track_clear else 0
            )
            if self._cross_track_clear_frames >= max(
                1, int(self.config.get("cross_track_exit_frames", 3))
            ):
                self._cross_track_active = False
                self._cross_track_frames = 0

        debug["class41_turn_seen"] = bool(self._turn_seen)
        debug["class41_turn_seen_frames"] = int(self._turn_seen_frames)
        debug["class41_rejoin_aligned_frames"] = int(
            self._rejoin_aligned_frames
        )
        debug["class41_rejoin_active"] = bool(self._rejoin_active)
        debug["class41_cross_track_frames"] = int(self._cross_track_frames)
        debug["class41_cross_track_clear_frames"] = int(
            self._cross_track_clear_frames
        )
        debug["class41_cross_track_active"] = bool(
            self._cross_track_active
        )
        debug["class41_route_geometry"] = {
            "near_heading_error_rad": near_error,
            "far_heading_error_rad": far_error,
            "far_distance_m": far_distance,
            "route_lateral_m": route_lateral,
        }
        # Formal route23910 telemetry disproved the aligned-only handoff: the
        # learned controller had already drifted 15--22 m from the online route
        # before alignment could ever occur, then became trapped against a
        # static boundary.  Once two live frames prove that the actual turn has
        # started, the dense online route is the correct geometric reference;
        # radar safety remains responsible for longitudinal flow clearance.
        if not self._turn_seen:
            debug["class41_control_phase"] = "MODEL_SINGLE_FLOW_CROSSING"
            return output, debug

        if self._cross_track_active:
            debug["class41_control_phase"] = "SIGNED_WAYPOINT_RECOVERY"
            debug["ordinary_route_waypoint"] = {
                "active": True,
                "phase": "ordinary_class41_signed_waypoint_recovery",
                "steer_kp": _number(
                    self.config.get("waypoint_recovery_steer_kp"), 0.72
                ),
                "model_steer_blend": _number(
                    self.config.get("waypoint_recovery_model_blend"), 0.0
                ),
                "steer_limit": _number(
                    self.config.get("waypoint_recovery_steer_limit"), 0.55
                ),
                "high_speed_start_mps": _number(
                    self.config.get("waypoint_recovery_high_speed_mps"), 7.5
                ),
                "high_speed_steer_limit": _number(
                    self.config.get("waypoint_recovery_high_speed_limit"), 0.38
                ),
                "steer_rate_per_frame": _number(
                    self.config.get("waypoint_recovery_steer_rate"), 0.10
                ),
                "prefer_lower_error_far": True,
                "far_lifecycle_max_distance_m": 60.0,
                "far_lifecycle_error_margin_rad": 0.10,
            }
            return output, debug
        elif self._rejoin_active:
            debug["class41_control_phase"] = "DENSE_ROUTE_REJOIN"
        else:
            debug["class41_control_phase"] = "DENSE_TURN_AND_REJOIN"
        directive = {
            "active": True,
            "phase": "ordinary_class41_dense_turn_and_rejoin",
            "target_route_offset_m": 0.0,
            "lookahead_base_m": _number(
                self.config.get("dense_pid_lookahead_base_m"), 5.0
            ),
            "lookahead_speed_s": _number(
                self.config.get("dense_pid_lookahead_speed_s"), 0.40
            ),
            "lookahead_min_m": _number(
                self.config.get("dense_pid_lookahead_min_m"), 4.5
            ),
            "lookahead_max_m": _number(
                self.config.get("dense_pid_lookahead_max_m"), 8.0
            ),
            "steer_limit": _number(
                self.config.get("dense_pid_steer_limit"), 0.40
            ),
            "steer_rate_per_frame": _number(
                self.config.get("dense_pid_steer_rate"), 0.08
            ),
            "pose_filter_frames": int(
                self.config.get("dense_pid_pose_filter_frames", 5)
            ),
        }
        debug["ordinary_dense_route_pid"] = directive
        return output, debug


def build_rule():
    return Rule41(get_config(41))
