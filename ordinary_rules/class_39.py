from .common import CategoryRule, _number
from .configs import get_config


class Rule39(CategoryRule):
    """Follow the live route through the two-flow interurban left turn."""

    def __init__(self, cfg):
        super(Rule39, self).__init__(cfg)
        self._route_entry_aligned_frames = 0
        self._route_entry_complete = False

    def reset(self):
        super(Rule39, self).reset()
        self._route_entry_aligned_frames = 0
        self._route_entry_complete = False

    def apply(self, control, radar, features):
        output, debug = super(Rule39, self).apply(control, radar, features)
        observation = radar.get("_route_waypoint_observation") or {}
        near = observation.get("near") or {}
        far = observation.get("far") or {}
        near_error = abs(_number(near.get("heading_error_rad"), 99.0))
        far_error = abs(_number(far.get("heading_error_rad"), 99.0))
        far_distance = _number(far.get("distance_m"), 0.0)
        aligned = bool(
            near_error <= _number(
                self.config.get("entry_near_heading_max_rad"), 0.60
            )
            and far_error <= _number(
                self.config.get("entry_far_heading_max_rad"), 0.20
            )
            and far_distance >= _number(
                self.config.get("entry_far_distance_min_m"), 20.0
            )
        )
        self._route_entry_aligned_frames = (
            self._route_entry_aligned_frames + 1 if aligned else 0
        )
        if self._route_entry_aligned_frames >= max(
            1, int(self.config.get("entry_align_confirm_frames", 3))
        ):
            self._route_entry_complete = True
        debug["class39_route_entry_aligned_frames"] = int(
            self._route_entry_aligned_frames
        )
        debug["class39_route_entry_complete"] = bool(
            self._route_entry_complete
        )
        debug["class39_route_entry_geometry"] = {
            "near_heading_error_rad": near_error,
            "far_heading_error_rad": far_error,
            "far_distance_m": far_distance,
        }
        # Let the learned controller negotiate the two live ActorFlows.  The
        # generic waypoint controller is useful only after the vehicle has
        # completed the incorporation and both live route horizons agree;
        # applying it at entry steered the ego directly into the first flow.
        if not self._route_entry_complete:
            debug["class39_control_phase"] = "MODEL_FLOW_ENTRY"
            return output, debug
        debug["class39_control_phase"] = "DENSE_ROUTE_REJOIN"
        debug["ordinary_dense_route_pid"] = {
            "active": True,
            "phase": "ordinary_class39_dense_route_rejoin",
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
        return output, debug


def build_rule():
    return Rule39(get_config(39))
