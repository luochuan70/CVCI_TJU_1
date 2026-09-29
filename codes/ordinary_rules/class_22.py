"""Class-22 one-way accident bypass."""

from __future__ import absolute_import

from .class_21 import Rule21
from .configs import get_config


class Rule22(Rule21):
    """Approach from the warning cue, stop near the accident, then bypass."""

    # The warning is at the scenario trigger; the police car is 40--44 m past
    # it and the two accident cars are another 10 m and 6 m downstream.  Stop
    # close enough to execute the lane change, retain the adjacent lane until
    # the measured second-car envelope is behind the ego, then merge back.
    APPROACH_DISTANCE_M = 42.0
    SHIFT_END_M = 6.5
    HOLD_END_M = 32.0
    REJOIN_END_M = 39.0
    FINISH_M = 44.0

    def _build_approach_plan(self, route_lateral_m):
        plan = super(Rule22, self)._build_approach_plan(route_lateral_m)
        plan.update({
            "class_id": 22,
            "path_id": "c22_online_accident_approach_v1",
            "route_pid_phase": "ordinary_class22_accident_approach",
            "use_dense_route_pid": True,
            "dense_route_pid_phase": "ordinary_class22_dense_approach",
            "trust_live_route_lateral": True,
            "finish_use_live_route_lateral": True,
            "finish_use_live_route_heading": True,
        })
        return plan

    def _build_plan(self, route_lateral_m):
        plan = super(Rule22, self)._build_plan(route_lateral_m)
        plan.update({
            "class_id": 22,
            "path_id": "c22_dense_full_train_bypass_v4",
            "route_pid_phase": "ordinary_class22_stop_plan_rejoin",
            "use_dense_route_pid": True,
            "dense_route_pid_phase": "ordinary_class22_dense_bypass",
        })
        return plan

    def apply(self, control, radar, features):
        handoff = str(radar.get("_classifier_reason", "") or "") == (
            "online_accident_warning_without_construction_geometry"
        )
        if self.phase == "CRUISE" and handoff and not self.maneuver_complete:
            self.cycle += 1
            self.approach_plan = self._build_approach_plan(
                radar.get("_route_lateral_m")
            )
            self.phase = "APPROACH"
            self.phase_frames = 0
            self.approach_progress_m = 0.0

        output, debug = super(Rule22, self).apply(control, radar, features)
        debug["class22_phase"] = self.phase
        debug["class22_warning_handoff"] = handoff
        if str(debug.get("state", "")).startswith("CLASS21_"):
            debug["state"] = "CLASS22_" + str(debug["state"])[8:]
        override = str(debug.get("override_source", "") or "")
        if override.startswith("class21_"):
            debug["override_source"] = "class22_" + override[8:]
        return output, debug


def build_rule():
    return Rule22(get_config(22))
