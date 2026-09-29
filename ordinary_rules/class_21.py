"""Class-21 one-way construction bypass with explicit path ownership."""

from __future__ import absolute_import

from .class_04 import Rule04, _number
from .configs import get_config


class Rule21(Rule04):
    """Stop, plan, enter the adjacent lane, and rejoin after construction."""

    PLAN_OBSERVE_FRAMES = 8
    PLAN_MIN_GAP_OBSERVE_FRAMES = 8
    GAP_CLEAR_FRAMES = 3
    CONFLICT_SPEED_MPS = 7.5

    def __init__(self, cfg):
        self._ignore_static_immediate = False
        super(Rule21, self).__init__(cfg)

    def reset(self):
        super(Rule21, self).reset()
        self._ignore_static_immediate = False

    def _cancel_reason(self, features):
        reason = super(Rule21, self)._cancel_reason(features)
        if reason == "immediate_hazard" and self._ignore_static_immediate:
            return ""
        return reason

    def _build_plan(self, route_lateral_m):
        plan = super(Rule21, self)._build_plan(route_lateral_m)
        plan.update({
            "class_id": 21,
            "path_id": "c21_one_way_construction_bypass_v1",
            "route_pid_phase": "ordinary_class21_stop_plan_rejoin",
            # The construction may sit immediately before a bend.  Keep the
            # path in the live navigation frame instead of letting the
            # stop-pose tangent accumulate metres of lateral drift.
            "trust_live_route_lateral": True,
            "finish_use_live_route_lateral": True,
            "finish_use_live_route_heading": True,
            "committed_ignore_late_oncoming": True,
        })
        return plan

    def apply(self, control, radar, features):
        handoff = str(radar.get("_classifier_reason", "") or "") == (
            "online_construction_no_opposing_flow"
        )
        if self.phase == "CRUISE" and handoff and not self.maneuver_complete:
            self.phase = "BRAKE"
            self.phase_frames = 0
            self.stop_frames = 0
            self.entry_front_m = _number(radar.get("front_min"), 0.0)
            self.corridor_samples.clear()
            self.cycle += 1

        self._ignore_static_immediate = bool(
            self.phase == "BYPASS"
            and float(radar.get("front_dynamic_count", 0.0) or 0.0) <= 1.0
            and float(radar.get("front_closing_count", 0.0) or 0.0) <= 1.0
        )
        output, debug = super(Rule21, self).apply(control, radar, features)
        debug.update({
            "class21_phase": self.phase,
            "class21_online_handoff": handoff,
            "class21_committed_static_hazard_suppressed": bool(
                self._ignore_static_immediate
                and bool(getattr(features, "immediate_hazard", False))
            ),
        })
        if str(debug.get("state", "")).startswith("CLASS04_"):
            debug["state"] = "CLASS21_" + str(debug["state"])[8:]
        if str(debug.get("override_source", "")).startswith("class04_"):
            debug["override_source"] = "class21_" + str(
                debug["override_source"]
            )[8:]
        return output, debug


def build_rule():
    return Rule21(get_config(21))
