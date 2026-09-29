"""Ordinary-only model agent whose gate never reads evaluator XML truth."""

from __future__ import absolute_import

import os

try:
    from radar_model_switch_agent import RadarModelSwitchAgent
    from ordinary_online_gate import OrdinaryOnlineGate
except ImportError:
    from .radar_model_switch_agent import RadarModelSwitchAgent
    from .ordinary_online_gate import OrdinaryOnlineGate


class OrdinaryOnlineModelAgent(RadarModelSwitchAgent):
    def _build_scene_gate(self):
        # Parent setup calls this factory directly, so the XML-backed gate is
        # never instantiated on an ordinary formal run.
        return OrdinaryOnlineGate(
            output_path=os.environ.get("CVCI_SCENE_GATE_LOG_PATH", "")
        )


def get_entry_point():
    return "OrdinaryOnlineModelAgent"
