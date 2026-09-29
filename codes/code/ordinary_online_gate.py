"""XML-free live first-radar gate for the isolated ordinary-220 harness."""

from __future__ import absolute_import

import json
import os
import time

import numpy as np


def _packet_data(packet):
    if packet is None:
        return None, None
    if isinstance(packet, (tuple, list)) and len(packet) == 2:
        return packet[0], packet[1]
    return None, packet


def _quantile(values, fraction):
    if values is None or len(values) == 0:
        return 0.0
    return float(np.quantile(values, fraction))


def summarize_live_radar(packet):
    """Summarize leaderboard radar order: depth, altitude, azimuth, velocity."""
    sensor_frame, data = _packet_data(packet)
    if data is None:
        return {
            "packet_present": False, "sensor_frame": sensor_frame,
            "raw_shape": None, "points_total": 0, "finite_points": 0,
        }
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 4:
        return {
            "packet_present": True, "sensor_frame": sensor_frame,
            "raw_shape": list(array.shape), "points_total": int(array.size),
            "finite_points": 0,
            "parse_error": "expected Nx4 leaderboard radar array",
        }
    array = array[:, :4]
    finite = array[np.isfinite(array).all(axis=1)]
    depth = finite[:, 0] if len(finite) else np.empty(0, dtype=np.float32)
    altitude = finite[:, 1] if len(finite) else np.empty(0, dtype=np.float32)
    azimuth = finite[:, 2] if len(finite) else np.empty(0, dtype=np.float32)
    velocity = finite[:, 3] if len(finite) else np.empty(0, dtype=np.float32)
    valid = np.isfinite(depth) & (depth >= 0.0)
    front = valid & (np.abs(azimuth) <= 0.55)
    return {
        "packet_present": True,
        "sensor_frame": sensor_frame,
        "raw_shape": list(array.shape),
        "points_total": int(len(array)),
        "finite_points": int(len(finite)),
        "depth_min_m": _quantile(depth[valid], 0.0),
        "depth_median_m": _quantile(depth[valid], 0.5),
        "depth_max_m": _quantile(depth[valid], 1.0),
        "front_points": int(np.count_nonzero(front)),
        "front_min_depth_m": _quantile(depth[front], 0.0),
        "velocity_abs_q90_mps": _quantile(np.abs(velocity), 0.9),
        "altitude_abs_q90_rad": _quantile(np.abs(altitude), 0.9),
        "azimuth_abs_q90_rad": _quantile(np.abs(azimuth), 0.9),
    }


class OrdinaryOnlineGate(object):
    """Require live radar before enabling the isolated ordinary profile.

    This is deliberately not described as a learned cross-family classifier:
    there is no labelled 144 raw first-frame corpus yet.  It removes XML truth
    from ordinary formal decisions while retaining auditable live validation.
    """

    def __init__(self, output_path=None, **unused):
        self.output_path = output_path or os.environ.get(
            "CVCI_SCENE_GATE_LOG_PATH", ""
        )

    def decide(self, packet):
        summary = summarize_live_radar(packet)
        if not summary.get("packet_present") or summary.get("finite_points", 0) <= 0:
            raise RuntimeError("ordinary profile requires a finite live first radar frame")
        if summary.get("parse_error"):
            raise RuntimeError(summary["parse_error"])
        return {
            "decision": "ordinary_220",
            "decision_source": "live_first_radar_validated_ordinary_profile",
            "radar_candidate": "ordinary_profile_not_cross_family_classified",
            "radar_extreme_score": 0,
            "first_radar": summary,
            "truth_inputs_used": [],
            "route_reference_used": False,
            "route_id_used_for_decision": False,
            "decided_at_epoch": time.time(),
        }

    def write(self, payload):
        if not self.output_path:
            return
        parent = os.path.dirname(os.path.abspath(self.output_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        temporary = self.output_path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, self.output_path)
