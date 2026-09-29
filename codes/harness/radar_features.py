"""Temporal, radar-only feature extraction for ordinary-scene classification."""

from __future__ import absolute_import

import math
from collections import deque

import numpy as np


FEATURE_NAMES = (
    "point_count", "front_count", "front_close_count", "left_close_count",
    "right_close_count", "front_dynamic_count", "front_closing_count",
    "depth_min", "depth_p10", "depth_median", "depth_mean", "depth_std",
    "front_width", "left_nearest", "right_nearest", "velocity_abs_q90",
    "closing_velocity_q90", "altitude_abs_q90", "front_ttc", "ego_speed",
    "steer_abs", "brake", "depth_trend", "front_depth_trend",
)


def _packet_data(packet):
    if packet is None:
        return None, None
    if isinstance(packet, (tuple, list)) and len(packet) == 2:
        return packet[0], packet[1]
    return None, packet


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _quantile(values, q, default=0.0):
    if values is None or len(values) == 0:
        return float(default)
    return float(np.quantile(values, q))


def _safe_array(packet):
    frame, data = _packet_data(packet)
    if data is None:
        return frame, np.empty((0, 4), dtype=np.float32), "missing"
    try:
        array = np.asarray(data, dtype=np.float32)
    except Exception as exc:
        return frame, np.empty((0, 4), dtype=np.float32), "parse_error:{}".format(exc)
    if array.size == 0:
        return frame, np.empty((0, 4), dtype=np.float32), "empty"
    if array.ndim != 2 or array.shape[1] < 4:
        return frame, np.empty((0, 4), dtype=np.float32), "shape:{}".format(tuple(array.shape))
    array = array[:, :4]
    finite = array[np.isfinite(array).all(axis=1)]
    # leaderboard sensor_interface.py reverses CARLA's raw packet.  CARLA
    # 0.9.15's RadarDetection storage is [velocity, azimuth, altitude,
    # depth], so the post-flip packet delivered to the agent is
    # [depth, altitude, azimuth, velocity].  Keep this explicit: treating the
    # first value as velocity makes ordinary routes look like zero-distance
    # obstacles and can change the driving control.
    finite = finite[finite[:, 0] >= 0.0]
    return frame, finite, "ok"


class RadarFeatureExtractor(object):
    """Extract a fixed-size temporal vector and preserve an auditable window."""

    def __init__(self, max_frames=32, dt_seconds=0.05):
        self.max_frames = int(max_frames)
        self.dt_seconds = float(dt_seconds)
        self.history = deque(maxlen=self.max_frames)
        self.frame_numbers = deque(maxlen=self.max_frames)
        self.last_timestamp = None

    def reset(self):
        self.history.clear()
        self.frame_numbers.clear()
        self.last_timestamp = None

    def _one_frame(self, packet, ego_speed=0.0, steer=0.0, brake=0.0):
        frame, points, quality = _safe_array(packet)
        if len(points):
            depth = points[:, 0]
            altitude = points[:, 1]
            azimuth = points[:, 2]
            velocity = points[:, 3]
        else:
            velocity = altitude = azimuth = depth = np.empty(0, dtype=np.float32)
        front = np.abs(azimuth) <= math.radians(22.0)
        left = azimuth < -math.radians(22.0)
        right = azimuth > math.radians(22.0)
        close = depth <= 10.0
        front_close = front & close
        # CARLA reports radial velocity relative to the radar sensor.  A
        # stationary world return therefore looks like approximately
        # ``-ego_speed*cos(azimuth)`` while the vehicle is moving forward.
        # Compensate that projection before calling a return dynamic or
        # closing; otherwise every roadside/static return is classified as a
        # moving obstacle and suppresses the ordinary static recovery state.
        ego_speed_abs = abs(_finite(ego_speed))
        ego_radial_projection = ego_speed_abs * np.cos(azimuth)
        relative_velocity = velocity + ego_radial_projection
        dynamic = np.abs(relative_velocity) >= 1.0
        # The depth trend below remains an independent, sign-insensitive
        # approach signal.  The residual radial velocity is the dynamic
        # evidence field used for cluster/recovery gates.
        sign_closing = front & (relative_velocity < -0.5)
        # Two non-vector semantic counts preserve coarse startup geometry for
        # the ordinary classifier.  Keep the post-flip CARLA packet order
        # explicit: column 1 is altitude and column 2 is azimuth.  These
        # fields intentionally stay out of FEATURE_NAMES so existing centroid
        # dimensions and trained weights remain byte-for-byte compatible.
        near_level = (depth < 20.0) & (np.abs(altitude) < 0.05)
        left_wide_mid = (depth < 40.0) & (azimuth < -0.30)
        # Online motion summary for a medium-range traffic-flow event.  Keep
        # it outside FEATURE_NAMES so existing centroid inputs do not change.
        fast_approach = (
            (depth >= 8.0)
            & (depth <= 60.0)
            & (np.abs(azimuth) <= math.radians(35.0))
            & (relative_velocity >= -13.0)
            & (relative_velocity <= -5.0)
        )
        fast_approach_depth = depth[fast_approach]
        flow_fast_close = (
            (depth > 1.0)
            & (depth <= 6.0)
            & (np.abs(azimuth) <= math.radians(70.0))
            & (relative_velocity >= -13.0)
            & (relative_velocity <= -5.0)
        )
        flow_slow_near = (
            (depth > 1.0)
            & (depth < 25.0)
            & (np.abs(azimuth) <= math.radians(35.0))
            & (np.abs(relative_velocity) < 2.0)
        )
        # Same-direction traffic-flow signature.  A moving actor travelling
        # with the road flow produces a positive ego-compensated radial speed
        # while guardrails/background remain near zero.  Keep this semantic
        # count outside FEATURE_NAMES so the trained centroid dimensions stay
        # byte-for-byte compatible.
        same_direction_flow = (
            (depth > 1.0)
            & (depth <= 40.0)
            & (np.abs(altitude) <= 0.35)
            & (np.abs(azimuth) <= math.radians(70.0))
            & (relative_velocity >= 6.0)
            & (relative_velocity <= 12.5)
        )
        same_direction_flow_speed = relative_velocity[same_direction_flow]
        same_direction_flow_depth = depth[same_direction_flow]
        # Estimate the actual speed of the oncoming flow while it is still
        # sufficiently far and frontal for radial speed to approximate actor
        # speed.  This remains a non-vector semantic observation: it does not
        # change any trained centroid dimension.
        flow_approach_speed = (
            (depth > 3.0)
            & (depth < 30.0)
            & (np.abs(azimuth) < math.radians(50.0))
            & (relative_velocity < -4.0)
            & (relative_velocity > -15.0)
        )
        front_depth = depth[front]
        left_depth = depth[left]
        right_depth = depth[right]
        if len(front_depth):
            front_min = float(np.min(front_depth))
            front_width = float(np.max(azimuth[front]) - np.min(azimuth[front]))
        else:
            front_min = 0.0
            front_width = 0.0
        return {
            "point_count": float(len(points)),
            "front_count": float(np.count_nonzero(front)),
            "front_close_count": float(np.count_nonzero(front_close)),
            "left_close_count": float(np.count_nonzero(left & close)),
            "right_close_count": float(np.count_nonzero(right & close)),
            "front_dynamic_count": float(np.count_nonzero(front & dynamic)),
            "front_closing_count": float(np.count_nonzero(sign_closing)),
            "near_level_count": float(np.count_nonzero(near_level)),
            "left_wide_mid_count": float(np.count_nonzero(left_wide_mid)),
            "fast_approach_count": float(np.count_nonzero(fast_approach)),
            "fast_approach_depth_median": _quantile(
                fast_approach_depth, 0.50, 0.0
            ),
            "flow_fast_close_count": float(np.count_nonzero(flow_fast_close)),
            "flow_slow_near_count": float(np.count_nonzero(flow_slow_near)),
            "same_direction_flow_count": float(
                np.count_nonzero(same_direction_flow)
            ),
            # Non-vector semantic measurements used to distinguish a fast,
            # distant highway stream from the already-validated slow merge
            # stream.  They deliberately stay outside FEATURE_NAMES so the
            # centroid model and every existing class keep identical inputs.
            "same_direction_flow_speed_median_mps": _quantile(
                same_direction_flow_speed, 0.50, 0.0
            ),
            "same_direction_flow_depth_median_m": _quantile(
                same_direction_flow_depth, 0.50, 0.0
            ),
            "flow_approach_speed_count": float(np.count_nonzero(flow_approach_speed)),
            "flow_approach_speed_median_mps": _quantile(
                -relative_velocity[flow_approach_speed], 0.50, 0.0
            ),
            "depth_min": _quantile(depth, 0.0, 0.0),
            "depth_p10": _quantile(depth, 0.10, 0.0),
            "depth_median": _quantile(depth, 0.50, 0.0),
            "depth_mean": _quantile(depth, 0.50, 0.0) if len(depth) == 0 else float(np.mean(depth)),
            "depth_std": float(np.std(depth)) if len(depth) else 0.0,
            "front_width": front_width,
            "left_nearest": _quantile(left_depth, 0.0, 0.0),
            "right_nearest": _quantile(right_depth, 0.0, 0.0),
            "velocity_abs_q90": _quantile(np.abs(relative_velocity), 0.90, 0.0),
            "closing_velocity_q90": _quantile(np.maximum(-relative_velocity, 0.0), 0.90, 0.0),
            "altitude_abs_q90": _quantile(np.abs(altitude), 0.90, 0.0),
            "front_ttc": 0.0,
            "front_min": front_min,
            "ego_speed": ego_speed_abs,
            "steer_abs": abs(_finite(steer)),
            "brake": max(0.0, min(1.0, _finite(brake))),
            "depth_trend": 0.0,
            "front_depth_trend": 0.0,
            "sensor_frame": frame,
            "quality": quality,
        }

    def extract(self, packet, ego_speed=0.0, steer=0.0, brake=0.0, timestamp=None):
        current = self._one_frame(packet, ego_speed, steer, brake)
        previous = self.history[-1] if self.history else None
        if previous is not None:
            dt = self.dt_seconds
            if timestamp is not None and self.last_timestamp is not None:
                observed = _finite(timestamp) - _finite(self.last_timestamp)
                if observed > 1e-4 and observed < 2.0:
                    dt = observed
            # Positive value means the observed object got closer.
            current["depth_trend"] = previous["depth_median"] - current["depth_median"]
            current["front_depth_trend"] = previous["front_min"] - current["front_min"]
            if current["front_min"] > 0.0 and current["front_depth_trend"] > 0.0:
                current["front_ttc"] = current["front_min"] / max(
                    current["front_depth_trend"] / max(dt, 1e-3), 0.1
                )
        # There is no useful TTC when the frame has no front target.
        if current["front_min"] <= 0.0:
            current["front_ttc"] = 0.0
        self.history.append(current)
        self.frame_numbers.append(current["sensor_frame"])
        self.last_timestamp = timestamp

        matrix = np.asarray(
            [[float(item.get(name, 0.0)) for name in FEATURE_NAMES] for item in self.history],
            dtype=np.float32,
        )
        latest = matrix[-1] if len(matrix) else np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        median = np.median(matrix, axis=0) if len(matrix) else latest
        minimum = np.min(matrix, axis=0) if len(matrix) else latest
        maximum = np.max(matrix, axis=0) if len(matrix) else latest
        if len(matrix) > 1:
            trend = matrix[-1] - matrix[0]
        else:
            trend = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        # The vector deliberately has no route/class/file metadata.  Keep the
        # same order in recordings and the trained model.
        vector = np.concatenate([latest, median, minimum, maximum, trend]).astype(float)
        named = dict(current)
        named["history_frames"] = len(self.history)
        named["feature_names"] = list(FEATURE_NAMES)
        named["vector"] = vector.tolist()
        named["frame_numbers"] = list(self.frame_numbers)
        return named


def vector_feature_names():
    names = []
    for suffix in ("latest", "median", "minimum", "maximum", "trend"):
        names.extend("{}_{}".format(suffix, name) for name in FEATURE_NAMES)
    return names
