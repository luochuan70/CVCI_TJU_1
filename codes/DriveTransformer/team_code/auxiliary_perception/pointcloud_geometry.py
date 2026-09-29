import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class LidarGeometryConfig:
    x_min: float = 0.0
    x_max: float = 35.0
    y_abs_max: float = 2.4
    z_min: float = -1.8
    z_max: float = 2.5
    near_x: float = 8.0
    stale_timeout_sec: float = 0.35
    min_points: int = 12
    blockage_points: int = 80
    side_blockage_points: int = 35
    center_y_abs: float = 0.65
    side_y_min: float = 0.65


class PointCloudGeometryFallback:
    """Lightweight legal LiDAR geometry fallback.

    The input is raw ray-cast LiDAR points in the agent/local LiDAR frame. It never reads
    CARLA actors, labels, semantic LiDAR, XML actor coordinates, or world state.
    """

    def __init__(self, config: Optional[LidarGeometryConfig] = None):
        self.config = config or LidarGeometryConfig()
        self.last_result: Dict[str, Any] = {"available": False, "stale": True}
        self.last_timestamp: Optional[float] = None

    @staticmethod
    def normalize_points(raw_points: Any) -> np.ndarray:
        if raw_points is None:
            return np.zeros((0, 4), dtype=np.float32)
        arr = np.asarray(raw_points)
        if arr.dtype.fields:
            names = arr.dtype.names or ()
            cols = []
            for name in ("x", "y", "z", "intensity"):
                if name in names:
                    cols.append(arr[name].astype(np.float32).reshape(-1, 1))
            if len(cols) >= 3:
                if len(cols) == 3:
                    cols.append(np.zeros_like(cols[0]))
                return np.concatenate(cols[:4], axis=1)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim == 1:
            if arr.size % 4 == 0:
                arr = arr.reshape(-1, 4)
            elif arr.size % 3 == 0:
                arr = arr.reshape(-1, 3)
            else:
                return np.zeros((0, 4), dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 3:
            return np.zeros((0, 4), dtype=np.float32)
        if arr.shape[1] == 3:
            arr = np.concatenate([arr[:, :3], np.zeros((arr.shape[0], 1), dtype=np.float32)], axis=1)
        return arr[:, :4]

    def update(self, raw_points: Any, timestamp: Optional[float] = None) -> Dict[str, Any]:
        cfg = self.config
        points = self.normalize_points(raw_points)
        now = float(timestamp if timestamp is not None else time.time())
        result: Dict[str, Any] = {
            "available": bool(points.shape[0] >= cfg.min_points),
            "stale": False,
            "timestamp": now,
            "num_points": int(points.shape[0]),
            "front_blocked": False,
            "near_blocked": False,
            "front_distance": None,
            "front_point_count": 0,
            "left_density": 0,
            "right_density": 0,
            "center_density": 0,
            "left_nearest_distance": None,
            "right_nearest_distance": None,
            "center_nearest_distance": None,
            "left_blockage_ratio": 0.0,
            "right_blockage_ratio": 0.0,
            "center_blockage_ratio": 0.0,
            "lateral_centroid": 0.0,
            "open_side": "unknown",
            "corridor_blockage_ratio": 0.0,
        }
        if not result["available"]:
            self.last_result = result
            self.last_timestamp = now
            return result

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        roi = (
            (x >= cfg.x_min) & (x <= cfg.x_max) &
            (np.abs(y) <= cfg.y_abs_max) &
            (z >= cfg.z_min) & (z <= cfg.z_max)
        )
        front = points[roi]
        result["front_point_count"] = int(front.shape[0])
        if front.shape[0] > 0:
            result["front_distance"] = float(np.min(front[:, 0]))
            result["front_blocked"] = bool(front.shape[0] >= cfg.blockage_points)
            result["near_blocked"] = bool(np.any(front[:, 0] <= cfg.near_x))
            left = front[front[:, 1] < -cfg.side_y_min]
            right = front[front[:, 1] > cfg.side_y_min]
            center = front[np.abs(front[:, 1]) <= cfg.center_y_abs]
            result["left_density"] = int(left.shape[0])
            result["right_density"] = int(right.shape[0])
            result["center_density"] = int(center.shape[0])
            if left.shape[0] > 0:
                result["left_nearest_distance"] = float(np.min(left[:, 0]))
            if right.shape[0] > 0:
                result["right_nearest_distance"] = float(np.min(right[:, 0]))
            if center.shape[0] > 0:
                result["center_nearest_distance"] = float(np.min(center[:, 0]))
            result["left_blockage_ratio"] = float(min(1.0, left.shape[0] / float(max(cfg.side_blockage_points, 1))))
            result["right_blockage_ratio"] = float(min(1.0, right.shape[0] / float(max(cfg.side_blockage_points, 1))))
            result["center_blockage_ratio"] = float(min(1.0, center.shape[0] / float(max(cfg.side_blockage_points, 1))))
            result["lateral_centroid"] = float(np.mean(front[:, 1]))
            if left.shape[0] + 8 < right.shape[0]:
                result["open_side"] = "left"
            elif right.shape[0] + 8 < left.shape[0]:
                result["open_side"] = "right"
            else:
                result["open_side"] = "balanced"
            result["corridor_blockage_ratio"] = float(min(1.0, front.shape[0] / float(max(cfg.blockage_points, 1))))
        self.last_result = result
        self.last_timestamp = now
        return result

    def get_latest(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        if self.last_timestamp is None:
            return {"available": False, "stale": True}
        now = float(timestamp if timestamp is not None else time.time())
        result = dict(self.last_result)
        result["stale"] = bool(now - float(self.last_timestamp) > self.config.stale_timeout_sec)
        return result
