from __future__ import absolute_import

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


_VEHICLE_CLASSES = ("car", "van", "truck", "bus", "motorcycle", "bicycle")


@dataclass
class TrackerConfig:
    max_match_distance: float = 3.5
    max_age: int = 5
    min_score: float = 0.28
    ego_corridor_half_width: float = 2.4
    ego_corridor_forward: float = 40.0
    closing_speed_epsilon: float = 0.15


class ObjectTracker:
    """Lightweight ego-frame multi-object tracker.

    The tracker only uses detector outputs that are already available to the
    agent. Track ids are local bookkeeping ids, not CARLA actor ids.
    """

    def __init__(self, max_match_distance: Optional[float] = None, max_age: Optional[int] = None, config: Optional[TrackerConfig] = None):
        self.config = config or TrackerConfig()
        if max_match_distance is not None:
            self.config.max_match_distance = float(max_match_distance)
        if max_age is not None:
            self.config.max_age = int(max_age)
        self._next_id = 1
        self._tracks: Dict[int, Dict[str, Any]] = {}

    @staticmethod
    def _position(obj: Dict[str, Any]) -> Optional[Tuple[float, float, float]]:
        box = obj.get("box_lidar") or obj.get("box") or {}
        if isinstance(box, dict):
            try:
                return float(box.get("x", 999.0)), float(box.get("y", 999.0)), float(box.get("z", 0.0))
            except Exception:
                return None
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        if arr.size < 2:
            return None
        z = float(arr[2]) if arr.size >= 3 else 0.0
        return float(arr[0]), float(arr[1]), z

    def _candidate_detections(self, objects: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], float, float, float]]:
        detections = []
        for obj in objects or []:
            if float(obj.get("score", obj.get("confidence", 0.0)) or 0.0) < self.config.min_score:
                continue
            pos = self._position(obj)
            if pos is None:
                continue
            detections.append((obj, pos[0], pos[1], pos[2]))
        return detections

    def _match(self, x: float, y: float, class_name: str, source: str, assigned: set) -> Optional[int]:
        best_id = None
        best_dist = float(self.config.max_match_distance)
        for tid, track in self._tracks.items():
            if tid in assigned:
                continue
            old_class = str(track.get("class_name", "")).lower()
            if old_class and class_name and old_class != class_name:
                continue
            # Camera/model and OpenPCDet observations have independent update
            # cadences.  Keeping their tracks separate prevents a weak camera
            # alias from overwriting the semantic 3D object's velocity.
            if str(track.get("source", "")) != source:
                continue
            dist = float(np.hypot(x - float(track["x"]), y - float(track["y"])))
            if dist < best_dist:
                best_id = tid
                best_dist = dist
        return best_id

    def _track_view(self, track: Dict[str, Any]) -> Dict[str, Any]:
        x = float(track["x"])
        y = float(track["y"])
        vx = float(track.get("vx", 0.0))
        vy = float(track.get("vy", 0.0))
        closing_speed = max(0.0, -vx)
        ttc = None
        if x > 0.0 and closing_speed > self.config.closing_speed_epsilon:
            ttc = float(x / closing_speed)
        class_name = str(track.get("class_name", "")).lower()
        intersects = bool(
            0.0 <= x <= self.config.ego_corridor_forward
            and abs(y) <= self.config.ego_corridor_half_width
        )
        return {
            "track_id": int(track["track_id"]),
            "class_name": track.get("class_name", ""),
            "confidence": float(track.get("confidence", 0.0)),
            "score": float(track.get("confidence", 0.0)),
            "x": x,
            "y": y,
            "z": float(track.get("z", 0.0)),
            "longitudinal_velocity": vx,
            "lateral_velocity": vy,
            "relative_velocity": vx,
            "vx": vx,
            "vy": vy,
            "speed": float(np.hypot(vx, vy)),
            "closing_speed": closing_speed,
            "distance": float(np.hypot(x, y)),
            "ttc": ttc,
            "time_headway": ttc,
            "intersects_ego_corridor": intersects,
            "lane_relation": "ego_corridor" if intersects else ("left" if y < 0.0 else "right"),
            "observed_frames": int(track.get("observed_frames", 1)),
            "age": int(track.get("age", 0)),
            "timestamp": float(track.get("timestamp", 0.0)),
            "measurement_timestamp": float(
                track.get("measurement_timestamp", track.get("timestamp", 0.0))
            ),
            "source": str(track.get("source", "")),
            "is_reversing_candidate": bool(class_name in _VEHICLE_CLASSES and intersects and closing_speed > 0.45),
        }

    def update(self, objects: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        timestamp = float(timestamp)
        detections = self._candidate_detections(objects)
        assigned = set()
        for obj, x, y, z in detections:
            class_name = str(obj.get("class_name", "")).lower()
            source = str(obj.get("source", ""))
            try:
                measurement_timestamp = float(
                    obj.get("measurement_timestamp", timestamp)
                )
            except Exception:
                measurement_timestamp = timestamp
            track_id = self._match(x, y, class_name, source, assigned)
            confidence = float(obj.get("score", obj.get("confidence", 0.0)) or 0.0)
            if track_id is None:
                track_id = self._next_id
                self._next_id += 1
                vx = vy = 0.0
                observed_frames = 1
            else:
                prev = self._tracks[track_id]
                previous_measurement_timestamp = float(
                    prev.get("measurement_timestamp", prev.get("timestamp", timestamp))
                )
                repeated_measurement = bool(
                    source == "openpcdet"
                    and measurement_timestamp <= previous_measurement_timestamp + 1e-6
                )
                if repeated_measurement:
                    # latest_or_submit intentionally republishes the most
                    # recent semantic result between neural inferences.  It is
                    # a keepalive, not a new velocity sample.
                    x = float(prev["x"])
                    y = float(prev["y"])
                    z = float(prev.get("z", z))
                    vx = float(prev.get("vx", 0.0))
                    vy = float(prev.get("vy", 0.0))
                    observed_frames = int(prev.get("observed_frames", 1))
                    measurement_timestamp = previous_measurement_timestamp
                else:
                    dt = max(
                        measurement_timestamp - previous_measurement_timestamp,
                        1e-3,
                    )
                    vx = (x - float(prev["x"])) / dt
                    vy = (y - float(prev["y"])) / dt
                    observed_frames = int(prev.get("observed_frames", 1)) + 1
            assigned.add(track_id)
            self._tracks[track_id] = {
                "track_id": track_id,
                "class_name": obj.get("class_name", ""),
                "confidence": confidence,
                "x": float(x),
                "y": float(y),
                "z": float(z),
                "vx": float(vx),
                "vy": float(vy),
                "timestamp": timestamp,
                "measurement_timestamp": measurement_timestamp,
                "source": source,
                "age": 0,
                "observed_frames": observed_frames,
            }

        stale = []
        for tid, track in self._tracks.items():
            if tid not in assigned:
                track["age"] = int(track.get("age", 0)) + 1
                if track["age"] > self.config.max_age:
                    stale.append(tid)
        for tid in stale:
            self._tracks.pop(tid, None)
        return [self._track_view(track) for track in self._tracks.values()]
