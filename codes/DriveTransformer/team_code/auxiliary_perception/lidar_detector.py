from __future__ import absolute_import

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in ("1", "true", "yes", "on")


@dataclass
class LidarDetectorConfig:
    enabled: bool = field(default_factory=lambda: _env_flag("CVCI_LIDAR_DETECTOR_ENABLED", "1"))
    required: bool = field(default_factory=lambda: _env_flag("CVCI_LIDAR_DETECTOR_REQUIRED", "1"))
    backend: str = field(default_factory=lambda: os.environ.get("CVCI_LIDAR_DETECTOR_BACKEND", "openpcdet"))
    root: str = field(default_factory=lambda: os.environ.get("CVCI_OPENPCDET_ROOT", str(DEFAULT_OPENPCDET_ROOT)))
    config_path: str = field(
        default_factory=lambda: os.environ.get(
            "CVCI_LIDAR_DETECTOR_CONFIG",
            str(DEFAULT_OPENPCDET_ROOT / "tools" / "cfgs" / "kitti_models" / "pointpillar.yaml"),
        )
    )
    checkpoint_path: str = field(
        default_factory=lambda: os.environ.get(
            "CVCI_LIDAR_DETECTOR_MODEL",
            str(DEFAULT_OPENPCDET_ROOT / "checkpoints" / "pointpillar_kitti.pth"),
        )
    )
    score_threshold: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_DETECTOR_SCORE_THRESHOLD", "0.35")))
    stale_timeout_sec: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_DETECTOR_STALE_SEC", "0.75")))
    min_interval_sec: float = field(default_factory=lambda: float(os.environ.get("CVCI_LIDAR_DETECTOR_INTERVAL_SEC", "0.5")))
    max_points: int = field(default_factory=lambda: int(os.environ.get("CVCI_LIDAR_DETECTOR_MAX_POINTS", "120000")))
    device: str = field(default_factory=lambda: os.environ.get("CVCI_LIDAR_DETECTOR_DEVICE", "cuda"))
    async_enabled: bool = field(default_factory=lambda: _env_flag("CVCI_LIDAR_DETECTOR_ASYNC", "1"))


class LidarDetector:
    """Optional neural LiDAR 3D detector wrapper.

    The wrapper only consumes raw ray-cast LiDAR points from the agent. It never
    reads CARLA actors, ScenarioRunner state, route ids, or XML ground truth.
    When OpenPCDet, the config, or the checkpoint is unavailable it returns a
    typed unavailable status so the caller can fall back to DriveTransformer and
    the raw point-cloud geometry path.
    """

    def __init__(self, config: Optional[LidarDetectorConfig] = None):
        self.config = config or LidarDetectorConfig()
        self._ready = False
        self._load_attempted = False
        self._load_error = ""
        self._model = None
        self._dataset = None
        self._class_names: List[str] = []
        self._load_future = None
        self._future = None
        self._executor = None
        self._last_submit_timestamp: Optional[float] = None
        self._latest: Dict[str, Any] = {
            "enabled": bool(self.config.enabled),
            "available": False,
            "stale": True,
            "status": "disabled" if not self.config.enabled else "not_loaded",
            "objects": [],
        }

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
                    cols.append(np.zeros_like(cols[0], dtype=np.float32))
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
        return arr[:, :4].astype(np.float32, copy=False)

    @staticmethod
    def carla_to_detector_points(points: np.ndarray) -> np.ndarray:
        """Convert agent/CARLA LiDAR frame to KITTI/OpenPCDet LiDAR convention.

        The existing agent-side code treats y < 0 as left and y > 0 as right.
        KITTI/OpenPCDet uses x-forward, y-left, z-up, so only the lateral axis is
        mirrored before neural inference.
        """
        converted = np.array(points, dtype=np.float32, copy=True)
        if converted.size:
            converted[:, 1] *= -1.0
        return converted

    @staticmethod
    def detector_box_to_agent(box: np.ndarray) -> Dict[str, float]:
        arr = np.asarray(box, dtype=np.float32).reshape(-1)
        x = float(arr[0]) if arr.size > 0 else 0.0
        y = float(-arr[1]) if arr.size > 1 else 0.0
        z = float(arr[2]) if arr.size > 2 else 0.0
        dx = float(arr[3]) if arr.size > 3 else 0.0
        dy = float(arr[4]) if arr.size > 4 else 0.0
        dz = float(arr[5]) if arr.size > 5 else 0.0
        yaw = float(-arr[6]) if arr.size > 6 else 0.0
        return {"x": x, "y": y, "z": z, "dx": dx, "dy": dy, "dz": dz, "yaw": yaw}

    @staticmethod
    def normalize_class_name(class_name: Any) -> str:
        value = str(class_name).lower()
        return {"cyclist": "bicycle", "person": "pedestrian"}.get(value, value)

    def _unavailable(self, status: str, timestamp: float, error: str = "") -> Dict[str, Any]:
        if self.config.required and self.config.enabled and status not in ("loading", "no_points"):
            detail = error or status
            raise RuntimeError("CVCI_REQUIRED_LIDAR unavailable: %s" % detail)
        return {
            "enabled": bool(self.config.enabled),
            "available": False,
            "stale": True,
            "status": status,
            "objects": [],
            "timestamp": float(timestamp),
            "error": error,
        }

    def _load_openpcdet(self, timestamp: float) -> bool:
        try:
            root = Path(self.config.root)
            cfg_path = Path(self.config.config_path)
            ckpt_path = Path(self.config.checkpoint_path)
            def _exists(path: Path) -> bool:
                try:
                    return path.exists()
                except OSError:
                    return False

            def _nonempty_file(path: Path) -> bool:
                try:
                    return path.exists() and path.stat().st_size > 0
                except OSError:
                    return False

            if self.config.backend.lower() != "openpcdet":
                self._load_error = "unsupported_backend:%s" % self.config.backend
                self._latest = self._unavailable("unsupported_backend", timestamp, self._load_error)
                return False
            if not _nonempty_file(ckpt_path):
                self._load_error = "missing_checkpoint:%s" % ckpt_path
                self._latest = self._unavailable("missing_checkpoint", timestamp, self._load_error)
                return False
            if not _exists(root):
                self._load_error = "missing_openpcdet_root:%s" % root
                self._latest = self._unavailable("missing_openpcdet_root", timestamp, self._load_error)
                return False
            if not _exists(cfg_path):
                self._load_error = "missing_config:%s" % cfg_path
                self._latest = self._unavailable("missing_config", timestamp, self._load_error)
                return False

            from pcdet.config import cfg, cfg_from_yaml_file
            from pcdet.datasets import DatasetTemplate
            from pcdet.models import build_network
            from pcdet.utils import common_utils

            old_cwd = os.getcwd()
            try:
                os.chdir(str(root / "tools"))
                cfg_from_yaml_file(str(cfg_path), cfg)
            finally:
                os.chdir(old_cwd)

            class _OnlineDataset(DatasetTemplate):
                def __len__(self):
                    return 1

                def __getitem__(self, index):
                    raise IndexError(index)

            logger = common_utils.create_logger(log_file=None, rank=0)
            dataset = _OnlineDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, root_path=root, logger=logger)
            model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=dataset)
            model.load_params_from_file(filename=str(ckpt_path), logger=logger, to_cpu=True)
            if self.config.device == "cuda":
                model.cuda()
            model.eval()
            self._dataset = dataset
            self._model = model
            self._class_names = list(cfg.CLASS_NAMES)
            self._ready = True
            if self.config.async_enabled:
                self._executor = ThreadPoolExecutor(max_workers=1)
            return True
        except Exception as exc:
            self._load_error = repr(exc)
            if self.config.required:
                raise RuntimeError("CVCI_REQUIRED_LIDAR load failed: %s" % self._load_error)
            self._latest = self._unavailable("load_failed", timestamp, self._load_error)
            return False

    def _ensure_ready(self, timestamp: float) -> bool:
        if self._ready:
            return True
        if not self.config.enabled:
            self._latest = self._unavailable("disabled", timestamp)
            return False
        if self.config.async_enabled:
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=1)
            if not self._load_attempted:
                self._load_attempted = True
                self._latest = self._unavailable("loading", timestamp)
                self._latest["stale"] = False
                self._load_future = self._executor.submit(self._load_openpcdet, timestamp)
                return False
            if self._load_future is not None and self._load_future.done():
                try:
                    ok = bool(self._load_future.result())
                except Exception as exc:
                    ok = False
                    self._load_error = repr(exc)
                    if self.config.required:
                        raise RuntimeError("CVCI_REQUIRED_LIDAR load failed: %s" % self._load_error)
                    self._latest = self._unavailable("load_failed", timestamp, self._load_error)
                self._load_future = None
                return bool(ok and self._ready)
            if self._load_future is not None:
                self._latest = self._unavailable("loading", timestamp)
                self._latest["stale"] = False
                return False
            self._latest = self._unavailable("load_failed", timestamp, self._load_error)
            return False
        if self._load_attempted:
            self._latest = self._unavailable("load_failed", timestamp, self._load_error)
            return False
        self._load_attempted = True
        return self._load_openpcdet(timestamp)

    def warmup(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """Load the network and initialize CUDA kernels before driving."""
        now = float(timestamp if timestamp is not None else time.time())
        if not self._ensure_ready(now):
            return dict(self._latest)
        warmup_count = max(
            512, int(os.environ.get("CVCI_LIDAR_DETECTOR_WARMUP_POINTS", "4000"))
        )
        rng = np.random.RandomState(20260716)
        points = np.empty((warmup_count, 4), dtype=np.float32)
        points[:, 0] = rng.uniform(0.0, 69.5, size=warmup_count)
        points[:, 1] = rng.uniform(-39.5, 39.5, size=warmup_count)
        points[:, 2] = rng.uniform(-2.7, 0.8, size=warmup_count)
        points[:, 3] = rng.uniform(0.0, 1.0, size=warmup_count)
        self._infer(points, now)
        return {
            "enabled": True,
            "available": True,
            "stale": False,
            "status": "model_ready",
            "timestamp": now,
            "objects": [],
            "object_count": 0,
        }

    def _infer(self, points: np.ndarray, timestamp: float) -> Dict[str, Any]:
        import torch
        from pcdet.models import load_data_to_gpu

        detector_points = self.carla_to_detector_points(points)
        data_dict = {"points": detector_points, "frame_id": 0}
        data_dict = self._dataset.prepare_data(data_dict=data_dict)
        batch_dict = self._dataset.collate_batch([data_dict])
        if self.config.device == "cuda":
            load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, _ = self._model.forward(batch_dict)
        pred = pred_dicts[0]
        boxes = pred["pred_boxes"].detach().cpu().numpy()
        scores = pred["pred_scores"].detach().cpu().numpy()
        labels = pred["pred_labels"].detach().cpu().numpy()
        objects = []
        for box, score, label in zip(boxes, scores, labels):
            score = float(score)
            if score < self.config.score_threshold:
                continue
            idx = int(label) - 1
            raw_class_name = self._class_names[idx] if 0 <= idx < len(self._class_names) else "unknown"
            # Match the vocabulary consumed by the shared CVCI feature and
            # rule stack. KITTI calls a bicycle rider "Cyclist" while the
            # existing online rules consistently use "bicycle".
            class_name = self.normalize_class_name(raw_class_name)
            box_lidar = self.detector_box_to_agent(box)
            objects.append(
                {
                    "class_name": class_name,
                    "detector_class_name": str(raw_class_name).lower(),
                    "score": score,
                    "confidence": score,
                    "box_lidar": box_lidar,
                    "source": "openpcdet",
                    "measurement_timestamp": float(timestamp),
                }
            )
        return {
            "enabled": True,
            "available": True,
            "stale": False,
            "status": "ok",
            "timestamp": float(timestamp),
            "objects": objects,
            "object_count": len(objects),
        }

    def latest_or_submit(self, raw_points: Any, timestamp: Optional[float] = None) -> Dict[str, Any]:
        now = float(timestamp if timestamp is not None else time.time())
        points = self.normalize_points(raw_points)
        if points.shape[0] <= 0 and not self._ready:
            self._latest = self._unavailable("no_points", now)
            self._latest["stale"] = False
            return dict(self._latest)
        if not self._ensure_ready(now):
            return dict(self._latest)

        if self._future is not None and self._future.done():
            try:
                self._latest = self._future.result()
            except Exception as exc:
                if self.config.required:
                    raise RuntimeError("CVCI_REQUIRED_LIDAR inference failed: %s" % repr(exc))
                self._latest = self._unavailable("inference_failed", now, repr(exc))
            self._future = None

        if points.shape[0] > 0:
            if points.shape[0] > self.config.max_points:
                indices = np.linspace(0, points.shape[0] - 1, int(self.config.max_points)).astype(np.int64)
                points = points[indices]
            can_submit = (
                self._last_submit_timestamp is None
                or now - float(self._last_submit_timestamp) >= float(self.config.min_interval_sec)
            )
            if self.config.async_enabled:
                if can_submit and self._future is None and self._executor is not None:
                    self._future = self._executor.submit(self._infer, points.copy(), now)
                    self._last_submit_timestamp = now
            else:
                if can_submit:
                    self._latest = self._infer(points, now)
                    self._last_submit_timestamp = now

        latest = dict(self._latest)
        latest["stale"] = bool(now - float(latest.get("timestamp", 0.0) or 0.0) > self.config.stale_timeout_sec)
        latest["min_interval_sec"] = float(self.config.min_interval_sec)
        latest["last_submit_timestamp"] = self._last_submit_timestamp
        return latest
