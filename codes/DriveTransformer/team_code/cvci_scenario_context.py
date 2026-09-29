from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ScenarioContext:
    route_id: Optional[str] = None
    macro_scenario: str = "unknown"
    scenario_name: str = ""
    scenario_type: str = ""
    town: str = ""
    weather_bucket: str = "normal"
    light_bucket: str = "day"
    ego_speed: float = 0.0
    route_command: Optional[str] = None
    waypoint_features: Dict[str, Any] = field(default_factory=dict)
    detections: List[Dict[str, Any]] = field(default_factory=list)
    radar_objects: List[Dict[str, Any]] = field(default_factory=list)
    lidar_objects: List[Dict[str, Any]] = field(default_factory=list)
    phase: str = "unknown"
    risk_flags: Dict[str, bool] = field(default_factory=dict)
    confidence: float = 0.0
    frame_idx: int = 0


@dataclass
class RuleAction:
    target_speed: Optional[float] = None
    throttle_scale: float = 1.0
    brake: Optional[float] = None
    steer_scale: float = 1.0
    steer_smoothing: Optional[float] = None
    steer_bias: float = 0.0
    steer_limit: Optional[float] = None
    hold_frames: int = 0
    reason: str = ""
    active_rule: str = ""
