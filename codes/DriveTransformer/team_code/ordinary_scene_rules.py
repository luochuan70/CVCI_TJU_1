"""Conservative sensor-only rules for ordinary-220 routes.

The learned ordinary controller remains the owner of normal driving, traffic
lights, stop signs, VRU encounters, junctions, merges, and parking.  This
module deliberately adds only one narrow recovery: after a verified static
obstacle has left the vehicle stationary for several frames, a known-clear
side receives a small bounded forward bias.

The XML mapping at the bottom is audit documentation only.  It is never read
by the runtime classifier and cannot create a route-prior controller.
"""
from __future__ import absolute_import

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class OrdinaryRuleDecision:
    """A bounded action proposal plus an auditable behaviour-family label."""

    category: str = "normal_cruise"
    confidence: float = 0.0
    state: str = "NORMAL"
    active: bool = False
    target_speed: Optional[float] = None
    throttle_cap: Optional[float] = None
    throttle_floor: Optional[float] = None
    brake_cap: Optional[float] = None
    brake: Optional[float] = None
    steer_limit: Optional[float] = None
    steer_bias: Optional[float] = None
    steer_min_magnitude: Optional[float] = None
    reason: str = "ordinary_passthrough"


def _float(features: Any, name: str, default: float = 0.0) -> float:
    value = getattr(features, name, default)
    try:
        return float(default if value is None else value)
    except (TypeError, ValueError):
        return float(default)


def _optional_float(features: Any, name: str) -> Optional[float]:
    value = getattr(features, name, None)
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _text(features: Any, name: str) -> str:
    return str(getattr(features, name, "") or "").strip().lower()


def _object_xy(obj: Dict[str, Any]) -> Tuple[float, float]:
    box = obj.get("box_lidar") or obj.get("box") or {}
    try:
        return float(box.get("x", obj.get("x", 999.0))), float(
            box.get("y", obj.get("y", 999.0))
        )
    except (TypeError, ValueError, AttributeError):
        return 999.0, 999.0


def _class(obj: Dict[str, Any]) -> str:
    return str(obj.get("class_name", "") or "").strip().lower()


def _is_vru(name: str) -> bool:
    return name in {
        "person",
        "pedestrian",
        "walker",
        "bicycle",
        "bike",
        "cyclist",
        "motorcycle",
        "motorbike",
    }


def _is_turn_command(value: str) -> bool:
    if not value:
        return False
    if any(token in value for token in ("left", "right", "turn", "change")):
        return True
    # Bench2Drive's RoutePlanner exposes the six RoadOption values as ints.
    return value in {"1", "2", "5", "6"}


class OrdinarySceneRuleEngine:
    """Stateful, deliberately conservative ordinary-route rule engine."""

    def __init__(self, min_clear_frames: int = 8):
        # Keep the old constructor contract for callers, but require a longer
        # stationary observation before the sole ordinary rule can act.
        self.min_clear_frames = max(1, int(min_clear_frames))
        self.min_blocked_frames = max(12, self.min_clear_frames)
        self.stopped_frames = 0
        self.clear_frames = 0
        self.blocked_frames = 0
        self.last_category = "normal_cruise"
        self.category_frames = 0

    @staticmethod
    def _front_vru_distance(features: Any) -> Optional[float]:
        values = [_optional_float(features, "front_pedestrian_distance")]
        advanced_live = bool(
            getattr(features, "advanced_lidar_available", False)
            and not getattr(features, "advanced_lidar_stale", True)
        )
        if advanced_live:
            values.append(
                _optional_float(features, "advanced_lidar_front_vru_distance")
            )
            corridor_class = _text(
                features, "advanced_lidar_nearest_corridor_class"
            )
            if _is_vru(corridor_class):
                values.append(
                    _optional_float(
                        features, "advanced_lidar_nearest_corridor_distance"
                    )
                )
        values = [value for value in values if value is not None]
        return min(values) if values else None

    @staticmethod
    def _verified_lidar(features: Any) -> bool:
        """Return true only for a current point-cloud geometry packet."""

        advanced = bool(
            getattr(features, "advanced_lidar_available", False)
            and not getattr(features, "advanced_lidar_stale", True)
        )
        fallback = bool(
            getattr(features, "lidar_available", False)
            and not getattr(features, "lidar_stale", True)
        )
        return advanced or fallback

    @staticmethod
    def _open_side_is_clear(features: Any, open_side: str) -> bool:
        if open_side not in {"left", "right"}:
            return False
        if not bool(getattr(features, "%s_clear" % open_side, True)):
            return False
        blockage = _optional_float(
            features, "lidar_%s_blockage_ratio" % open_side
        )
        return blockage is None or blockage < 0.68

    @staticmethod
    def _static_blockage(features: Any) -> Tuple[Optional[float], str, bool]:
        """Return distance, clear-side label, and explicit-static evidence."""

        front_obstacle = _optional_float(features, "front_obstacle_distance")
        front_vehicle = _optional_float(features, "front_vehicle_distance")
        distances = []
        explicit_static = False

        if front_obstacle is not None:
            distances.append(front_obstacle)
            # An obstacle without a tracked lead vehicle is the useful,
            # observable distinction from ordinary moving traffic.
            explicit_static = front_vehicle is None

        if bool(getattr(features, "disabled_static_vehicle_evidence", False)):
            explicit_static = True
            if front_vehicle is not None:
                distances.append(front_vehicle)

        for obj in list(getattr(features, "detector_objects", []) or []) + list(
            getattr(features, "tracked_objects", []) or []
        ):
            name = _class(obj)
            if name not in {
                "traffic_cone",
                "barrier",
                "construction",
                "debris",
                "roadwork",
            }:
                continue
            x, y = _object_xy(obj)
            try:
                score = float(
                    obj.get("score", obj.get("confidence", 0.0)) or 0.0
                )
            except (TypeError, ValueError):
                score = 0.0
            if 0.0 <= x <= 15.0 and abs(y) <= 3.2 and score >= 0.20:
                distances.append(x)
                explicit_static = True

        if _float(features, "construction_object_count", 0.0) > 0:
            explicit_static = True

        distances = [value for value in distances if value is not None]
        distance = min(distances) if distances else None

        left_clear = bool(getattr(features, "left_clear", True))
        right_clear = bool(getattr(features, "right_clear", True))
        open_side = _text(features, "advanced_lidar_semantic_open_side")
        if open_side not in {"left", "right"}:
            open_side = _text(features, "lidar_open_side")
        if open_side not in {"left", "right"}:
            if left_clear and not right_clear:
                open_side = "left"
            elif right_clear and not left_clear:
                open_side = "right"
            else:
                open_side = "unknown"
        return distance, open_side, explicit_static

    def _set_category(self, category: str) -> None:
        if category == self.last_category:
            self.category_frames += 1
            return
        self.last_category = category
        self.category_frames = 1
        self.stopped_frames = 0
        self.blocked_frames = 0

    @staticmethod
    def _passthrough(
        category: str,
        reason: str,
        confidence: float = 0.55,
        state: str = "APPROACH",
    ) -> OrdinaryRuleDecision:
        return OrdinaryRuleDecision(
            category=category,
            confidence=confidence,
            state=state,
            active=False,
            reason=reason,
        )

    def decide(self, features: Any) -> OrdinaryRuleDecision:
        speed = abs(_float(features, "ego_speed"))
        red = bool(getattr(features, "red_light_active", False))
        traffic_kind = _text(features, "traffic_control_kind")
        front_vru = self._front_vru_distance(features)
        static_distance, open_side, explicit_static = self._static_blockage(
            features
        )

        # The online stop-sign/signal fields are not reliable enough to
        # replace the learned controller.  In particular, stop-sign rules
        # previously held genuine signalized routes at zero speed.  Preserve
        # the original no-extra-rule behaviour and let the existing safety
        # layer handle confirmed red/emergency hazards.
        if traffic_kind in {"stop_sign", "stopsign"}:
            category = "stop_controlled_intersection"
            self._set_category(category)
            return self._passthrough(
                category,
                "ordinary_traffic_control_passthrough",
                confidence=0.55,
            )
        if traffic_kind in {"traffic_light", "trafficlight"} or red:
            category = "signalized_intersection"
            self._set_category(category)
            return self._passthrough(
                category,
                "ordinary_traffic_control_passthrough",
                confidence=0.60 if red else 0.55,
                state="YIELD_OR_BRAKE" if red else "APPROACH",
            )

        if static_distance is not None:
            category = "static_blockage_or_construction"
            self._set_category(category)
            self.blocked_frames = (
                self.blocked_frames + 1 if speed <= 0.8 else 0
            )

            # Close/emergency geometry remains exclusively owned by the
            # pre-existing safety layer.  Ordinary rules never add a second
            # brake or steering command there.
            if static_distance <= 2.5:
                return self._passthrough(
                    category,
                    "ordinary_static_blockage_deferred_to_safety_layer",
                    confidence=0.70,
                    state="EMERGENCY",
                )

            # One intentionally narrow action: only a clearly static,
            # moderate-distance obstruction, current LiDAR, known-clear side,
            # no VRU, and a sustained low-speed stall.  Moving traffic,
            # high-speed approach, ambiguous side geometry, and short pauses
            # all remain learned-controller passthrough.
            recovery_ready = (
                explicit_static
                and 4.0 <= static_distance <= 9.0
                and speed <= 0.8
                and self.blocked_frames >= self.min_blocked_frames
                and self._verified_lidar(features)
                and self._open_side_is_clear(features, open_side)
                and open_side in {"left", "right"}
                and front_vru is None
                and not bool(getattr(features, "immediate_hazard", False))
            )
            if recovery_ready:
                bias = 0.12 if open_side == "right" else -0.12
                return OrdinaryRuleDecision(
                    category=category,
                    confidence=0.78,
                    state="AVOID_OR_PASS",
                    active=True,
                    target_speed=2.4,
                    throttle_cap=0.30,
                    throttle_floor=0.16,
                    brake_cap=0.35,
                    steer_limit=0.45,
                    steer_bias=bias,
                    steer_min_magnitude=0.12,
                    reason="ordinary_static_blockage_long_stop_recovery",
                )
            return self._passthrough(
                category,
                "ordinary_static_blockage_passthrough",
                confidence=0.62,
                state="AVOID_OR_PASS",
            )

        if front_vru is not None and front_vru <= 14.0:
            category = "vru_crossing"
            self._set_category(category)
            return self._passthrough(
                category,
                "ordinary_vru_passthrough",
                confidence=0.60,
            )

        if bool(getattr(features, "junction_like", False)) or _is_turn_command(
            _text(features, "route_command")
        ):
            category = "junction_flow_or_merge"
            self._set_category(category)
            return self._passthrough(
                category,
                "ordinary_junction_passthrough",
                confidence=0.55,
            )

        self._set_category("normal_cruise")
        self.clear_frames += 1
        return self._passthrough(
            "normal_cruise",
            "ordinary_learned_control_passthrough",
            confidence=0.35,
            state="NORMAL",
        )


# Audit-only mapping from XML scenario names to observable families.  It is
# intentionally retained for offline coverage checks and is never consulted
# by OrdinarySceneRuleEngine.
ORDINARY_XML_CATEGORY_FAMILY_MAP = {
    "Accident": "static_blockage_or_construction",
    "AccidentTwoWays": "static_blockage_or_construction",
    "BlockedIntersection": "junction_flow_or_merge",
    "ConstructionObstacle": "static_blockage_or_construction",
    "ConstructionObstacleTwoWays": "static_blockage_or_construction",
    "ControlLoss": "normal_cruise",
    "CrossingBicycleFlow": "vru_crossing",
    "DynamicObjectCrossing": "vru_crossing",
    "EnterActorFlow": "junction_flow_or_merge",
    "HardBreakRoute": "normal_cruise",
    "HazardAtSideLane": "static_blockage_or_construction",
    "HazardAtSideLaneTwoWays": "static_blockage_or_construction",
    "HighwayCutIn": "highway_merge_or_cut_in",
    "HighwayExit": "highway_merge_or_cut_in",
    "InterurbanActorFlow": "junction_flow_or_merge",
    "InterurbanAdvancedActorFlow": "junction_flow_or_merge",
    "InvadingTurn": "junction_flow_or_merge",
    "MergerIntoSlowTraffic": "highway_merge_or_cut_in",
    "MergerIntoSlowTrafficV2": "highway_merge_or_cut_in",
    "NonSignalizedJunctionLeftTurn": "junction_flow_or_merge",
    "NonSignalizedJunctionLeftTurnEnterFlow": "junction_flow_or_merge",
    "NonSignalizedJunctionRightTurn": "junction_flow_or_merge",
    "OppositeVehicleRunningRedLight": "signalized_intersection",
    "OppositeVehicleTakingPriority": "junction_flow_or_merge",
    "ParkedObstacle": "static_blockage_or_construction",
    "ParkedObstacleTwoWays": "static_blockage_or_construction",
    "ParkingCrossingPedestrian": "vru_crossing",
    "ParkingCutIn": "parking_low_speed_manoeuvre",
    "ParkingExit": "parking_low_speed_manoeuvre",
    "PedestrianCrossing": "vru_crossing",
    "SequentialLaneChange": "highway_merge_or_cut_in",
    "SignalizedJunctionLeftTurn": "signalized_intersection",
    "SignalizedJunctionLeftTurnEnterFlow": "signalized_intersection",
    "SignalizedJunctionRightTurn": "signalized_intersection",
    "StaticCutIn": "highway_merge_or_cut_in",
    "T_Junction": "junction_flow_or_merge",
    "VanillaNonSignalizedTurn": "junction_flow_or_merge",
    "VanillaNonSignalizedTurnEncounterStopsign": "stop_controlled_intersection",
    "VanillaSignalizedTurnEncounterGreenLight": "signalized_intersection",
    "VanillaSignalizedTurnEncounterRedLight": "signalized_intersection",
    "VehicleOpensDoorTwoWays": "static_blockage_or_construction",
    "VehicleTurningRoute": "junction_flow_or_merge",
    "VehicleTurningRoutePedestrian": "vru_crossing",
    "YieldToEmergencyVehicle": "junction_flow_or_merge",
}
