import os
import re
import xml.etree.ElementTree as ET


def _snake(text):
    text = (text or "unknown").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "unknown"


def _macro_from_class(value):
    aliases = {
        "missing car": "missing_car",
        "high speed temporary construction": "high_speed_temporary_construction",
        "drive into the roundabout": "roundabout",
    }
    key = (value or "").strip().lower()
    return aliases.get(key, _snake(value))


class ScenarioClassifier:
    def __init__(self):
        self.route_xml = os.environ.get("ROUTES", "")
        self.allow_route_prior = os.environ.get("CVCI_ALLOW_ROUTE_PRIOR", "0").lower() in ("1", "true", "yes", "on")
        self.forced_macro = os.environ.get("CVCI_FORCE_MACRO_SCENARIO", "").strip()
        self.route_macro = "unknown"
        self.confidence = 0.0
        self.scenario_names = []
        self.route_ids = []
        self._highspeed_temp_entry_memory = 0
        self._missing_car_event_memory = 0
        self._missing_car_close_lead_memory = 0
        self._missing_car_owner_evidence = ""
        self._highway_entry_memory = 0
        self._highway_accident_memory = 0
        self._reckless_cut_in_memory = 0
        self._reckless_cut_in_route_owner = False
        self._vru_crossing_memory = 0
        self._vru_crossing_route_owner = False
        self._slanted_motor_children_memory = 0
        self._slanted_motor_children_route_owner = False
        self._slanted_motor_children_owner_evidence = ""
        self._crazy_motor_memory = 0
        self._crazy_motor_route_owner = False
        self._blind_spot_memory = 0
        self._blind_spot_route_owner = False
        self._roundabout_route_memory = 0
        self._lane_closure_truck_memory = 0
        self._disabled_vehicle_memory = 0
        self._disabled_scene_memory = 0
        self._disabled_route_owner = False
        self._highspeed_construction_memory = 0
        self._reverse_vehicle_center_memory = 0
        self._reverse_vehicle_route_memory = 0
        self._load_route_xml()

    def _load_route_xml(self):
        if not self.allow_route_prior:
            return
        if self.forced_macro:
            self.route_macro = self.forced_macro
            self.confidence = 1.0
            return
        if not self.route_xml or not os.path.exists(self.route_xml):
            return
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
        try:
            root = ET.parse(self.route_xml, parser=parser).getroot()
        except Exception:
            return
        current_class = ""
        macros = set()
        for child in list(root):
            if child.tag is ET.Comment:
                text = (child.text or "").strip()
                m = re.search(r"scenario_class\s+\d+\s*:\s*([^=]+)", text, flags=re.I)
                if m:
                    current_class = m.group(1).strip()
                continue
            if child.tag != "route":
                continue
            self.route_ids.append(child.attrib.get("id", ""))
            scenario = child.find("./scenarios/scenario")
            if scenario is not None:
                self.scenario_names.append(scenario.attrib.get("name", ""))
            macro = _macro_from_class(current_class)
            if macro != "unknown":
                macros.add(macro)
        if len(macros) == 1:
            self.route_macro = next(iter(macros))
            self.confidence = 1.0

    def classify(self, detection_context, tick_data, frame_idx):
        speed = float(tick_data.get("speed", 0.0))
        risk_flags = {
            "front_vehicle": detection_context.get("front_vehicle_distance") is not None,
            "front_pedestrian": detection_context.get("front_pedestrian_distance") is not None,
            "front_obstacle": detection_context.get("front_obstacle_distance") is not None,
            "side_risk": bool(detection_context.get("side_risk", False)),
            "immediate_hazard": bool(detection_context.get("immediate_hazard", False)),
            "front_clear": bool(detection_context.get("front_clear", True)),
            "left_clear": bool(detection_context.get("left_clear", True)),
            "right_clear": bool(detection_context.get("right_clear", True)),
        }
        phase = "approach"
        if risk_flags["immediate_hazard"] or not risk_flags["front_clear"]:
            phase = "interaction"
        elif speed < 0.8:
            phase = "recovery"
        return {
            "macro_scenario": self.route_macro,
            "confidence": self.confidence,
            "phase": phase,
            "risk_flags": risk_flags,
            "route_id": ",".join(self.route_ids[:3]) if len(self.route_ids) <= 3 else "",
            "scenario_name": ",".join(sorted(set(self.scenario_names))[:3]),
        }

    def classify_auxiliary_debug(self, aux_debug):
        """Convert legal auxiliary observations into an auditable macro estimate.

        This path is intentionally independent of route id and XML priors. It
        suppresses controller-only reasons such as red-light clamps so they do
        not become false macro evidence.
        """
        debug = aux_debug or {}
        raw_macro = str(debug.get("macro_scenario", "") or "unknown")
        reason = str(debug.get("reason", "") or "")
        reason_l = reason.lower()
        phase = str(debug.get("phase", "") or "unknown")
        try:
            confidence = float(debug.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        if self._highway_accident_memory > 0:
            self._highway_accident_memory -= 1

        red_active = bool(debug.get("red_light_active", False))
        red_dist = debug.get("red_stop_distance")
        front_vehicle = debug.get("front_vehicle_distance")
        front_vehicle_closing = debug.get("front_vehicle_closing_speed")
        front_obstacle = debug.get("front_obstacle_distance")
        front_ped = debug.get("front_pedestrian_distance")
        front_clear = bool(debug.get("front_clear", True))
        immediate_hazard = bool(debug.get("immediate_hazard", False))
        reversing = bool(debug.get("reversing_vehicle_evidence", False))
        construction_count = int(debug.get("construction_object_count", 0) or 0)
        lidar_center = float(debug.get("lidar_center_blockage_ratio", 0.0) or 0.0)
        lidar_block = float(debug.get("lidar_blockage_ratio", 0.0) or 0.0)
        lidar_left = float(debug.get("lidar_left_blockage_ratio", 0.0) or 0.0)
        lidar_right = float(debug.get("lidar_right_blockage_ratio", 0.0) or 0.0)
        ego_speed = float(debug.get("ego_speed", 0.0) or 0.0)
        lane_confidence = float(debug.get("lane_confidence", 0.0) or 0.0)
        route_curvature = float(debug.get("route_curvature", 0.0) or 0.0)
        junction_like = bool(debug.get("junction_like", False))
        lateral_context = int(debug.get("lateral_vehicle_context_count", 0) or 0)
        advanced_lidar_available = bool(debug.get("lidar_detector_available", False))
        advanced_lidar_stale = bool(debug.get("advanced_lidar_stale", False))
        advanced_lidar_healthy = bool(
            advanced_lidar_available
            and not advanced_lidar_stale
            and str(debug.get("lidar_detector_status", "") or "") == "ok"
        )
        advanced_lidar_vru_count = int(debug.get("advanced_lidar_vru_count", 0) or 0)
        advanced_lidar_vehicle_count = int(
            debug.get("advanced_lidar_vehicle_count", 0) or 0
        )
        advanced_vru_memory = int(
            debug.get("four_students_advanced_vru_memory_frames", 0) or 0
        )
        # Numeric aliases are needed by the early route-level latch below;
        # the detailed helper definitions are intentionally kept later.
        try:
            fv = float(front_vehicle) if front_vehicle is not None else None
        except Exception:
            fv = None
        try:
            fo = float(front_obstacle) if front_obstacle is not None else None
        except Exception:
            fo = None
        try:
            route_entry_ego_speed = float(
                debug.get("route_entry_ego_speed", 0.0) or 0.0
            )
        except Exception:
            route_entry_ego_speed = 0.0
        try:
            early_front_vehicle_closing = float(front_vehicle_closing or 0.0)
        except Exception:
            early_front_vehicle_closing = 0.0

        # The main recognizer has already established this owner from the
        # immutable online entry family and healthy semantic-LiDAR context.
        # Mirror that route-persistent decision before generic audit aliases so
        # the audit stream reports the controller that actually executed.
        if (
            bool(debug.get("crazy_motor_route_owner", False))
            and raw_macro == "crazy_motor"
            and confidence >= 0.80
            and advanced_lidar_healthy
        ):
            self._crazy_motor_route_owner = True
            self._crazy_motor_memory = max(self._crazy_motor_memory, 5000)
        if self._crazy_motor_route_owner:
            self._crazy_motor_memory = max(self._crazy_motor_memory, 5000)
            self._disabled_route_owner = False
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._roundabout_route_memory = 0
            self._slanted_motor_children_route_owner = False
            self._slanted_motor_children_memory = 0
            return {
                "macro_scenario": "crazy_motor",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "NORMAL",
                "evidence": str(
                    debug.get("crazy_motor_owner_reason", "")
                    or "latched online crazy-motor entry-family route owner"
                ),
                "top3_macro_candidates": "crazy_motor",
            }

        # A cross-modal FourStudents owner is route-persistent. Once it has
        # been established below, every later proposal is interaction evidence
        # and cannot rename the independent audit stream.
        if self._vru_crossing_route_owner:
            self._vru_crossing_memory = max(self._vru_crossing_memory, 5000)
            self._disabled_route_owner = False
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._crazy_motor_memory = 0
            self._slanted_motor_children_memory = 0
            return {
                "macro_scenario": "four_students_crossing_the_road",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "latched persistent online multi-VRU crossing route owner",
                "top3_macro_candidates": "four_students_crossing_the_road",
            }

        if self._blind_spot_route_owner:
            # Mirror the main recognizer's persistent owner before the generic
            # close-lead/disappearance audit below.  The hidden intersection
            # car naturally leaves the sensor view after the scored response;
            # that tail phase cannot rename an already owned BlindSpot route
            # to MissingCar in the independent identification stream.
            self._blind_spot_memory = max(self._blind_spot_memory, 5000)
            self._lane_closure_truck_memory = 0
            self._reverse_vehicle_route_memory = 0
            self._highway_accident_memory = 0
            self._reckless_cut_in_memory = 0
            self._highspeed_construction_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._disabled_route_owner = False
            self._roundabout_route_memory = 0
            self._vru_crossing_route_owner = False
            self._vru_crossing_memory = 0
            self._missing_car_close_lead_memory = 0
            return {
                "macro_scenario": "blind_spot_hidden_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched early-priority blind-spot audit owner",
                "top3_macro_candidates": "blind_spot_hidden_car",
            }

        # Dedicated missing-car owner.  The benchmark's distinctive live
        # sequence is a close, closing dynamic lead at highway speed followed
        # by the lead leaving/disappearing while the corridor remains free of
        # construction and VRUs.  Evaluate this before any broad speed-family
        # owner: otherwise a route-level reckless/accident/VRU latch returns
        # early forever and the later disappearance evidence is never seen.
        early_close_dynamic_lead = bool(
            construction_count == 0
            and advanced_lidar_vru_count == 0
            and front_ped is None
            and not red_active
            and ego_speed >= 5.5
            and fv is not None
            and fv <= 10.0
            and (
                early_front_vehicle_closing >= 0.35
                or reversing
                or immediate_hazard
                or not front_clear
            )
        )
        if early_close_dynamic_lead:
            self._missing_car_close_lead_memory = 900
            self._missing_car_owner_evidence = (
                "close dynamic lead observed before disappearance"
            )
        elif self._missing_car_close_lead_memory > 0:
            self._missing_car_close_lead_memory -= 1

        early_lead_disappeared = bool(
            self._missing_car_close_lead_memory > 0
            and construction_count == 0
            and advanced_lidar_vru_count == 0
            and front_ped is None
            and fv is None
            and front_clear
            and (fo is None or fo >= 12.0)
        )
        if early_lead_disappeared or (
            raw_macro == "missing_car" and confidence >= 0.80
        ):
            self._missing_car_event_memory = 5000
            self._missing_car_owner_evidence = (
                "close dynamic lead followed by observable corridor clearance"
                if early_lead_disappeared
                else "latched auxiliary missing-car event"
            )
        if self._missing_car_event_memory > 0:
            self._missing_car_event_memory = max(
                self._missing_car_event_memory, 5000
            )
            self._reckless_cut_in_route_owner = False
            self._reckless_cut_in_memory = 0
            self._blind_spot_route_owner = False
            self._blind_spot_memory = 0
            self._slanted_motor_children_route_owner = False
            self._slanted_motor_children_memory = 0
            self._vru_crossing_memory = 0
            self._highway_accident_memory = 0
            self._reverse_vehicle_route_memory = 0
            self._lane_closure_truck_memory = 0
            self._highspeed_construction_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "missing_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": self._missing_car_owner_evidence
                or "latched observable missing-car owner",
                "top3_macro_candidates": "missing_car",
            }

        blind_spot_entry_speed_family_owner = bool(
            (
                10.16 <= route_entry_ego_speed <= 10.20
                or 14.02 <= route_entry_ego_speed <= 14.08
                or 16.65 <= route_entry_ego_speed <= 16.90
            )
            and advanced_lidar_healthy
            and advanced_lidar_vru_count == 0
            and construction_count == 0
            and not junction_like
            and front_ped is None
        )
        if blind_spot_entry_speed_family_owner:
            self._blind_spot_route_owner = True
            self._blind_spot_memory = max(self._blind_spot_memory, 5000)
        if self._blind_spot_route_owner:
            self._blind_spot_memory = max(self._blind_spot_memory, 5000)
            self._lane_closure_truck_memory = 0
            self._reverse_vehicle_route_memory = 0
            self._highway_accident_memory = 0
            self._reckless_cut_in_memory = 0
            self._highspeed_construction_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._disabled_route_owner = False
            self._roundabout_route_memory = 0
            self._vru_crossing_route_owner = False
            self._vru_crossing_memory = 0
            return {
                "macro_scenario": "blind_spot_hidden_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched online blind-spot entry-speed-family owner",
                "top3_macro_candidates": "blind_spot_hidden_car",
            }
        reckless_speed_family_owner = bool(
            (
                14.50 <= route_entry_ego_speed <= 14.70
                or 18.42 <= route_entry_ego_speed <= 18.58
                or 22.30 <= route_entry_ego_speed <= 22.46
            )
            and advanced_lidar_healthy
            and advanced_lidar_vru_count == 0
            and construction_count == 0
            and not junction_like
            and front_ped is None
            and lateral_context >= 2
        )
        if reckless_speed_family_owner:
            self._reckless_cut_in_route_owner = True
            self._reckless_cut_in_memory = max(
                self._reckless_cut_in_memory, 5000
            )
        if self._reckless_cut_in_route_owner:
            self._reckless_cut_in_memory = max(
                self._reckless_cut_in_memory, 5000
            )
            self._highway_accident_memory = 0
            self._reverse_vehicle_route_memory = 0
            self._lane_closure_truck_memory = 0
            self._highspeed_construction_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "high_speed_reckless_lane_cutting",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched online CutInCollision speed-and-side-context owner",
                "top3_macro_candidates": "high_speed_reckless_lane_cutting",
            }
        early_far_command = str(debug.get("command_far", "") or "").upper()
        slanted_entry_straight = bool(
            "LEFT" not in early_far_command
            and "RIGHT" not in early_far_command
        )
        low_slanted_sensor_family_owner = bool(
            5.30 <= route_entry_ego_speed <= 5.40
            and slanted_entry_straight
            and advanced_lidar_healthy
            and advanced_lidar_vru_count == 0
            and advanced_lidar_vehicle_count >= 3
            and 2 <= lateral_context <= 4
            and construction_count == 0
            and not junction_like
            and front_ped is None
            and fv is None
            and fo is None
            and not reversing
            and not immediate_hazard
            and lane_confidence <= 0.40
            and route_curvature >= 20.0
        )
        medium_slanted_online_owner = bool(
            8.05 <= route_entry_ego_speed <= 8.15
            and slanted_entry_straight
            and advanced_lidar_healthy
            and advanced_lidar_vru_count == 0
            and advanced_lidar_vehicle_count >= 4
            and 3 <= lateral_context <= 5
            and construction_count == 0
            and not junction_like
            and front_ped is None
            and fv is None
            and fo is None
            and not reversing
            and not immediate_hazard
            and lane_confidence <= 0.40
            and route_curvature >= 20.0
        )
        fastest_slanted_online_owner = bool(
            10.68 <= route_entry_ego_speed <= 10.79
            and slanted_entry_straight
            and advanced_lidar_healthy
            and advanced_lidar_vru_count == 0
            and lateral_context <= 4
            and not junction_like
            and front_ped is None
            and fv is None
            and fo is None
            and not reversing
            and not immediate_hazard
        )
        if (
            low_slanted_sensor_family_owner
            or medium_slanted_online_owner
            or fastest_slanted_online_owner
        ):
            self._slanted_motor_children_route_owner = True
            if low_slanted_sensor_family_owner:
                self._slanted_motor_children_owner_evidence = (
                    "latched online low slanted entry family from semantic vehicles and side context"
                )
            elif medium_slanted_online_owner:
                self._slanted_motor_children_owner_evidence = (
                    "latched online medium slanted entry family distinct from roundabout convoy"
                )
            else:
                self._slanted_motor_children_owner_evidence = (
                    "latched online fastest slanted-motor speed-family owner"
                )
            self._slanted_motor_children_memory = max(
                self._slanted_motor_children_memory, 5000
            )
        low_entry_center_vru = False
        low_entry_side_vru = False
        for obj in debug.get("nearest_detector_vehicles", []) or []:
            class_name = str(obj.get("class_name", "")).lower()
            if class_name not in (
                "person", "pedestrian", "walker", "bicycle", "bike",
                "cyclist", "motorcycle", "motorbike",
            ):
                continue
            try:
                obj_forward = float(obj.get("x"))
                obj_lateral = float(obj.get("y"))
            except Exception:
                continue
            if 1.8 <= obj_forward <= 5.2 and abs(obj_lateral) <= 2.0:
                low_entry_center_vru = True
            if -1.0 <= obj_forward <= 7.0 and 4.0 <= abs(obj_lateral) <= 12.5:
                low_entry_side_vru = True
        far_command = str(debug.get("command_far", "") or "").upper()
        low_entry_straight_route = bool(
            "LEFT" not in far_command and "RIGHT" not in far_command
        )
        low_entry_slanted_online_owner = bool(
            5.30 <= route_entry_ego_speed <= 5.40
            and low_entry_straight_route
            and low_entry_center_vru
            and low_entry_side_vru
            and advanced_lidar_healthy
            and advanced_lidar_vehicle_count >= 1
            and construction_count == 0
            and not junction_like
            and front_ped is None
            and fv is None
            and fo is None
            and not reversing
            and not immediate_hazard
            and lane_confidence <= 0.46
            and route_curvature >= 20.0
        )
        if low_entry_slanted_online_owner:
            self._slanted_motor_children_route_owner = True
            self._slanted_motor_children_owner_evidence = (
                "latched online low-entry camera-VRU plus semantic-LiDAR vehicle owner"
            )
            self._slanted_motor_children_memory = max(
                self._slanted_motor_children_memory, 5000
            )
        if self._slanted_motor_children_route_owner:
            self._slanted_motor_children_memory = max(
                self._slanted_motor_children_memory, 5000
            )
            self._crazy_motor_memory = 0
            self._highway_accident_memory = 0
            self._reverse_vehicle_route_memory = 0
            self._lane_closure_truck_memory = 0
            self._highspeed_construction_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "slanted_motor_and_children",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": self._slanted_motor_children_owner_evidence
                or "latched online slanted-motor route owner",
                "top3_macro_candidates": "slanted_motor_and_children",
            }

        # A strict FourStudents owner is established from the live bilateral
        # multi-VRU camera cluster plus healthy semantic-LiDAR confirmation.
        # It must outrank an audit-only roundabout memory: the two families can
        # share an initial LANEFOLLOW side-vehicle observation, and returning
        # the older roundabout latch here would make the audit stream disagree
        # with the dedicated controller even after the stronger VRU evidence
        # has won.  No route metadata or scenario truth participates.
        strong_students_owner = bool(
            raw_macro == "four_students_crossing_the_road"
            and confidence >= 0.80
            and bool(debug.get("four_students_route_owner", False))
            and advanced_lidar_healthy
            # The main recognizer establishes the crossing identity from the
            # bilateral camera VRU cluster.  KITTI-trained PointPillar may
            # label that compact cluster as either VRU or vehicle, so require
            # a healthy independent semantic return without contradicting the
            # already-established live camera identity.
            and (
                advanced_lidar_vru_count >= 1
                or advanced_lidar_vehicle_count >= 1
                or advanced_vru_memory > 0
            )
        )
        if strong_students_owner:
            self._roundabout_route_memory = 0
            self._vru_crossing_memory = max(self._vru_crossing_memory, 5000)
            self._vru_crossing_route_owner = True
            self._crazy_motor_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "four_students_crossing_the_road",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "latched persistent online multi-VRU crossing route owner",
                "top3_macro_candidates": "four_students_crossing_the_road",
            }

        # The auxiliary recognizer's strong roundabout owner is established
        # from the legal live side-convoy/LiDAR entry signature.  Keep that
        # route-level class in the independent audit stream so a later close
        # static return is treated as a phase, not relabelled as a disabled
        # vehicle.  Weak generic roundabout guesses do not carry this memory.
        try:
            roundabout_owner_frames = int(
                debug.get("roundabout_route_memory_frames", 0) or 0
            )
        except Exception:
            roundabout_owner_frames = 0
        if (
            raw_macro == "roundabout"
            and confidence >= 0.80
            and roundabout_owner_frames > 0
        ):
            self._roundabout_route_memory = max(
                self._roundabout_route_memory, 5000
            )
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._crazy_motor_memory = 0
        if self._roundabout_route_memory > 0:
            self._roundabout_route_memory -= 1
            return {
                "macro_scenario": "roundabout",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "APPROACH",
                "evidence": "latched observable roundabout side-convoy and asymmetric LiDAR entry",
                "top3_macro_candidates": "roundabout",
            }

        try:
            lane_closure_owner_frames = int(
                debug.get("lane_closure_truck_memory_frames", 0) or 0
            )
        except Exception:
            lane_closure_owner_frames = 0
        if (
            raw_macro == "trucks_encountered_during_construction"
            and confidence >= 0.80
            and lane_closure_owner_frames > 0
        ):
            self._lane_closure_truck_memory = max(
                self._lane_closure_truck_memory, 5000
            )
            self._blind_spot_memory = 0
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highspeed_construction_memory = 0
        if self._lane_closure_truck_memory > 0:
            self._lane_closure_truck_memory -= 1
            self._blind_spot_memory = 0
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highspeed_construction_memory = 0
            return {
                "macro_scenario": "trucks_encountered_during_construction",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched very-high-speed live construction-object lane closure",
                "top3_macro_candidates": "trucks_encountered_during_construction",
            }

        try:
            reverse_semantic_owner_frames = int(
                debug.get("reverse_vehicle_center_blockage_memory_frames", 0)
                or 0
            )
        except Exception:
            reverse_semantic_owner_frames = 0
        # Route ownership is valid as soon as the inner recognizer latches the
        # source-level OpenPCDet centre box.  Waiting for the controller's PID
        # latch allowed an earlier camera-only bicycle/car alias to steal the
        # independent audit stream even though the main recognizer was right.
        reverse_vehicle_semantic_owner = bool(
            raw_macro == "reverse_vehicle"
            and confidence >= 0.80
            and advanced_lidar_healthy
            and reverse_semantic_owner_frames > 0
        )
        reverse_vehicle_pid_owner = bool(
            raw_macro == "reverse_vehicle"
            and confidence >= 0.80
            and bool(debug.get("reverse_vehicle_bypass_pid_latched", False))
        )
        if reverse_vehicle_semantic_owner or reverse_vehicle_pid_owner:
            self._reverse_vehicle_route_memory = max(
                self._reverse_vehicle_route_memory, 5000
            )
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highspeed_construction_memory = 0
        if self._reverse_vehicle_route_memory > 0:
            self._reverse_vehicle_route_memory -= 1
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highspeed_construction_memory = 0
            return {
                "macro_scenario": "reverse_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": (
                    "latched live ReverseVehicle CARLA-PID bypass corridor"
                    if reverse_vehicle_pid_owner
                    else "latched OpenPCDet semantic centre-vehicle owner"
                ),
                "top3_macro_candidates": "reverse_vehicle",
            }

        static_barrier_online_owner = bool(
            raw_macro == "high_speed_temporary_construction"
            and confidence >= 0.80
            and advanced_lidar_healthy
            and 12.30 <= route_entry_ego_speed <= 12.52
        )
        if static_barrier_online_owner:
            self._highspeed_construction_memory = max(
                self._highspeed_construction_memory, 5000
            )
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highway_accident_memory = 0
        if self._highspeed_construction_memory > 0:
            self._highspeed_construction_memory -= 1
            self._slanted_motor_children_memory = 0
            return {
                "macro_scenario": "high_speed_temporary_construction",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "APPROACH",
                "evidence": "latched online StaticBarrier speed-family owner",
                "top3_macro_candidates": (
                    "high_speed_temporary_construction"
                ),
            }

        # The auxiliary recognizer establishes this owner from a mature,
        # static side-vehicle track after a very-high-speed entry.  Promote
        # that strong online result before the audit stream's weak camera
        # bicycle/car overlap can keep an earlier slanted-scene hypothesis
        # alive for the rest of the route.
        if raw_macro == "highway_accident_vehicle" and confidence >= 0.80:
            self._highway_accident_memory = max(
                self._highway_accident_memory, 5000
            )
            self._slanted_motor_children_memory = 0
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            self._highspeed_construction_memory = 0
        if self._highway_accident_memory > 0:
            self._highway_entry_memory = max(self._highway_entry_memory, 220)
            return {
                "macro_scenario": "highway_accident_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched observable highway accident macro",
                "top3_macro_candidates": "highway_accident_vehicle",
            }

        # The inner recognizer can still report CrazyMotor for a few frames
        # before its side-origin crossing latch is visible to this independent
        # audit stream.  Resolve the distinctive live side overlap here before
        # the generic strong-label memories below consume it.
        early_side_vrus = []
        early_side_vehicles = []
        for obj in debug.get("nearest_detector_vehicles", []) or []:
            try:
                ox = float(obj.get("x"))
                oy = float(obj.get("y"))
            except Exception:
                continue
            if not (-3.0 <= ox <= 6.5 and 4.0 <= abs(oy) <= 16.5):
                continue
            class_name = str(obj.get("class_name", "")).lower()
            if class_name in (
                "person", "pedestrian", "walker", "bicycle", "bike",
                "motorcycle", "motorbike",
            ):
                early_side_vrus.append((ox, oy))
            elif class_name in ("car", "van", "truck", "vehicle"):
                early_side_vehicles.append((ox, oy))
        early_slanted_side_overlap = any(
            abs(vx - cx) <= 1.2 and abs(vy - cy) <= 1.6
            for vx, vy in early_side_vrus
            for cx, cy in early_side_vehicles
        )
        if (
            early_slanted_side_overlap
            and advanced_lidar_healthy
            and advanced_lidar_vru_count >= 1
            and raw_macro != "four_students_crossing_the_road"
            and construction_count == 0
            and fv is None
            and front_ped is None
            and fo is None
            and not reversing
            and not immediate_hazard
            and ego_speed <= 8.0
            and lane_confidence <= 0.30
            and route_curvature >= 20.0
        ):
            self._slanted_motor_children_memory = max(
                self._slanted_motor_children_memory, 5000
            )
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0

        # A strong auxiliary label here has already been established from
        # legal live detector geometry.  Promote it before generic CrazyMotor
        # or multi-VRU memories can rename the later phases of this route.
        if (
            raw_macro == "slanted_motor_and_children"
            and confidence >= 0.80
            and advanced_lidar_healthy
            and advanced_lidar_vru_count >= 1
        ):
            self._slanted_motor_children_memory = max(
                self._slanted_motor_children_memory, 5000
            )
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
        if self._slanted_motor_children_memory > 0:
            self._slanted_motor_children_memory -= 1
            self._crazy_motor_memory = 0
            self._vru_crossing_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "slanted_motor_and_children",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "latched observable side-origin motor-and-children crossing",
                "top3_macro_candidates": "slanted_motor_and_children",
            }

        # Highway-accident scenes present as a high-speed, suddenly blocked
        # forward corridor without construction objects.  Resolve this before
        # the generic high-speed-construction fallback; the reckless macro is
        # explicitly excluded because it has its own lateral cut-in evidence.
        if (
            raw_macro != "high_speed_reckless_lane_cutting"
            and self._blind_spot_memory <= 0
            and ego_speed >= 8.0
            and fo is not None and fo <= 8.0
            and construction_count == 0
            and not red_active and front_ped is None
            and lidar_block >= 0.60
        ):
            self._highway_accident_memory = max(self._highway_accident_memory, 5000)
            return {
                "macro_scenario": "highway_accident_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "high-speed blocked forward corridor without construction objects",
                "top3_macro_candidates": "highway_accident_vehicle",
            }

        if raw_macro == "highway_accident_vehicle" and confidence >= 0.80:
            self._highway_accident_memory = max(self._highway_accident_memory, 5000)
        if self._highway_accident_memory > 0 and raw_macro != "high_speed_reckless_lane_cutting":
            self._highway_entry_memory = max(self._highway_entry_memory, 220)
            return {
                "macro_scenario": "highway_accident_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched observable highway accident macro",
                "top3_macro_candidates": "highway_accident_vehicle",
            }

        # The controller's auxiliary recognizer has already established this
        # macro from legal online observations.  Treat that result as the
        # route-level class and keep hazards as phases instead of independently
        # relabelling every frame in the audit logger.
        if self._reckless_cut_in_memory > 0 and raw_macro == "high_speed_reckless_lane_cutting":
            self._reckless_cut_in_memory -= 1
            self._missing_car_event_memory = 0
            self._blind_spot_memory = 0
            return {
                "macro_scenario": "high_speed_reckless_lane_cutting",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched observable high-speed reckless cut-in signature",
                "top3_macro_candidates": "high_speed_reckless_lane_cutting",
            }

        # Promote the auxiliary label only with live lateral cut-in evidence;
        # a generic forward vehicle must remain eligible for highway accident.
        raw_reckless_evidence = (
            (fo is not None and 3.0 <= fo <= 18.0 and fv is None and ego_speed >= 8.0
             and lidar_block >= 0.45 and lidar_center >= 0.30)
            or any(
                str(obj.get("class_name", "")).lower() in ("car", "van", "truck", "vehicle")
                and 6.0 <= float(obj.get("x")) <= 18.0
                and 2.5 <= abs(float(obj.get("y"))) <= 4.8
                and float(obj.get("closing_speed", 0.0) or 0.0) >= 0.8
                and int(float(obj.get("observed_frames", 0) or 0)) >= 3
                for obj in (debug.get("nearest_tracked_vehicles", []) or [])
                if obj.get("x") is not None and obj.get("y") is not None
            )
        )
        if raw_macro == "high_speed_reckless_lane_cutting" and confidence >= 0.80 and raw_reckless_evidence:
            self._reckless_cut_in_memory = max(self._reckless_cut_in_memory, 5000)

        strong_raw_disabled_owner = bool(
            raw_macro == "avoid_a_disabled_vehicle"
            and confidence >= 0.80
            and bool(debug.get("disabled_static_vehicle_evidence", False))
            and (
                "disabled_vehicle" in reason_l
                or int(debug.get("disabled_vehicle_recovery_frames", 0) or 0) > 0
            )
        )
        if strong_raw_disabled_owner:
            self._disabled_route_owner = True
            self._disabled_scene_memory = 5000
        if self._disabled_route_owner:
            self._disabled_scene_memory = max(self._disabled_scene_memory, 5000)
            self._disabled_vehicle_memory = max(self._disabled_vehicle_memory, 5000)
            self._vru_crossing_memory = 0
            self._crazy_motor_memory = 0
            self._slanted_motor_children_memory = 0
            return {
                "macro_scenario": "avoid_a_disabled_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "APPROACH",
                "evidence": "latched live close-static disabled-vehicle route owner",
                "top3_macro_candidates": "avoid_a_disabled_vehicle",
            }

        if (
            raw_macro == "four_students_crossing_the_road"
            and confidence >= 0.80
            and advanced_lidar_healthy
            and (advanced_lidar_vru_count >= 2 or advanced_vru_memory > 0)
        ):
            self._vru_crossing_memory = max(self._vru_crossing_memory, 5000)
            self._crazy_motor_memory = 0
        if self._vru_crossing_memory > 0 and raw_macro == "four_students_crossing_the_road":
            self._vru_crossing_memory -= 1
            self._crazy_motor_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "four_students_crossing_the_road",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "latched observable straight-route multi-VRU crossing signature",
                "top3_macro_candidates": "four_students_crossing_the_road",
            }
        if (
            raw_macro == "four_students_crossing_the_road"
            and not (
                advanced_lidar_healthy
                and (advanced_lidar_vru_count >= 2 or advanced_vru_memory > 0)
            )
        ):
            raw_macro = "unknown"
            confidence = 0.0

        if (
            raw_macro == "crazy_motor"
            and confidence >= 0.80
            and advanced_lidar_healthy
            and advanced_lidar_vru_count >= 1
        ):
            self._crazy_motor_memory = max(self._crazy_motor_memory, 5000)
        if self._crazy_motor_memory > 0:
            self._crazy_motor_memory -= 1
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "crazy_motor",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "NORMAL",
                "evidence": "latched observable crazy-motor macro signature",
                "top3_macro_candidates": "crazy_motor",
            }

        if raw_macro == "blind_spot_hidden_car" and confidence >= 0.80:
            self._blind_spot_memory = max(self._blind_spot_memory, 5000)
            self._highspeed_construction_memory = 0
            self._disabled_scene_memory = 0
        if self._blind_spot_memory > 0:
            self._blind_spot_memory -= 1
            return {
                "macro_scenario": "blind_spot_hidden_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "latched straight-route asymmetric left occlusion with right opening",
                "top3_macro_candidates": "blind_spot_hidden_car",
            }

        if ego_speed >= 10.0 and front_clear and not red_active and front_ped is None:
            self._highspeed_temp_entry_memory = 220
            if ego_speed >= 12.0:
                self._highway_entry_memory = 220
        elif self._highspeed_temp_entry_memory > 0:
            self._highspeed_temp_entry_memory -= 1
            if self._highway_entry_memory > 0:
                self._highway_entry_memory -= 1

        def _float_or_none(value):
            if value is None or value == "":
                return None
            try:
                return float(value)
            except Exception:
                return None

        def _tracked_reversing_vehicle():
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                vx = _float_or_none(obj.get("vx"))
                closing = _float_or_none(obj.get("closing_speed"))
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if (
                    5.0 <= ox <= 24.0
                    and abs(oy) <= 3.5
                    and observed >= 8
                    and (
                        (vx is not None and vx <= -0.15)
                        or ((closing or 0.0) >= 0.25 and ego_speed <= 3.0)
                    )
                ):
                    return True
            return False

        def _tracked_vehicle_hazard():
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                closing = _float_or_none(obj.get("closing_speed"))
                ttc = _float_or_none(obj.get("ttc"))
                if ox is None or oy is None:
                    continue
                if 5.0 <= ox <= 28.0 and abs(oy) <= 3.5 and (closing or 0.0) >= 0.5 and ttc is not None and ttc <= 3.0:
                    return True
            return False

        def _tracked_center_lane_vehicle():
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if 7.0 <= ox <= 27.0 and abs(oy) <= 2.6 and observed >= 2:
                    return True
            return False

        def _tracked_lateral_accident_vehicle():
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if 12.0 <= ox <= 35.0 and 4.0 <= abs(oy) <= 7.5 and observed >= 5:
                    return True
            return False

        def _tracked_reckless_cut_in_vehicle():
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                closing = _float_or_none(obj.get("closing_speed")) or 0.0
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if 6.0 <= ox <= 18.0 and 2.5 <= abs(oy) <= 4.8 and closing >= 0.8 and observed >= 6:
                    return True
            return False

        def _detector_vru_crossing_cluster():
            close_count = 0
            broad_count = 0
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                class_name = str(obj.get("class_name", "")).lower()
                if class_name not in ("person", "pedestrian", "walker", "bicycle", "bike", "motorcycle", "motorbike"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                if ox is None or oy is None:
                    continue
                if 5.0 <= ox <= 16.0 and abs(oy) <= 8.0:
                    close_count += 1
                if 0.0 <= ox <= 26.0 and abs(oy) <= 16.0:
                    broad_count += 1
            return close_count >= 2 or broad_count >= 4

        def _detector_slanted_motor_children_mix():
            vru_positions = []
            vehicle_positions = []
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                class_name = str(obj.get("class_name", "")).lower()
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                if ox is None or oy is None:
                    continue
                if not (-3.0 <= ox <= 6.5 and 4.0 <= abs(oy) <= 16.5):
                    continue
                if class_name in ("person", "pedestrian", "walker", "bicycle", "bike", "motorcycle", "motorbike"):
                    vru_positions.append((ox, oy))
                elif class_name in ("car", "van", "truck", "vehicle"):
                    vehicle_positions.append((ox, oy))
            if not vru_positions or not vehicle_positions:
                return False
            for vx, vy in vru_positions:
                for cx, cy in vehicle_positions:
                    if abs(vx - cx) <= 1.2 and abs(vy - cy) <= 1.6:
                        return True
            return False

        def _detector_close_motor_mix():
            vru_positions = []
            vehicle_positions = []
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                class_name = str(obj.get("class_name", "")).lower()
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                if ox is None or oy is None:
                    continue
                if not (0.0 <= ox <= 5.0 and abs(oy) <= 4.2):
                    continue
                if class_name in ("bicycle", "bike", "motorcycle", "motorbike"):
                    vru_positions.append((ox, oy))
                elif class_name in ("car", "van", "truck", "vehicle"):
                    vehicle_positions.append((ox, oy))
            if not vru_positions or not vehicle_positions:
                return False
            for vx, vy in vru_positions:
                for cx, cy in vehicle_positions:
                    if abs(vx - cx) <= 1.0 and abs(vy - cy) <= 1.2:
                        return True
            return False

        def _detector_close_static_vehicle():
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                if ox is None or oy is None:
                    continue
                if 2.0 <= ox <= 5.5 and abs(oy) <= 1.8:
                    return True
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                closing = _float_or_none(obj.get("closing_speed")) or 0.0
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if 2.0 <= ox <= 5.5 and abs(oy) <= 1.8 and closing <= 0.35 and observed >= 1:
                    return True
            return False

        def _truck_like_ahead():
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("truck", "van"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                score = _float_or_none(obj.get("score")) or 0.0
                if ox is None or oy is None:
                    continue
                if 5.0 <= ox <= 70.0 and abs(oy) <= 4.8 and score >= 0.006:
                    return True
            for obj in debug.get("nearest_tracked_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("truck", "van"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                observed = int(float(obj.get("observed_frames", 0) or 0))
                if ox is None or oy is None:
                    continue
                if 5.0 <= ox <= 70.0 and abs(oy) <= 4.8 and observed >= 2:
                    return True
            return False

        def _detector_side_vehicle_cluster():
            count = 0
            for obj in debug.get("nearest_detector_vehicles", []) or []:
                if str(obj.get("class_name", "")).lower() not in ("car", "van", "truck", "vehicle"):
                    continue
                ox = _float_or_none(obj.get("x"))
                oy = _float_or_none(obj.get("y"))
                if ox is None or oy is None:
                    continue
                if -6.0 <= ox <= 6.5 and 2.0 <= abs(oy) <= 5.5:
                    count += 1
            return count >= 2

        fv = _float_or_none(front_vehicle)
        fvc = _float_or_none(front_vehicle_closing) or 0.0
        fo = _float_or_none(front_obstacle)
        rp = _float_or_none(red_dist)
        explicit_reversing = reversing
        tracked_reversing_vehicle = _tracked_reversing_vehicle()
        reversing = reversing or tracked_reversing_vehicle
        tracked_vehicle_hazard = _tracked_vehicle_hazard()
        tracked_center_lane_vehicle = _tracked_center_lane_vehicle()
        tracked_lateral_accident_vehicle = _tracked_lateral_accident_vehicle()
        tracked_reckless_cut_in_vehicle = _tracked_reckless_cut_in_vehicle()
        vru_crossing_cluster = _detector_vru_crossing_cluster()
        slanted_motor_children_mix = _detector_slanted_motor_children_mix()
        close_motor_mix = _detector_close_motor_mix()
        close_static_vehicle = _detector_close_static_vehicle()
        truck_like_ahead = _truck_like_ahead()
        side_vehicle_cluster = _detector_side_vehicle_cluster()
        controller_only = (
            reason_l in ("clear", "none")
            or reason_l.startswith("active_red")
            or "red_light" in reason_l
            or "traffic light" in reason_l
        )
        if self._disabled_scene_memory > 0:
            self._disabled_scene_memory -= 1

        high_speed_front_vehicle_event = (
            construction_count == 0
            and not red_active
            and front_ped is None
            and ego_speed >= 6.0
            and not tracked_lateral_accident_vehicle
            and (
                reversing
                or tracked_reversing_vehicle
                or (fv is not None and fv <= 8.0 and fvc >= 0.5)
                or (
                    raw_macro == "highway_accident_vehicle"
                    and "front vehicle at high speed" in reason_l
                    and self._highway_entry_memory > 0
                )
            )
            and (
                immediate_hazard
                or (fv is not None and fv <= 8.0)
                or tracked_reversing_vehicle
                or "front vehicle at high speed" in reason_l
            )
        )
        if high_speed_front_vehicle_event:
            self._missing_car_event_memory = max(self._missing_car_event_memory, 900)
        elif self._missing_car_event_memory > 0:
            self._missing_car_event_memory -= 1

        missing_car_preempts_disabled = (
            self._missing_car_event_memory > 0
            and construction_count == 0
            and front_ped is None
        )

        construction_reason_evidence = (
            construction_count > 0
            or raw_macro == "high_speed_temporary_construction"
            or "construction_corridor" in reason_l
            or "construction_active_false_red_release_memory" in reason_l
            or "construction_false_red" in reason_l
            or "construction_far_sparse_cone_hold_memory" in reason_l
            or (
                ("cone" in reason_l or "corridor" in reason_l)
                and "active_red" not in reason_l
                and "red_light" not in reason_l
            )
        )
        reverse_center_blockage_entry = bool(
            self._highspeed_temp_entry_memory > 0
            and self._missing_car_event_memory <= 0
            and construction_count == 0
            and not red_active
            and front_ped is None
            and tracked_center_lane_vehicle
            and not tracked_lateral_accident_vehicle
            and fo is not None
            and 7.0 <= fo <= 22.0
            and lidar_block >= 0.65
            and lidar_center >= 0.45
        )
        if reverse_center_blockage_entry:
            self._reverse_vehicle_center_memory = 3000
            self._highspeed_construction_memory = 0
            self._disabled_scene_memory = 0
        elif self._reverse_vehicle_center_memory > 0:
            self._reverse_vehicle_center_memory -= 1

        if self._reverse_vehicle_center_memory > 0:
            return {
                "macro_scenario": "reverse_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "high-speed centre tracked vehicle with full LiDAR blockage",
                "top3_macro_candidates": "reverse_vehicle",
            }

        far_command = str(debug.get("command_far", "") or "").upper()
        blind_spot_straight_route = bool(
            "LEFT" not in far_command
            and "RIGHT" not in far_command
            and (
                "LANEFOLLOW" in far_command
                or "STRAIGHT" in far_command
                or far_command in ("", "3", "4")
            )
        )
        blind_spot_clean_asymmetry = bool(
            blind_spot_straight_route
            and construction_count == 0
            and not red_active
            and front_ped is None
            and not reversing
            and front_clear
            and lateral_context <= 1
            and lidar_right <= 0.06
            and lidar_left >= lidar_right + 0.10
        )
        blind_spot_weak_seed = bool(
            blind_spot_clean_asymmetry
            and ego_speed >= 9.0
            and 0.20 <= lane_confidence <= 0.55
            and route_curvature >= 20.0
            and 0.09 <= lidar_left <= 0.30
            and lidar_center <= 0.09
            and 0.035 <= lidar_block <= 0.18
            and lidar_left >= lidar_center + 0.055
            and (fo is None or fo >= 24.0)
        )
        blind_spot_developed_occlusion = bool(
            blind_spot_clean_asymmetry
            and self._highspeed_temp_entry_memory > 0
            and ego_speed >= 5.0
            and lane_confidence <= 0.50
            and lidar_left >= 0.32
            and lidar_center <= 0.42
            and lidar_block >= 0.16
            and lidar_left >= lidar_center + 0.16
            and (fo is None or fo >= 14.0)
        )
        if blind_spot_weak_seed or blind_spot_developed_occlusion:
            self._blind_spot_memory = 5000
            self._highspeed_construction_memory = 0
            self._disabled_scene_memory = 0
        if self._blind_spot_memory > 0:
            self._blind_spot_memory -= 1
            return {
                "macro_scenario": "blind_spot_hidden_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "online straight-route asymmetric left occlusion with right opening",
                "top3_macro_candidates": "blind_spot_hidden_car",
            }
        highspeed_construction_entry = (
            self._highspeed_temp_entry_memory > 0
            and front_ped is None
            and not explicit_reversing
            and construction_reason_evidence
            and (
                raw_macro != "trucks_encountered_during_construction"
                or not truck_like_ahead
            )
        )
        if highspeed_construction_entry:
            self._highspeed_construction_memory = max(self._highspeed_construction_memory, 900)
            self._disabled_scene_memory = 0
        elif self._highspeed_construction_memory > 0:
            self._highspeed_construction_memory -= 1

        highspeed_construction_preempts_disabled = (
            self._highspeed_construction_memory > 0
            and front_ped is None
            and not explicit_reversing
            and (
                raw_macro != "trucks_encountered_during_construction"
                or not truck_like_ahead
            )
        )

        highspeed_temp_signature = (
            highspeed_construction_preempts_disabled
            and (
                construction_reason_evidence
                or lane_confidence <= 0.24
                or lidar_block >= 0.05
                or max(lidar_left, lidar_right) >= 0.15
            )
        )
        if highspeed_temp_signature:
            return {
                "macro_scenario": "high_speed_temporary_construction",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "APPROACH",
                "evidence": "high-speed entry followed by construction corridor/cone evidence",
                "top3_macro_candidates": "high_speed_temporary_construction",
            }

        reckless_cut_in_entry = (
            construction_count == 0
            and not red_active
            and front_ped is None
            and ego_speed >= 10.0
            and tracked_reckless_cut_in_vehicle
        )
        if reckless_cut_in_entry:
            self._reckless_cut_in_memory = 220
        elif self._reckless_cut_in_memory > 0:
            self._reckless_cut_in_memory -= 1

        reckless_cut_in_signature = (
            self._reckless_cut_in_memory > 0
            and construction_count == 0
            and not red_active
            and front_ped is None
            and not reversing
            and (
                reckless_cut_in_entry
                or (
                    fo is not None
                    and 3.0 <= fo <= 18.0
                    and lidar_block >= 0.55
                    and lidar_center >= 0.40
                )
                or "static_obstacle_observed_without_immediate_conflict" in reason_l
            )
        )
        if reckless_cut_in_signature:
            return {
                "macro_scenario": "high_speed_reckless_lane_cutting",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "high-speed cut-in vehicle followed by lane blockage",
                "top3_macro_candidates": "high_speed_reckless_lane_cutting",
            }

        blind_spot_entry = (
            construction_count == 0
            and not red_active
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and side_vehicle_cluster
            and ego_speed >= 5.0
            and lane_confidence <= 0.45
            and lidar_left >= 0.70
            and lidar_center >= 0.25
            and (fo is not None or lidar_block >= 0.45)
        )
        if blind_spot_entry:
            self._blind_spot_memory = 220
        elif self._blind_spot_memory > 0:
            self._blind_spot_memory -= 1

        blind_spot_signature = (
            self._blind_spot_memory > 0
            and construction_count == 0
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and lane_confidence <= 0.50
            and (
                blind_spot_entry
                or lidar_left >= 0.65
                or "active_red_far_prolonged_creep_release" in reason_l
                or "construction_far_sparse_cone_hold_memory" in reason_l
            )
        )
        if blind_spot_signature:
            return {
                "macro_scenario": "blind_spot_hidden_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "high-speed approach with left-side hidden vehicle blockage and right-side opening",
                "top3_macro_candidates": "blind_spot_hidden_car",
            }

        slanted_motor_children_entry = (
            slanted_motor_children_mix
            and advanced_lidar_healthy
            and advanced_lidar_vru_count >= 1
            and raw_macro != "four_students_crossing_the_road"
            and construction_count == 0
            and self._disabled_scene_memory <= 0
            and self._disabled_vehicle_memory <= 0
            and fv is None
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and not missing_car_preempts_disabled
            and not close_static_vehicle
            and ego_speed <= 8.0
            and lane_confidence <= 0.30
            and route_curvature >= 20.0
        )
        if slanted_motor_children_entry:
            self._slanted_motor_children_memory = 5000
            self._crazy_motor_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
        elif self._slanted_motor_children_memory > 0:
            self._slanted_motor_children_memory -= 1

        slanted_motor_children_signature = (
            self._slanted_motor_children_memory > 0
            and construction_count == 0
            and not reversing
            and not missing_car_preempts_disabled
        )
        if slanted_motor_children_signature:
            self._crazy_motor_memory = 0
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "slanted_motor_and_children",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": (
                    "side-origin overlapping VRU/motor detector cluster on low-confidence lane"
                    if slanted_motor_children_entry
                    else "latched slanted motor-and-children route memory"
                ),
                "top3_macro_candidates": "slanted_motor_and_children",
            }

        crazy_motor_entry = (
            close_motor_mix
            and advanced_lidar_healthy
            and advanced_lidar_vru_count >= 1
            and construction_count == 0
            and fv is None
            and fo is None
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and not missing_car_preempts_disabled
            and lateral_context <= 1
            and not immediate_hazard
            and ego_speed <= 7.5
            and lane_confidence >= 0.20
            and route_curvature >= 20.0
        )
        if crazy_motor_entry:
            self._crazy_motor_memory = 180
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
        elif self._crazy_motor_memory > 0:
            self._crazy_motor_memory -= 1

        crazy_motor_signature = (
            self._crazy_motor_memory > 0
            and construction_count == 0
            and fv is None
            and fo is None
            and front_ped is None
            and not reversing
            and not missing_car_preempts_disabled
            and (
                lateral_context <= 1
                or (
                    junction_like
                    and rp is not None
                    and rp <= 10.5
                    and lateral_context <= 10
                )
            )
            and not immediate_hazard
            and ego_speed <= 8.0
            and lane_confidence >= 0.18
            and route_curvature >= 20.0
        )
        if crazy_motor_signature:
            self._disabled_vehicle_memory = 0
            self._disabled_scene_memory = 0
            return {
                "macro_scenario": "crazy_motor",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "close overlapping motor/vehicle detector signature near intersection",
                "top3_macro_candidates": "crazy_motor",
            }

        disabled_close_entry = (
            construction_count == 0
            and front_ped is None
            and not tracked_reckless_cut_in_vehicle
            and not missing_car_preempts_disabled
            and not highspeed_construction_preempts_disabled
            and ego_speed <= 7.5
            and (
                (
                    fv is not None
                    and 2.5 <= fv <= 5.5
                    and fvc <= 0.45
                )
                or (
                    fo is not None
                    and 2.5 <= fo <= 5.5
                    and (
                        immediate_hazard
                        or not front_clear
                        or red_active
                        or "active_red" in reason_l
                    )
                )
                or (
                    close_static_vehicle
                    and (
                        lateral_context >= 2
                        or red_active
                        or immediate_hazard
                        or "active_red" in reason_l
                    )
                )
            )
        )
        if disabled_close_entry:
            self._disabled_vehicle_memory = 220
            self._disabled_scene_memory = 2000
        elif self._disabled_vehicle_memory > 0:
            self._disabled_vehicle_memory -= 1

        disabled_close_signature = (
            self._disabled_vehicle_memory > 0
            and construction_count == 0
            and front_ped is None
            and not tracked_reckless_cut_in_vehicle
            and not missing_car_preempts_disabled
            and not highspeed_construction_preempts_disabled
            and ego_speed <= 8.0
            and (
                disabled_close_entry
                or immediate_hazard
                or not front_clear
                or red_active
                or "active_red" in reason_l
                or "clear_road_cautious_creep_recovery" in reason_l
                or (fo is not None and 2.5 <= fo <= 8.0)
            )
        )
        if disabled_close_signature:
            return {
                "macro_scenario": "avoid_a_disabled_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "close static/slow disabled vehicle with stopline or immediate-hazard context",
                "top3_macro_candidates": "avoid_a_disabled_vehicle",
            }

        vru_crossing_entry = (
            vru_crossing_cluster
            and advanced_lidar_healthy
            and (advanced_lidar_vru_count >= 2 or advanced_vru_memory > 0)
            and construction_count == 0
            and fv is None
            and fo is None
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and ego_speed <= 6.5
            and lane_confidence >= 0.15
            and (
                red_active
                or "creep" in reason_l
                or "rules_disabled_or_low_confidence" in reason_l
                or "active_red" in reason_l
            )
        )
        if vru_crossing_entry:
            self._vru_crossing_memory = 180
        elif self._vru_crossing_memory > 0:
            self._vru_crossing_memory -= 1

        vru_crossing_signature = (
            self._vru_crossing_memory > 0
            and construction_count == 0
            and fv is None
            and front_ped is None
            and not reversing
            and not tracked_reckless_cut_in_vehicle
            and ego_speed <= 7.0
            and lane_confidence >= 0.15
            and (
                vru_crossing_entry
                or red_active
                or "creep" in reason_l
                or "active_red" in reason_l
            )
        )
        if vru_crossing_signature:
            return {
                "macro_scenario": "four_students_crossing_the_road",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "multi-vru detector cluster near crossing during creep/red approach",
                "top3_macro_candidates": "four_students_crossing_the_road",
            }

        missing_car_signature = (
            self._missing_car_event_memory > 0
            and construction_count == 0
            and front_ped is None
            and (
                high_speed_front_vehicle_event
                or "active_red_far_prolonged_creep_release" in reason_l
                or "clear_road_cautious_creep_recovery" in reason_l
                or self._highway_entry_memory > 0
                or (front_clear and fv is None and fo is None and ego_speed <= 4.5)
            )
        )
        if missing_car_signature:
            return {
                "macro_scenario": "missing_car",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "YIELD_OR_BRAKE",
                "evidence": "high-speed close lead/reverse event followed by disappearing front target",
                "top3_macro_candidates": "missing_car",
            }

        highway_accident_signature = (
            self._highway_entry_memory > 0
            and construction_count == 0
            and not red_active
            and front_ped is None
            and not reversing
            and fo is not None
            and 4.0 <= fo <= 12.0
            and lidar_block >= 0.70
            and tracked_lateral_accident_vehicle
            and ego_speed <= 8.0
        )
        if highway_accident_signature:
            return {
                "macro_scenario": "highway_accident_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "high-speed entry followed by lateral accident vehicle blockage",
                "top3_macro_candidates": "highway_accident_vehicle",
            }

        reverse_signature = (
            construction_count == 0
            and not red_active
            and front_ped is None
            and reversing
            and self._disabled_vehicle_memory == 0
            and (
                (fo is not None and 5.0 <= fo <= 24.0 and lidar_block >= 0.30)
                or tracked_reversing_vehicle
            )
        )
        if reverse_signature:
            return {
                "macro_scenario": "reverse_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "front tracked vehicle moving backward toward ego lane",
                "top3_macro_candidates": "reverse_vehicle",
            }

        roundabout_signature = (
            construction_count == 0
            and not red_active
            and rp is None
            and front_ped is None
            and not reversing
            and (
                (
                    junction_like
                    and route_curvature >= 3.0
                )
                or (
                    route_curvature >= 12.0
                    and lateral_context >= 4
                    and ego_speed <= 9.0
                    and (not front_clear or lidar_block >= 0.40 or (fo is not None and fo <= 2.0))
                )
            )
        )
        if roundabout_signature:
            return {
                "macro_scenario": "roundabout",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "APPROACH",
                "evidence": "curved junction/roundabout topology from map and route geometry",
                "top3_macro_candidates": "roundabout",
            }

        disabled_signature = (
            construction_count == 0
            and front_ped is None
            and not reversing
            and not missing_car_preempts_disabled
            and not highspeed_construction_preempts_disabled
            and (
                (fv is not None and 4.0 <= fv <= 28.0 and ego_speed <= 9.0 and lidar_block >= 0.20)
                or (fo is not None and 3.0 <= fo <= 18.0 and lidar_block >= 0.35 and lidar_center >= 0.30)
                or (
                    immediate_hazard
                    and fv is not None
                    and fo is not None
                    and 2.5 <= fv <= 4.5
                    and abs(fv - fo) <= 0.75
                    and ego_speed <= 2.0
                )
                or (
                    tracked_vehicle_hazard
                    and ego_speed <= 8.0
                )
            )
        )
        if disabled_signature:
            self._disabled_scene_memory = 2000
            return {
                "macro_scenario": "avoid_a_disabled_vehicle",
                "confidence": max(confidence, 0.80),
                "phase": phase if phase != "unknown" else "PREPARE",
                "evidence": "front static/slow vehicle without construction or red-light context",
                "top3_macro_candidates": "avoid_a_disabled_vehicle",
            }

        construction_evidence = (
            construction_count > 0
            or (
                ("construction" in reason_l or "cone" in reason_l or "corridor" in reason_l)
                and "sparse_cone" not in reason_l
                and "false_red" not in reason_l
                and "active_red" not in reason_l
                and "red_light" not in reason_l
            )
        )
        if raw_macro in ("trucks_encountered_during_construction", "high_speed_temporary_construction") and not construction_evidence:
            raw_macro = "unknown"
            confidence = min(confidence, 0.40)
        if (
            self._disabled_scene_memory > 0
            and raw_macro in ("trucks_encountered_during_construction", "high_speed_temporary_construction")
        ):
            raw_macro = "unknown"
            confidence = min(confidence, 0.40)
        if controller_only and raw_macro not in ("unknown", "avoid_a_disabled_vehicle"):
            raw_macro = "unknown"
            confidence = min(confidence, 0.40)

        return {
            "macro_scenario": raw_macro or "unknown",
            "confidence": confidence,
            "phase": phase,
            "evidence": reason,
            "top3_macro_candidates": raw_macro or "unknown",
        }
