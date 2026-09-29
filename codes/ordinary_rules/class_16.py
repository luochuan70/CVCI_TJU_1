"""Class-16 longitudinal controller for merging into same-direction slow flow."""

from __future__ import absolute_import

import math
from collections import deque

import numpy as np

from .common import clone_control, feature_value, _number
from .configs import get_config


class Rule16(object):
    """Match the observed flow speed while preserving learned route steering.

    Highway merge ramps contain close guardrail/background returns for long
    periods.  React only to radar points whose ego-compensated radial speed is
    consistent with observed same-direction slow traffic.
    """

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.last_timestamp = None
        self.last_flow_distance_m = None
        self.flow_seen = False
        self.flow_clear_frames = 0
        self.conflict_history = deque(maxlen=5)
        self.conflict_lateral_history = deque(maxlen=7)
        self.conflict_speed_history = deque(maxlen=5)
        self.conflict_motion_mode = "unknown"
        self.clearance_side_history = deque(maxlen=7)
        self.flow_corridor_margin_history = deque(maxlen=80)
        self.static_left_clearance_history = deque(maxlen=3)
        self.static_right_clearance_history = deque(maxlen=3)
        self.conflict_active = False
        self.conflict_clear_frames = 0
        self.last_conflict_cluster_timestamp = None
        self.stop_confirm_frames = 0
        self.left_clear_confirm_frames = 0
        self.right_clear_confirm_frames = 0
        self.avoidance_committed = False
        self.avoidance_side_sign = 0.0
        self.avoidance_progress_m = 0.0
        self.avoidance_last_timestamp = None
        self.avoidance_clear_start_progress_m = None
        self.last_selected_boundary_clearance_m = None
        self.last_selected_boundary_timestamp = None
        self.selected_boundary_closing_rate_mps = 0.0
        self.boundary_predicted_deficit_frames = 0
        self.rear_side_pressure_latched = False
        self.rear_side_pressure_clear_frames = 0
        self.last_reliable_flow_speed_mps = 0.0
        self.last_reliable_flow_timestamp = None
        # A merge is a physical lifecycle, not a one-second radar sample.
        # Once an adjacent rear vehicle has established that lifecycle, retain
        # the last reliable surrounding-flow speed until the lifecycle clears.
        # This prevents a sparse radar interval from first allowing ego to
        # decelerate and then commanding a late acceleration as the side
        # vehicle reaches the body envelope.
        self.merge_flow_target_speed_mps = 0.0
        self.fast_crossing_conflict = False
        self.post_fast_follow_active = False
        self.post_fast_follow_miss_frames = 0
        self.merge_entry_follow_active = False
        self.merge_entry_follow_last_seen_timestamp = None
        self.merge_entry_follow_lifecycle_observed = False
        self.merge_entry_follow_clear_frames = 0
        self.longitudinal_merge_latched = False
        self.merge_gap_wait_active = False
        self.merge_gap_near_pass_seen = False
        self.merge_gap_near_last_seen_timestamp = None
        self.merge_gap_clear_frames = 0
        self.merge_gap_consecutive_empty_frames = 0
        self.merge_gap_safe_following_frames = 0
        self.merge_gap_committed = False
        self.merge_gap_last_rear_side_timestamp = None
        self.merge_gap_commit_flow_target_speed_mps = 0.0
        self.merge_gap_commit_progress_m = 0.0
        self.merge_gap_commit_last_timestamp = None
        self.merge_gap_entry_offset_history = deque(maxlen=32)
        self.merge_gap_commit_start_offset_m = 0.0
        self.merge_gap_lateral_rejoin_started = False
        self.merge_gap_lateral_rejoin_start_progress_m = None
        self.rear_side_stationary_distance_history = deque(maxlen=40)
        self.front_track_last_distance_m = None
        self.front_track_last_timestamp = None
        self.front_track_closing_history = deque(maxlen=5)
        self.right_obstacle_latched = False
        self.right_obstacle_stop_frames = 0
        self.right_obstacle_left_clear_frames = 0
        self.right_obstacle_plan = None
        self.right_obstacle_plan_phase = "idle"
        self.right_obstacle_plan_completed = False
        self.right_obstacle_pass_rear_seen = False
        self.right_obstacle_pass_clear_frames = 0
        self.right_obstacle_pass_progress_m = 0.0
        self.right_obstacle_intrusion_latched = False
        self.right_obstacle_intrusion_clear_frames = 0
        self.right_obstacle_intrusion_hold_frames = 0
        self.right_obstacle_intrusion_detour_active = False
        self.right_obstacle_intrusion_detour_count = 0
        self.right_obstacle_handoff_pending = False
        self.right_obstacle_handoff_observe_frames = 0
        self.right_obstacle_handoff_resolved_as_flow = False
        self.lead_reversal_obstacle_confirm_frames = 0
        self.lead_reversal_obstacle_latched = False
        self.lead_reversal_dynamic_wait_active = False
        self.lead_reversal_front_clear_frames = 0
        self.lead_reversal_stationary_frames = 0
        self.lead_reversal_last_front_timestamp = None
        self.right_obstacle_last_side_occupied_timestamp = None
        self.right_obstacle_gap_required_s = 0.0
        self.right_obstacle_route_offsets = deque(maxlen=9)
        self.right_track_last_distance_m = None
        self.right_track_last_timestamp = None
        self.right_track_closing_history = deque(maxlen=5)
        self.lead_departure_commit = False

    @staticmethod
    def _packet_points(packet):
        if isinstance(packet, (tuple, list)) and len(packet) == 2:
            packet = packet[1]
        try:
            points = np.asarray(packet, dtype=np.float32)
        except Exception:
            return np.empty((0, 4), dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 4:
            return np.empty((0, 4), dtype=np.float32)
        points = points[:, :4]
        return points[np.isfinite(points).all(axis=1)]

    def _flow_observation(self, radar, ego_speed, timestamp):
        points = self._packet_points(radar.get("_online_radar_packet"))
        result = {
            "count": 0, "front_count": 0, "front_distance_m": 0.0,
            "flow_speed_mps": 0.0, "closing_rate_mps": 0.0,
            "conflict_cluster_count": 0,
            "conflict_cluster_distance_m": 0.0,
            "conflict_cluster_speed_mps": 0.0,
            "conflict_cluster_lateral_m": 0.0,
            "conflict_cluster_confirmed": False,
            "conflict_active": False,
            "conflict_clear_frames": 0,
            "static_left_clearance_m": 0.0,
            "static_right_clearance_m": 0.0,
        }
        if len(points) == 0:
            self.conflict_history.append(False)
            if self.conflict_active:
                self.conflict_clear_frames += 1
                clearance_elapsed = (
                    timestamp - self.last_conflict_cluster_timestamp
                    if self.last_conflict_cluster_timestamp is not None
                    else 0.0
                )
                if (
                    self.conflict_clear_frames >= max(1, int(_number(
                        self.config.get("conflict_clear_confirm_frames"), 5
                    )))
                    and clearance_elapsed >= _number(
                        self.config.get("conflict_post_loss_clearance_s"), 1.5
                    )
                ):
                    self.conflict_active = False
            result["conflict_active"] = bool(self.conflict_active)
            result["conflict_clear_frames"] = int(self.conflict_clear_frames)
            if not self.conflict_active and self.fast_crossing_conflict:
                self.post_fast_follow_active = True
                self.post_fast_follow_miss_frames = 0
                self.fast_crossing_conflict = False
            self.last_timestamp = timestamp
            self.last_flow_distance_m = None
            return result

        depth, altitude, azimuth, velocity = (
            points[:, 0], points[:, 1], points[:, 2], points[:, 3]
        )
        compensated_speed = velocity + abs(ego_speed) * np.cos(azimuth)
        longitudinal = depth * np.cos(azimuth)
        lateral = depth * np.sin(azimuth)
        # Guardrails and road edges are world-static returns: after ego-speed
        # compensation their radial speed is near zero.  Keep this estimator
        # separate from the moving-flow cluster and require several points so
        # one radar speck cannot steer the vehicle.  CARLA radar positive
        # lateral is the route-left side used by the signed route offset.
        static_boundary = (
            (depth > 1.0)
            & (depth <= _number(
                self.config.get("boundary_radar_max_depth_m"), 30.0
            ))
            & (np.abs(altitude) <= _number(
                self.config.get("boundary_max_altitude_rad"), 0.25
            ))
            & (np.abs(azimuth) >= math.radians(_number(
                self.config.get("boundary_min_azimuth_deg"), 5.7
            )))
            & (longitudinal > 0.0)
            & (longitudinal <= _number(
                self.config.get("boundary_forward_window_m"), 15.0
            ))
            & (np.abs(compensated_speed) <= _number(
                self.config.get("boundary_static_speed_abs_mps"), 0.3
            ))
        )
        boundary_min_points = max(3, int(_number(
            self.config.get("boundary_min_points"), 3
        )))
        for side_mask, history, key in (
            (lateral > 0.0, self.static_left_clearance_history,
             "static_left_clearance_m"),
            (lateral < 0.0, self.static_right_clearance_history,
             "static_right_clearance_m"),
        ):
            values = np.abs(lateral[static_boundary & side_mask])
            if len(values) >= boundary_min_points:
                history.append(float(np.percentile(values, 10.0)))
            if history:
                result[key] = float(np.median(np.asarray(history)))
        same_direction = (
            (depth > 1.0)
            & (depth <= _number(self.config.get("flow_radar_max_m"), 60.0))
            & (np.abs(altitude) <= _number(
                self.config.get("flow_max_altitude_rad"), 0.35
            ))
            & (np.abs(azimuth) <= math.radians(_number(
                self.config.get("flow_radar_half_angle_deg"), 70.0
            )))
            & (compensated_speed >= _number(
                self.config.get("flow_min_speed_mps"), 6.0
            ))
            & (compensated_speed <= _number(
                self.config.get("flow_max_speed_mps"), 12.5
            ))
        )
        front = same_direction & (
            np.abs(azimuth) <= math.radians(_number(
                self.config.get("flow_front_half_angle_deg"), 22.0
            ))
        )
        # During the actual merge, a traffic-flow vehicle crosses the sensor
        # axis obliquely before becoming longitudinal.  Its ego-compensated
        # radial speed is a modest negative value, not the positive road speed
        # used by the following gate above.  Require a compact multi-point
        # depth cluster so curved guardrail residuals cannot claim this event.
        # The initial gate excludes sub-2 m returns to reject ego-body and
        # road-edge clutter.  Once a real actor has been confirmed, keep
        # tracking its same coherent cluster down to the radar near field;
        # otherwise the actor's body is still beside the ego when the initial
        # depth gate goes false and the controller releases into contact.
        conflict_min_depth = _number(
            self.config.get(
                "conflict_tracking_min_depth_m"
                if self.conflict_active else "conflict_min_depth_m"
            ),
            0.5 if self.conflict_active else 2.0,
        )
        conflict_points = (
            (depth >= conflict_min_depth)
            & (depth <= _number(self.config.get("conflict_max_depth_m"), 35.0))
            & (np.abs(altitude) <= _number(
                self.config.get("flow_max_altitude_rad"), 0.35
            ))
            & (np.abs(azimuth) <= math.radians(_number(
                self.config.get("conflict_half_angle_deg"), 35.0
            )))
            & (compensated_speed >= _number(
                self.config.get("conflict_min_speed_mps"), -4.0
            ))
            & (compensated_speed <= _number(
                self.config.get("conflict_max_speed_mps"), -0.3
            ))
        )
        cluster_width = _number(
            self.config.get("conflict_depth_cluster_half_width_m"), 0.75
        )
        best_indices = None
        best_count = 0
        for centre in depth[conflict_points]:
            indices = np.where(
                conflict_points & (np.abs(depth - centre) <= cluster_width)
            )[0]
            if len(indices) > best_count:
                best_indices = indices
                best_count = int(len(indices))
        minimum_cluster = max(1, int(_number(
            self.config.get("conflict_min_cluster_points"), 4
        )))
        cluster_frame = bool(best_indices is not None and best_count >= minimum_cluster)
        if cluster_frame:
            self.last_conflict_cluster_timestamp = timestamp
            cluster_lateral = depth[best_indices] * np.sin(azimuth[best_indices])
            result["conflict_cluster_lateral_m"] = float(np.median(
                cluster_lateral
            ))
            minimum_lateral = _number(
                self.config.get("avoidance_side_min_lateral_m"), 0.30
            )
            lateral_value = result["conflict_cluster_lateral_m"]
            if abs(lateral_value) >= minimum_lateral:
                self.conflict_lateral_history.append(lateral_value)
        self.conflict_history.append(cluster_frame)
        confirm_frames = max(1, int(_number(
            self.config.get("conflict_confirm_frames"), 2
        )))
        result["conflict_cluster_confirmed"] = bool(
            sum(1 for value in self.conflict_history if value) >= confirm_frames
        )
        if result["conflict_cluster_confirmed"]:
            self.conflict_active = True
            self.conflict_clear_frames = 0
            if cluster_frame:
                cluster_speed = float(np.median(compensated_speed[best_indices]))
                if cluster_speed <= _number(
                    self.config.get("fast_crossing_conflict_max_speed_mps"),
                    -1.8,
                ):
                    self.fast_crossing_conflict = True
                    self.post_fast_follow_active = False
                    self.post_fast_follow_miss_frames = 0
        elif self.conflict_active:
            if cluster_frame:
                self.conflict_clear_frames = 0
            else:
                self.conflict_clear_frames += 1
                clearance_elapsed = (
                    timestamp - self.last_conflict_cluster_timestamp
                    if self.last_conflict_cluster_timestamp is not None
                    else 0.0
                )
                if (
                    self.conflict_clear_frames >= max(1, int(_number(
                        self.config.get("conflict_clear_confirm_frames"), 5
                    )))
                    and clearance_elapsed >= _number(
                        self.config.get("conflict_post_loss_clearance_s"), 1.5
                    )
                ):
                    self.conflict_active = False
        result["conflict_active"] = bool(self.conflict_active)
        result["conflict_clear_frames"] = int(self.conflict_clear_frames)
        if not self.conflict_active and self.fast_crossing_conflict:
            self.post_fast_follow_active = True
            self.post_fast_follow_miss_frames = 0
            self.fast_crossing_conflict = False
        if cluster_frame:
            result["conflict_cluster_count"] = best_count
            result["conflict_cluster_distance_m"] = float(np.median(
                depth[best_indices]
            ))
            result["conflict_cluster_speed_mps"] = float(np.median(
                compensated_speed[best_indices]
            ))
        result["count"] = int(np.count_nonzero(same_direction))
        result["front_count"] = int(np.count_nonzero(front))
        if result["count"]:
            result["flow_speed_mps"] = float(np.median(
                compensated_speed[same_direction]
            ))
            self.flow_seen = True
            self.flow_clear_frames = 0
        else:
            self.flow_clear_frames += 1
        if result["front_count"]:
            distance = float(np.min(depth[front]))
            result["front_distance_m"] = distance
            if (
                self.last_flow_distance_m is not None
                and self.last_timestamp is not None
                and timestamp > self.last_timestamp
            ):
                dt = min(0.25, max(0.02, timestamp - self.last_timestamp))
                result["closing_rate_mps"] = max(
                    -15.0,
                    min(15.0, (self.last_flow_distance_m - distance) / dt),
                )
            self.last_flow_distance_m = distance
        else:
            self.last_flow_distance_m = None
        self.last_timestamp = timestamp
        return result

    @staticmethod
    def _traffic_hold(features, ignore_vehicle_immediate=False):
        kind = str(getattr(features, "traffic_control_kind", "") or "").lower()
        return bool(
            getattr(features, "red_light_active", False)
            or kind in {"traffic_light", "trafficlight", "stop_sign", "stopsign"}
            or getattr(features, "front_pedestrian_distance", None) is not None
            or (
                getattr(features, "immediate_hazard", False)
                and not ignore_vehicle_immediate
            )
        )

    @staticmethod
    def _track_forward_lateral(item):
        """Normalize heterogeneous detector tracks to forward/lateral.

        The legacy DriveTransformer detector emits x=lateral/y=forward,
        while OpenPCDet emits x=forward/y=lateral.  Runtime tracks preserve
        the source-native convention.  Synthetic unit-test tracks without a
        source field retain the explicitly standardized x-forward contract.
        """
        x_value = _number(item.get("x"), float("nan"))
        y_value = _number(item.get("y"), float("nan"))
        if "source" not in item:
            return x_value, y_value
        if str(item.get("source", "") or "").lower() == "openpcdet":
            return x_value, y_value
        return y_value, x_value

    def _rear_side_pressure(self, features):
        """Return online evidence for a vehicle closing through a rear side.

        Class-16 is a merge into an already moving flow.  A vehicle directly
        behind the ego is not enough to override the learned throttle, because
        it may simply be a normal follower.  The measured route-3785 failure
        instead contained a persistent adjacent-lane vehicle moving from the
        rear quarter toward the ego.  Use only compact online perception
        tracks in that physical envelope; route identity and map coordinates
        are deliberately unavailable here.
        """
        vehicles = {"car", "van", "truck", "vehicle"}
        minimum_score = _number(
            self.config.get("rear_side_min_track_score"), 0.25
        )
        minimum_lateral = _number(
            self.config.get("rear_side_min_lateral_m"), 1.0
        )
        maximum_lateral = _number(
            self.config.get("rear_side_max_lateral_m"), 4.5
        )
        minimum_rear = _number(
            self.config.get("rear_side_min_rear_m"), 2.0
        )
        maximum_rear = _number(
            self.config.get("rear_side_max_rear_m"), 15.0
        )
        acquisition_matches = []
        lifecycle_matches = []
        for item in list(getattr(features, "tracked_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in vehicles:
                continue
            score = _number(item.get("score"), 0.0)
            forward, lateral = self._track_forward_lateral(item)
            if not (math.isfinite(forward) and math.isfinite(lateral)):
                continue
            if not (
                score >= minimum_score
                and abs(lateral) <= maximum_lateral
                and -maximum_rear <= forward <= -minimum_rear
            ):
                continue
            match = (abs(forward), lateral, score)
            lifecycle_matches.append(match)
            if abs(lateral) >= minimum_lateral:
                acquisition_matches.append(match)
        if acquisition_matches:
            self.rear_side_pressure_latched = True
            self.rear_side_pressure_clear_frames = 0
        elif self.rear_side_pressure_latched and lifecycle_matches:
            self.rear_side_pressure_clear_frames = 0
        elif self.rear_side_pressure_latched:
            self.rear_side_pressure_clear_frames += 1
            if self.rear_side_pressure_clear_frames >= max(1, int(_number(
                self.config.get("rear_side_clear_confirm_frames"), 5
            ))):
                self.rear_side_pressure_latched = False
                self.rear_side_pressure_clear_frames = 0
        lifecycle_matches.sort(key=lambda value: value[0])
        nearest = lifecycle_matches[0] if lifecycle_matches else None
        return {
            "active": bool(self.rear_side_pressure_latched),
            "acquired_this_frame": bool(acquisition_matches),
            "lifecycle_observed": bool(lifecycle_matches),
            "clear_frames": int(self.rear_side_pressure_clear_frames),
            "count": len(lifecycle_matches),
            "nearest_rear_m": float(nearest[0]) if nearest else 0.0,
            "nearest_lateral_m": float(nearest[1]) if nearest else 0.0,
            "nearest_score": float(nearest[2]) if nearest else 0.0,
        }

    def _front_tracked_vehicle(self, features):
        vehicles = {"car", "van", "truck", "vehicle"}
        minimum_score = _number(
            self.config.get("post_fast_follow_min_track_score"), 0.5
        )
        maximum_lateral = _number(
            self.config.get("post_fast_follow_max_lateral_m"), 4.5
        )
        maximum_distance = _number(
            self.config.get("post_fast_follow_max_distance_m"), 25.0
        )
        matches = []
        for item in list(getattr(features, "tracked_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in vehicles:
                continue
            score = _number(item.get("score"), 0.0)
            forward, lateral = self._track_forward_lateral(item)
            if not (math.isfinite(forward) and math.isfinite(lateral)):
                continue
            if (
                score >= minimum_score
                and abs(lateral) <= maximum_lateral
                and 0.0 < forward <= maximum_distance
            ):
                matches.append((forward, lateral, score))
        matches.sort(key=lambda value: value[0])
        nearest = matches[0] if matches else None
        return {
            "active": bool(nearest),
            "count": len(matches),
            "distance_m": float(nearest[0]) if nearest else 0.0,
            "lateral_m": float(nearest[1]) if nearest else 0.0,
            "score": float(nearest[2]) if nearest else 0.0,
        }

    def _front_track_closing_rate(self, front_track, timestamp):
        """Estimate lead-vehicle closure from the normalized online track.

        The merge-gap controller previously accelerated at the captured road
        speed while ignoring a much slower tracked vehicle already inside the
        selected gap.  Keep a short robust distance derivative so the commit
        controller can follow that vehicle instead of treating the gap as
        empty.  This consumes only live detector output and time.
        """
        if not front_track["active"]:
            if (
                self.front_track_last_timestamp is not None
                and timestamp - self.front_track_last_timestamp > 0.5
            ):
                self.front_track_last_distance_m = None
                self.front_track_last_timestamp = None
                self.front_track_closing_history.clear()
            return 0.0
        distance = front_track["distance_m"]
        if (
            self.front_track_last_distance_m is not None
            and self.front_track_last_timestamp is not None
            and timestamp > self.front_track_last_timestamp
        ):
            dt = timestamp - self.front_track_last_timestamp
            if dt <= 0.5:
                closing = (
                    self.front_track_last_distance_m - distance
                ) / max(0.05, dt)
                self.front_track_closing_history.append(max(-12.0, min(12.0, closing)))
        self.front_track_last_distance_m = distance
        self.front_track_last_timestamp = timestamp
        if not self.front_track_closing_history:
            return 0.0
        return float(np.median(np.asarray(
            list(self.front_track_closing_history), dtype=np.float32
        )))

    def _right_side_tracked_vehicle(self, features, timestamp):
        matches = []
        for item in list(getattr(features, "tracked_objects", None) or []):
            if not isinstance(item, dict):
                continue
            if str(item.get("class_name", "") or "").lower() not in {
                "car", "van", "truck", "vehicle"
            }:
                continue
            forward, lateral = self._track_forward_lateral(item)
            if not (math.isfinite(forward) and math.isfinite(lateral)):
                continue
            if (
                _number(item.get("score"), 0.0) >= _number(
                    self.config.get("right_obstacle_min_track_score"), 0.5
                )
                and 0.0 < forward <= _number(
                    self.config.get("right_obstacle_max_forward_m"), 8.0
                )
                and _number(
                    self.config.get("right_obstacle_min_lateral_m"), 1.25
                ) <= lateral <= _number(
                    self.config.get("right_obstacle_max_lateral_m"), 4.5
                )
            ):
                matches.append((forward, lateral, _number(item.get("score"), 0.0)))
        matches.sort(key=lambda value: value[0])
        nearest = matches[0] if matches else None
        if nearest is None:
            if (
                self.right_track_last_timestamp is not None
                and timestamp - self.right_track_last_timestamp > 0.5
            ):
                self.right_track_last_distance_m = None
                self.right_track_last_timestamp = None
                self.right_track_closing_history.clear()
            return {"active": False, "distance_m": 0.0, "lateral_m": 0.0,
                    "score": 0.0, "closing_rate_mps": 0.0}
        distance = nearest[0]
        if (
            self.right_track_last_distance_m is not None
            and self.right_track_last_timestamp is not None
            and timestamp > self.right_track_last_timestamp
        ):
            dt = timestamp - self.right_track_last_timestamp
            if dt <= 0.5:
                closing = (self.right_track_last_distance_m - distance) / max(0.05, dt)
                self.right_track_closing_history.append(max(-12.0, min(12.0, closing)))
        self.right_track_last_distance_m = distance
        self.right_track_last_timestamp = timestamp
        rate = float(np.median(np.asarray(
            list(self.right_track_closing_history), dtype=np.float32
        ))) if self.right_track_closing_history else 0.0
        return {
            "active": True, "distance_m": float(distance),
            "lateral_m": float(nearest[1]), "score": float(nearest[2]),
            "closing_rate_mps": rate,
        }

    def _build_right_obstacle_left_plan(self, route_lateral):
        """Build the pass phase; rejoining is authorized by actor lifecycle."""
        origin = route_lateral if math.isfinite(route_lateral) else 0.0
        # Navigation's near-waypoint lateral error jumps by several metres
        # while ego is stopped at this junction.  Treating that value as the
        # path origin made an intended left bypass become a rightward target.
        # Initial body clearance is therefore a local, odometry-relative left
        # displacement derived from the combined vehicle envelope.
        relative_left = abs(_number(
            self.config.get("right_obstacle_relative_left_clearance_m"), 2.4
        ))
        corridor = origin - relative_left
        shift_end = _number(
            self.config.get("right_obstacle_initial_shift_end_m"), 4.0
        )
        hold_end = _number(
            self.config.get("right_obstacle_pass_horizon_m"), 90.0
        )
        target_y = -relative_left
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            waypoints.append([
                shift_end * fraction,
                target_y * 0.5 * (1.0 - math.cos(math.pi * fraction)),
            ])
        for fraction in (0.25, 0.5, 0.75, 1.0):
            waypoints.append([
                shift_end + (hold_end - shift_end) * fraction,
                target_y,
            ])
        return {
            "active": True,
            "class_id": 16,
            "path_id": "c16_stop_plan_left_pass_v65",
            "waypoints": waypoints,
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": False,
            "fuse_live_route_lateral": False,
            "route_pid_phase": "ordinary_class16_stop_plan_left",
            "pid_step_rejoin": True,
            "radar_corridor_m": corridor,
            "origin_route_lateral_m": origin,
            "hold_end_m": hold_end,
            "rejoin_end_m": hold_end,
            "stabilize_end_m": hold_end,
            "finish_lateral_tolerance_m": 0.35,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": 700,
        }

    def _build_right_obstacle_rejoin_plan(self, route_lateral):
        """Build a new local plan only after the obstacle is safely behind."""
        origin = route_lateral if math.isfinite(route_lateral) else -abs(
            _number(self.config.get("right_obstacle_left_corridor_m"), 1.6)
        )
        rejoin = _number(
            self.config.get("right_obstacle_event_rejoin_m"), 12.0
        )
        finish = rejoin + _number(
            self.config.get("right_obstacle_post_rejoin_stabilize_m"), 6.0
        )
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            waypoints.append([
                rejoin * fraction,
                -origin * 0.5 * (1.0 - math.cos(math.pi * fraction)),
            ])
        waypoints.append([finish, -origin])
        return {
            "active": True,
            "class_id": 16,
            "path_id": "c16_stop_plan_left_rejoin_v65",
            "waypoints": waypoints,
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": True,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class16_event_rejoin",
            "pid_step_rejoin": True,
            "radar_corridor_m": 0.0,
            "origin_route_lateral_m": origin,
            "hold_end_m": 0.0,
            "rejoin_end_m": rejoin,
            "stabilize_end_m": finish,
            "finish_lateral_tolerance_m": 0.35,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": 360,
        }

    def _build_right_obstacle_intrusion_detour(
        self, route_lateral, actor_lateral, combined_half_width
    ):
        """Replan away from a stopped actor that entered the pass corridor."""
        origin = route_lateral if math.isfinite(route_lateral) else 0.0
        away_sign = -1.0 if actor_lateral > 0.0 else 1.0
        clearance = combined_half_width + _number(
            self.config.get("right_obstacle_detour_extra_clearance_m"), 0.4
        )
        actor_route_offset = origin + actor_lateral
        corridor = actor_route_offset + away_sign * clearance
        limit = abs(_number(
            self.config.get("right_obstacle_detour_route_limit_m"), 2.2
        ))
        corridor = max(-limit, min(limit, corridor))
        shift_end = _number(
            self.config.get("right_obstacle_detour_shift_end_m"), 7.0
        )
        horizon = shift_end + 2.0
        target_y = corridor - origin
        waypoints = [[0.0, 0.0]]
        for fraction in (0.25, 0.5, 0.75, 1.0):
            waypoints.append([
                shift_end * fraction,
                target_y * 0.5 * (1.0 - math.cos(math.pi * fraction)),
            ])
        waypoints.append([horizon, target_y])
        return {
            "active": True,
            "class_id": 16,
            "cycle": int(self.right_obstacle_intrusion_detour_count),
            "path_id": "c16_intrusion_detour_shift_v69",
            "waypoints": waypoints,
            "lookahead_m": 2.5,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.06,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class16_intrusion_detour",
            "pid_step_rejoin": False,
            # The route PID moved only ~0.1 m laterally over the measured
            # six-metre escape despite a 1.2 m target change.  The short
            # detour therefore uses the local pure-pursuit controller; a new
            # route-relative hold plan takes over immediately after shifting.
            "use_carla_route_pid": False,
            "radar_corridor_m": corridor,
            "origin_route_lateral_m": origin,
            "hold_end_m": horizon,
            "rejoin_end_m": horizon,
            "stabilize_end_m": horizon,
            "finish_lateral_tolerance_m": 0.35,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": 900,
        }

    def _build_right_obstacle_detour_hold(self, route_lateral, corridor):
        """Follow the online route at the newly reached safe offset."""
        origin = route_lateral if math.isfinite(route_lateral) else corridor
        horizon = _number(
            self.config.get("right_obstacle_pass_horizon_m"), 200.0
        )
        target_y = corridor - origin
        return {
            "active": True,
            "class_id": 16,
            "cycle": int(self.right_obstacle_intrusion_detour_count),
            "path_id": "c16_intrusion_detour_hold_v69",
            "waypoints": [[0.0, 0.0], [4.0, target_y], [horizon, target_y]],
            "lookahead_m": 2.8,
            "steer_kp": 1.05,
            "steer_kd": 0.04,
            "steer_limit": 0.48,
            "steer_rate_per_frame": 0.06,
            "use_carla_route_pid": True,
            "fuse_live_route_lateral": True,
            "route_pid_phase": "ordinary_class16_intrusion_detour_hold",
            "pid_step_rejoin": False,
            "radar_corridor_m": corridor,
            "origin_route_lateral_m": origin,
            "hold_end_m": horizon,
            "rejoin_end_m": horizon,
            "stabilize_end_m": horizon,
            "finish_lateral_tolerance_m": 0.35,
            "finish_heading_tolerance_rad": 0.18,
            "timeout_frames": 900,
        }

    def apply(self, control, radar, features):
        result = clone_control(control)
        speed = max(0.0, feature_value(features, "ego_speed", 0.0))
        timestamp = _number(radar.get("_timestamp"), 0.0)
        route_lateral = _number(
            radar.get("_route_lateral_m"), float("nan")
        )
        local_path_feedback = radar.get("_local_path_feedback") or {}
        feedback_path_key = local_path_feedback.get("path_key") or []
        feedback_path_id = (
            str(feedback_path_key[2]) if len(feedback_path_key) >= 3 else ""
        )
        feedback_matches_plan = bool(
            self.right_obstacle_plan is not None
            and feedback_path_id
            == str(self.right_obstacle_plan.get("path_id", ""))
        )
        if (
            self.right_obstacle_plan is not None
            and self.right_obstacle_plan_phase == "rejoin"
            and feedback_matches_plan
        ):
            feedback_progress = (
                _number(local_path_feedback.get("x_m"), 0.0)
                if feedback_matches_plan else 0.0
            )
            hard_finish = _number(
                self.right_obstacle_plan.get("stabilize_end_m"), 34.0
            ) + _number(
                self.config.get("right_obstacle_finish_guard_m"), 2.0
            )
            if (
                bool(local_path_feedback.get("done", False))
                or feedback_progress >= hard_finish
            ):
                self.right_obstacle_plan = None
                self.right_obstacle_plan_phase = "complete"
                self.right_obstacle_plan_completed = True
                self.right_obstacle_latched = False
                self.avoidance_committed = False
                self.longitudinal_merge_latched = False
                self.merge_gap_wait_active = False
                self.merge_gap_near_pass_seen = False
                self.merge_gap_near_last_seen_timestamp = None
                self.merge_gap_clear_frames = 0
                self.merge_gap_consecutive_empty_frames = 0
                self.merge_gap_safe_following_frames = 0
                self.merge_gap_committed = False
                self.merge_entry_follow_active = False
                self.merge_entry_follow_lifecycle_observed = False
                self.merge_entry_follow_clear_frames = 0
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
        flow = self._flow_observation(radar, speed, timestamp)
        conflict_speed = _number(
            flow.get("conflict_cluster_speed_mps"), 0.0
        )
        if flow.get("conflict_cluster_count", 0) > 0 and conflict_speed < 0.0:
            self.conflict_speed_history.append(conflict_speed)
        if (
            self.conflict_motion_mode == "unknown"
            and len(self.conflict_speed_history) >= 3
        ):
            median_conflict_speed = float(np.median(np.asarray(
                list(self.conflict_speed_history), dtype=np.float32
            )))
            self.conflict_motion_mode = (
                "fast_flow"
                if median_conflict_speed <= _number(
                    self.config.get("fast_crossing_conflict_max_speed_mps"),
                    -1.8,
                )
                else "slow_queue"
            )
        # The radial label alone was unsafe on stopped-obstacle routes, but
        # retiring the proven slow-queue path entirely made an ego already
        # matched to the online flow brake from roughly 8 m/s to zero inside
        # route2273's moving queue. Restore that topology only when the live
        # motion context proves ego still has near-flow momentum and no
        # dedicated stopped/reversing-obstacle or independent hazard owns it.
        slow_queue_reference_speed_mps = max(
            _number(self.config.get("merge_gap_min_flow_speed_mps"), 8.0),
            self.last_reliable_flow_speed_mps,
            _number(flow.get("flow_speed_mps"), 0.0),
        )
        slow_queue_momentum_min_speed_mps = max(
            _number(
                self.config.get("slow_queue_momentum_min_speed_mps"), 5.5
            ),
            _number(
                self.config.get("slow_queue_momentum_min_flow_ratio"), 0.65
            ) * slow_queue_reference_speed_mps,
        )
        slow_queue_momentum_context = bool(
            speed >= slow_queue_momentum_min_speed_mps
            and not self.right_obstacle_latched
            and not self.right_obstacle_handoff_pending
            and not self.lead_reversal_obstacle_latched
            and not self.lead_reversal_dynamic_wait_active
            and not bool(getattr(features, "immediate_hazard", False))
            and not bool(getattr(features, "red_light_active", False))
            and getattr(features, "front_pedestrian_distance", None) is None
        )
        slow_queue_bypass = bool(
            self.conflict_motion_mode == "slow_queue"
            and slow_queue_momentum_context
        )
        flow_corridor_margin_sampled = False
        if (
            slow_queue_momentum_context
            and self.longitudinal_merge_latched
        ):
            # A five-frame conflict-only window is dominated by the moving
            # actor and flipped sides across repeated route2273 runs.  The
            # static road corridor exists for several seconds before that
            # actor arrives.  Preserve its robust online history while ego is
            # already matched to the flow; stopped obstacles never sample it.
            pre_left_clear = bool(getattr(features, "left_clear", False))
            pre_right_clear = bool(getattr(features, "right_clear", False))
            pre_side_risk = bool(getattr(features, "side_risk", False))
            pre_left_boundary = _number(
                flow.get("static_left_clearance_m"), 0.0
            )
            pre_right_boundary = _number(
                flow.get("static_right_clearance_m"), 0.0
            )
            pre_clearance_margin = 0.0
            if not pre_side_risk:
                if (
                    pre_left_clear and not pre_right_clear
                    and pre_left_boundary > 0.0
                ):
                    pre_clearance_margin = pre_left_boundary
                elif (
                    pre_right_clear and not pre_left_clear
                    and pre_right_boundary > 0.0
                ):
                    pre_clearance_margin = -pre_right_boundary
                elif (
                    pre_left_clear and pre_right_clear
                    and pre_left_boundary > 0.0 and pre_right_boundary > 0.0
                ):
                    pre_clearance_margin = pre_left_boundary - pre_right_boundary
            if pre_clearance_margin != 0.0:
                self.flow_corridor_margin_history.append(pre_clearance_margin)
                flow_corridor_margin_sampled = True
        slow_queue_corridor_median_m = (
            float(np.median(np.asarray(
                list(self.flow_corridor_margin_history), dtype=np.float32
            )))
            if self.flow_corridor_margin_history else 0.0
        )
        slow_queue_corridor_preferred_sign = 0.0
        if (
            slow_queue_bypass
            and len(self.flow_corridor_margin_history) >= max(5, int(_number(
                self.config.get("slow_queue_corridor_min_samples"), 20
            )))
        ):
            slow_queue_corridor_preferred_sign = (
                1.0 if slow_queue_corridor_median_m > 0.0
                else -1.0 if slow_queue_corridor_median_m < 0.0
                else 0.0
            )
        if slow_queue_bypass and not self.merge_gap_committed:
            # One or two conflict frames arrive before the three-frame motion
            # mode freezes, so the generic fast-flow gate may already have
            # latched a wait.  Transfer ownership atomically once the guarded
            # slow-queue evidence is complete.
            self.merge_gap_wait_active = False
            self.merge_gap_near_pass_seen = False
            self.merge_gap_near_last_seen_timestamp = None
            self.merge_gap_clear_frames = 0
            self.merge_gap_consecutive_empty_frames = 0
            self.merge_gap_safe_following_frames = 0
        rear_side_pressure = self._rear_side_pressure(features)
        # Once a side-flow actor has passed the ego it can rotate into the
        # direct rear corridor.  Treating that normal lateral transition as
        # loss of the stationary sample clears the only evidence that the
        # following vehicle is stopped because ego is yielding in front of
        # it.  Track the nearest *current* rear vehicle across that transition;
        # temporal distance stability, not a fixed lateral bin, establishes
        # that it is stationary.
        rear_stationary_sample = bool(
            rear_side_pressure["lifecycle_observed"]
            and rear_side_pressure["nearest_rear_m"] > 0.0
        )
        if rear_stationary_sample:
            self.rear_side_stationary_distance_history.append(
                rear_side_pressure["nearest_rear_m"]
            )
        else:
            self.rear_side_stationary_distance_history.clear()
        rear_stationary_required = max(5, int(_number(
            self.config.get("merge_gap_stationary_rear_confirm_frames"), 30
        )))
        rear_stationary_span_m = (
            max(self.rear_side_stationary_distance_history)
            - min(self.rear_side_stationary_distance_history)
            if self.rear_side_stationary_distance_history else float("inf")
        )
        rear_stationary_confirmed = bool(
            len(self.rear_side_stationary_distance_history)
            >= rear_stationary_required
            and rear_stationary_span_m <= _number(
                self.config.get("merge_gap_stationary_rear_span_m"), 0.5
            )
        )
        if (
            rear_side_pressure["active"]
            or (
                self.merge_gap_wait_active
                and not self.merge_gap_committed
                and rear_side_pressure["lifecycle_observed"]
            )
        ):
            self.merge_gap_last_rear_side_timestamp = timestamp
        merge_gap_rear_clear_elapsed = (
            timestamp - self.merge_gap_last_rear_side_timestamp
            if self.merge_gap_last_rear_side_timestamp is not None
            else float("inf")
        )
        front_tracked_vehicle = self._front_tracked_vehicle(features)
        front_track_closing_rate_mps = self._front_track_closing_rate(
            front_tracked_vehicle, timestamp
        )
        right_side_vehicle = self._right_side_tracked_vehicle(
            features, timestamp
        )
        if (
            self.right_obstacle_plan_phase == "initial_shift"
            and feedback_matches_plan
            and _number(local_path_feedback.get("x_m"), 0.0)
            >= _number(
                self.config.get("right_obstacle_initial_shift_end_m"), 4.0
            )
        ):
            # Keep the same local path and coordinate frame.  Only enable the
            # rear-quarter lifecycle gate after the relative left shift has
            # physically completed; rebuilding here would reintroduce the
            # unstable stopped-route lateral measurement.
            self.right_obstacle_plan_phase = "pass"
            self.right_obstacle_pass_progress_m = _number(
                local_path_feedback.get("x_m"), 0.0
            )
        if (
            self.right_obstacle_plan_phase == "detour_shift"
            and feedback_matches_plan
            and _number(local_path_feedback.get("x_m"), 0.0)
            >= _number(
                self.config.get("right_obstacle_detour_shift_end_m"), 4.0
            )
        ):
            detour_corridor = _number(
                self.right_obstacle_plan.get("radar_corridor_m"),
                route_lateral,
            )
            self.right_obstacle_plan = self._build_right_obstacle_detour_hold(
                route_lateral, detour_corridor
            )
            self.right_obstacle_plan_phase = "pass"
            self.right_obstacle_intrusion_detour_active = False
            self.right_obstacle_pass_progress_m = 0.0
        if self.right_obstacle_plan_phase == "pass":
            # A completed geometric segment is not proof that a dynamic actor
            # was passed.  Observe its transition into the right-rear quarter
            # and require body clearance before constructing the rejoin path.
            rear_lateral = _number(
                rear_side_pressure.get("nearest_lateral_m"), 0.0
            )
            rear_distance = _number(
                rear_side_pressure.get("nearest_rear_m"), 0.0
            )
            right_rear_observed = bool(
                rear_side_pressure.get("lifecycle_observed", False)
                and abs(rear_lateral) >= _number(
                    self.config.get("right_obstacle_rear_min_lateral_m"), 0.5
                )
            )
            if right_rear_observed:
                self.right_obstacle_pass_rear_seen = True
                self.right_obstacle_pass_clear_frames = 0
            elif self.right_obstacle_pass_rear_seen:
                self.right_obstacle_pass_clear_frames += 1

            feedback_progress = _number(
                local_path_feedback.get("x_m"), 0.0
            )
            self.right_obstacle_pass_progress_m = max(
                self.right_obstacle_pass_progress_m, feedback_progress
            )
            rear_body_clear = bool(
                self.right_obstacle_pass_rear_seen
                and (
                    (
                        right_rear_observed
                        and rear_distance >= _number(
                            self.config.get(
                                "right_obstacle_rejoin_rear_clearance_m"
                            ), 7.0
                        )
                    )
                    or self.right_obstacle_pass_clear_frames >= max(
                        1, int(_number(self.config.get(
                            "right_obstacle_rejoin_rear_clear_frames"
                        ), 6))
                    )
                )
            )
            if (
                rear_body_clear
                and self.right_obstacle_pass_progress_m >= _number(
                    self.config.get("right_obstacle_min_pass_progress_m"), 12.0
                )
                and not right_side_vehicle["active"]
                and not (
                    front_tracked_vehicle["active"]
                    and front_tracked_vehicle["distance_m"] <= _number(
                        self.config.get(
                            "right_obstacle_rejoin_front_clearance_m"
                        ), 15.0
                    )
                )
            ):
                self.right_obstacle_plan = (
                    self._build_right_obstacle_rejoin_plan(route_lateral)
                )
                self.right_obstacle_plan_phase = "rejoin"
                self.right_obstacle_intrusion_detour_active = False
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
            else:
                # A fast-flow radar cluster must not arm the legacy tail
                # follower while the pass lifecycle still owns control.
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
        startup_vehicle_handoff = bool(
            radar.get("_class04_vehicle_hazard_handoff", False)
        )
        stopped_handoff_candidate = bool(
            startup_vehicle_handoff
            and not self.right_obstacle_plan_completed
            and not self.right_obstacle_handoff_resolved_as_flow
            and right_side_vehicle["active"]
            and speed <= _number(
                self.config.get("avoidance_stop_max_speed_mps"), 0.15
            )
        )
        if (
            stopped_handoff_candidate
            and not self.right_obstacle_latched
            and not self.right_obstacle_handoff_pending
        ):
            self.right_obstacle_handoff_pending = True
            self.right_obstacle_handoff_observe_frames = 0
        if self.right_obstacle_handoff_pending:
            self.right_obstacle_handoff_observe_frames += 1
            departing_motion = bool(
                len(self.right_track_closing_history) >= max(2, int(_number(
                    self.config.get("right_obstacle_motion_confirm_frames"), 4
                )))
                and right_side_vehicle["closing_rate_mps"] <= -_number(
                    self.config.get(
                        "right_obstacle_flow_departure_mps"), 1.5
                )
            )
            online_flow_lifecycle = bool(
                rear_side_pressure["active"]
                or rear_side_pressure["lifecycle_observed"]
                or flow["count"] >= max(1, int(_number(
                    self.config.get(
                        "right_obstacle_flow_min_points"), 3
                )))
            )
            if departing_motion and online_flow_lifecycle:
                self.right_obstacle_handoff_pending = False
                self.right_obstacle_handoff_resolved_as_flow = True
                self.longitudinal_merge_latched = True
                self.merge_gap_wait_active = True
                self.lead_departure_commit = True
                if flow["flow_speed_mps"] > 0.0:
                    self.last_reliable_flow_speed_mps = flow["flow_speed_mps"]
                    self.last_reliable_flow_timestamp = timestamp
                    self.merge_flow_target_speed_mps = flow["flow_speed_mps"]
            elif self.right_obstacle_handoff_observe_frames >= max(
                1, int(_number(self.config.get(
                    "right_obstacle_handoff_observe_max_frames"), 10
                ))
            ):
                # No coherent flow lifecycle appeared.  Preserve the safer
                # stopped-obstacle interpretation instead of timing out into
                # learned throttle.
                self.right_obstacle_handoff_pending = False
                self.right_obstacle_latched = True
                self.right_obstacle_last_side_occupied_timestamp = timestamp
        # Once a conflict has been identified as the live fast-flow merge
        # episode, its front/right track remains owned by the merge-gap state
        # machine.  Reclassifying the same actor as a stopped right obstacle
        # after ego brakes for the gap made route3785 leave that state after
        # only three seconds, plan across the actor, and collide.  A genuine
        # stopped/reversing lead still transfers ownership explicitly through
        # lead_reversal_obstacle_latched below.
        fast_flow_merge_owns_right_actor = bool(
            self.conflict_motion_mode == "fast_flow"
            and self.longitudinal_merge_latched
            and (self.merge_gap_wait_active or self.merge_gap_committed)
            and not self.lead_reversal_obstacle_latched
        )
        right_obstacle_candidate = bool(
            not self.right_obstacle_plan_completed
            and not self.right_obstacle_handoff_pending
            and not self.right_obstacle_handoff_resolved_as_flow
            and not fast_flow_merge_owns_right_actor
            and (
                (
                    right_side_vehicle["active"]
                    and len(self.right_track_closing_history) >= max(2, int(_number(
                        self.config.get("right_obstacle_motion_confirm_frames"), 4
                    )))
                    and right_side_vehicle["closing_rate_mps"] >= -_number(
                        self.config.get("right_obstacle_max_departure_mps"), 0.75
                    )
                )
            )
        )
        # A vehicle which initially departed into the traffic stream can later
        # stop or reverse back toward ego.  The handoff is an observation, not
        # a permanent semantic label: once the merge envelope has been cleared,
        # repeated live-track evidence that the lead is nearly stationary or
        # reversing must transfer ownership back to the stop-and-plan chain.
        # Otherwise the merge follower keeps approaching an actor that can
        # physically reverse into a stopped ego.
        estimated_lead_speed_mps = (
            speed - front_track_closing_rate_mps
            if front_tracked_vehicle["active"] else float("inf")
        )
        lead_reversal_obstacle_sample = bool(
            self.lead_departure_commit
            and self.merge_gap_committed
            and self.merge_gap_commit_progress_m >= _number(
                self.config.get("lead_reversal_min_commit_progress_m"), 25.0
            )
            and front_tracked_vehicle["active"]
            and front_tracked_vehicle["distance_m"] <= _number(
                self.config.get("lead_reversal_max_distance_m"), 25.0
            )
            and front_track_closing_rate_mps >= _number(
                self.config.get("lead_reversal_min_closing_mps"), 4.0
            )
            and estimated_lead_speed_mps <= _number(
                self.config.get("lead_reversal_max_lead_speed_mps"), 1.0
            )
            and not bool(getattr(features, "red_light_active", False))
            and not str(
                getattr(features, "traffic_control_kind", "") or ""
            ).strip()
        )
        self.lead_reversal_obstacle_confirm_frames = (
            self.lead_reversal_obstacle_confirm_frames + 1
            if lead_reversal_obstacle_sample else 0
        )
        if self.lead_reversal_obstacle_confirm_frames >= max(2, int(_number(
            self.config.get("lead_reversal_confirm_frames"), 2
        ))):
            self.lead_reversal_obstacle_latched = True
            # This is a longitudinally reversing/stopped lead, not a lateral
            # crossing actor.  Waiting in its lane allowed it to run into a
            # fully stopped ego.  Brake first, then let the dedicated
            # stop-gap-left-plan chain move out of the swept lane as soon as
            # the observed left corridor is safe.
            self.lead_reversal_dynamic_wait_active = False
            self.lead_reversal_front_clear_frames = 0
            self.lead_reversal_stationary_frames = 0
            self.lead_reversal_last_front_timestamp = timestamp
            self.right_obstacle_handoff_resolved_as_flow = False
            self.right_obstacle_handoff_pending = False
            self.right_obstacle_latched = True
            self.merge_gap_committed = False
            self.merge_gap_wait_active = False
            self.longitudinal_merge_latched = False
            self.merge_entry_follow_active = False
            self.merge_entry_follow_lifecycle_observed = False
            self.post_fast_follow_active = False
            self.lead_reversal_obstacle_confirm_frames = 0
        if self.lead_reversal_dynamic_wait_active:
            if front_tracked_vehicle["active"]:
                self.lead_reversal_last_front_timestamp = timestamp
                self.lead_reversal_front_clear_frames = 0
                self.lead_reversal_stationary_frames = (
                    self.lead_reversal_stationary_frames + 1
                    if speed <= _number(
                        self.config.get("avoidance_stop_max_speed_mps"), 0.15
                    )
                    and abs(estimated_lead_speed_mps) <= _number(
                        self.config.get(
                            "lead_reversal_stationary_speed_mps"), 0.5
                    )
                    else 0
                )
            else:
                self.lead_reversal_front_clear_frames += 1
            clear_elapsed_s = (
                timestamp - self.lead_reversal_last_front_timestamp
                if self.lead_reversal_last_front_timestamp is not None else 0.0
            )
            passed_to_rear = bool(
                rear_side_pressure["lifecycle_observed"]
                and rear_side_pressure["nearest_rear_m"] >= _number(
                    self.config.get("lead_reversal_rear_clearance_m"), 7.0
                )
            )
            dynamic_actor_cleared = bool(
                not front_tracked_vehicle["active"]
                and not bool(getattr(features, "immediate_hazard", False))
                and (
                    passed_to_rear
                    and self.lead_reversal_front_clear_frames >= max(
                        2, int(_number(self.config.get(
                            "lead_reversal_rear_clear_frames"), 3
                        ))
                    )
                    or self.lead_reversal_front_clear_frames >= max(
                        3, int(_number(self.config.get(
                            "lead_reversal_front_clear_frames"), 15
                        ))
                    )
                    and clear_elapsed_s >= _number(
                        self.config.get("lead_reversal_clear_min_s"), 1.5
                    )
                )
            )
            if dynamic_actor_cleared:
                self.lead_reversal_dynamic_wait_active = False
                self.lead_reversal_obstacle_latched = False
                self.right_obstacle_latched = False
                self.right_obstacle_plan_completed = True
                self.avoidance_committed = False
                self.avoidance_side_sign = 0.0
                self.clearance_side_history.clear()
                self.stop_confirm_frames = 0
                self.left_clear_confirm_frames = 0
                self.right_clear_confirm_frames = 0
                self.fast_crossing_conflict = False
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
            elif self.lead_reversal_stationary_frames >= max(3, int(_number(
                self.config.get("lead_reversal_stationary_confirm_frames"), 20
            ))):
                # The actor did not continue through the crossing lifecycle.
                # It is now a true stopped obstacle, so the normal stop-gap-
                # plan chain may construct a category-level left bypass.
                self.lead_reversal_dynamic_wait_active = False
                # Transfer ownership atomically.  The generic oblique-flow
                # branch observes the same radar body and may already have
                # accumulated a lateral side vote while we were waiting.  It
                # must not pre-empt the dedicated stop-gap-plan chain at the
                # exact frame the actor becomes a stationary obstacle.
                self.avoidance_committed = False
                self.avoidance_side_sign = 0.0
                self.clearance_side_history.clear()
                self.stop_confirm_frames = 0
                self.left_clear_confirm_frames = 0
                self.right_clear_confirm_frames = 0
                self.fast_crossing_conflict = False
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
        if right_obstacle_candidate:
            if not self.right_obstacle_latched:
                self.right_obstacle_last_side_occupied_timestamp = timestamp
            self.right_obstacle_latched = True
            self.merge_gap_committed = False
            self.merge_gap_wait_active = False
        if (
            self.right_obstacle_latched
            and not self.avoidance_committed
            and not self.lead_reversal_dynamic_wait_active
            and math.isfinite(route_lateral)
        ):
            self.right_obstacle_route_offsets.append(route_lateral)
        if (
            self.right_obstacle_latched
            and not self.avoidance_committed
            and not self.lead_reversal_dynamic_wait_active
        ):
            side_corridor_occupied = bool(
                rear_side_pressure["active"]
                or rear_side_pressure["lifecycle_observed"]
            )
            if side_corridor_occupied:
                self.right_obstacle_last_side_occupied_timestamp = timestamp
            self.right_obstacle_gap_required_s = (
                _number(self.config.get("right_obstacle_shift_end_m"), 7.0)
                / max(0.5, _number(
                    self.config.get("right_obstacle_target_speed_mps"), 5.5
                ))
                + _number(
                    self.config.get("right_obstacle_body_clearance_s"), 0.8
                )
            )
            side_clear_elapsed_s = (
                timestamp - self.right_obstacle_last_side_occupied_timestamp
                if self.right_obstacle_last_side_occupied_timestamp is not None
                # No observed side traffic means the gap is already clear;
                # do not invent a mandatory wait before the stopped vehicle.
                else self.right_obstacle_gap_required_s
            )
            self.right_obstacle_stop_frames = (
                self.right_obstacle_stop_frames + 1
                if speed <= _number(
                    self.config.get("avoidance_stop_max_speed_mps"), 0.15
                ) else 0
            )
            left_boundary = _number(flow.get("static_left_clearance_m"), 0.0)
            left_plan_feasible = bool(
                bool(getattr(features, "left_clear", False))
                and not bool(getattr(features, "side_risk", False))
                and not side_corridor_occupied
                and side_clear_elapsed_s >= self.right_obstacle_gap_required_s
                and (
                    left_boundary <= 0.0
                    or left_boundary >= _number(
                        self.config.get(
                            "right_obstacle_min_left_clearance_m"
                        ), 1.5
                    )
                )
            )
            self.right_obstacle_left_clear_frames = (
                self.right_obstacle_left_clear_frames + 1
                if left_plan_feasible else 0
            )
            if (
                self.right_obstacle_stop_frames >= max(2, int(_number(
                    self.config.get("right_obstacle_stop_confirm_frames"), 3
                )))
                and self.right_obstacle_left_clear_frames >= max(2, int(_number(
                    self.config.get(
                        "right_obstacle_left_clear_confirm_frames"
                    ), 3
                )))
            ):
                self.right_obstacle_plan = self._build_right_obstacle_left_plan(
                    route_lateral
                )
                self.right_obstacle_plan_phase = "initial_shift"
                self.right_obstacle_pass_rear_seen = False
                self.right_obstacle_pass_clear_frames = 0
                self.right_obstacle_pass_progress_m = 0.0
                self.post_fast_follow_active = False
                self.post_fast_follow_miss_frames = 0
                self.avoidance_committed = True
                self.avoidance_side_sign = -1.0
                self.avoidance_last_timestamp = timestamp
        entry_max_distance = _number(
            self.config.get("merge_entry_track_max_distance_m"), 4.5
        )
        entry_max_lateral = _number(
            self.config.get("merge_entry_track_max_lateral_m"), 1.75
        )
        entry_lateral = front_tracked_vehicle["lateral_m"]
        entry_candidate = bool(
            not slow_queue_bypass
            and not self.right_obstacle_plan_completed
            and not self.avoidance_committed
            and front_tracked_vehicle["active"]
            and 0.0 < front_tracked_vehicle["distance_m"] <= entry_max_distance
            and abs(entry_lateral) <= entry_max_lateral
            and flow["count"] >= 3
        )
        if entry_candidate:
            self.merge_entry_follow_active = True
            self.merge_entry_follow_last_seen_timestamp = timestamp
            self.merge_entry_follow_lifecycle_observed = True
            self.longitudinal_merge_latched = True
            self.merge_entry_follow_clear_frames = 0
        elif (
            self.merge_entry_follow_active
            and self.merge_entry_follow_last_seen_timestamp is not None
            and timestamp - self.merge_entry_follow_last_seen_timestamp
            > _number(self.config.get("merge_entry_follow_dropout_s"), 0.45)
        ):
            self.merge_entry_follow_active = False
        if (
            self.merge_entry_follow_lifecycle_observed
            and not self.merge_entry_follow_active
        ):
            if flow["conflict_active"]:
                self.merge_entry_follow_clear_frames = 0
            else:
                self.merge_entry_follow_clear_frames += 1
                if self.merge_entry_follow_clear_frames >= max(1, int(_number(
                    self.config.get("merge_entry_follow_clear_frames"), 5
                ))):
                    self.merge_entry_follow_lifecycle_observed = False
                    self.merge_entry_follow_clear_frames = 0
        if self.post_fast_follow_active:
            if front_tracked_vehicle["active"]:
                self.post_fast_follow_miss_frames = 0
            else:
                self.post_fast_follow_miss_frames += 1
                if self.post_fast_follow_miss_frames >= max(1, int(_number(
                    self.config.get("post_fast_follow_clear_frames"), 10
                ))):
                    self.post_fast_follow_active = False
                    self.post_fast_follow_miss_frames = 0
        minimum_reliable_flow_points = max(1, int(_number(
            self.config.get("rear_side_min_front_flow_points"), 3
        )))
        if (
            not self.right_obstacle_plan_completed
            and
            flow["front_count"] >= minimum_reliable_flow_points
            and flow["flow_speed_mps"] > 0.0
            and not flow["conflict_active"]
        ):
            # Rule16 is invoked only after the online classifier selected the
            # class.  A stable same-direction flow is sufficient to establish
            # longitudinal-merge semantics even on geometries where no actor
            # enters the <=4.5m detector envelope (for example route 3785).
            # Never acquire this semantic from a currently active crossing
            # cluster: a close crossing body can momentarily resemble road
            # speed after ego-motion compensation.
            self.longitudinal_merge_latched = True
            self.last_reliable_flow_speed_mps = flow["flow_speed_mps"]
            self.last_reliable_flow_timestamp = timestamp
            if rear_side_pressure["active"]:
                self.merge_flow_target_speed_mps = flow["flow_speed_mps"]
        if (
            self.longitudinal_merge_latched
            and not slow_queue_bypass
            and not self.avoidance_committed
        ):
            # After stable same-direction traffic has established the
            # longitudinal merge semantics, an oblique radar cluster belongs
            # to that traffic episode.  Reinterpreting it as a crossing actor
            # made route3785 stop in-lane for four seconds and get rear-ended.
            # Real lane-front distance, traffic controls and VRUs retain their
            # independent higher-priority braking branches below.
            self.fast_crossing_conflict = False
            self.post_fast_follow_active = False
            self.post_fast_follow_miss_frames = 0
            if flow["conflict_active"] and not self.merge_gap_committed:
                self.merge_gap_wait_active = True
            if self.merge_gap_wait_active and not self.merge_gap_committed:
                if (
                    math.isfinite(route_lateral)
                    and speed <= _number(
                        self.config.get("merge_gap_offset_sample_max_speed_mps"),
                        0.5,
                    )
                ):
                    self.merge_gap_entry_offset_history.append(route_lateral)
                lead_departing = bool(
                    not self.right_obstacle_latched
                    and front_tracked_vehicle["active"]
                    and front_tracked_vehicle["distance_m"] >= _number(
                        self.config.get("merge_departing_lead_min_gap_m"), 6.0
                    )
                    and len(self.front_track_closing_history) >= max(
                        2, int(_number(self.config.get(
                            "merge_departing_lead_confirm_frames"
                        ), 3))
                    )
                    and front_track_closing_rate_mps <= -_number(
                        self.config.get(
                            "merge_departing_lead_min_departure_mps"
                        ), 1.5
                    )
                    and not rear_side_pressure["active"]
                    and not rear_side_pressure["lifecycle_observed"]
                )
                if lead_departing:
                    # The side vehicle has become a forward-moving lead. Enter
                    # behind it instead of waiting for a fictitious radar
                    # near-pass or attempting a lateral overtake.
                    self.merge_gap_committed = True
                    self.merge_gap_wait_active = False
                    self.lead_departure_commit = True
                    self.merge_gap_commit_flow_target_speed_mps = min(
                        _number(self.config.get("flow_speed_hard_mps"), 9.0),
                        max(
                            _number(self.config.get(
                                "merge_gap_commit_target_speed_mps"
                            ), 8.2),
                            self.last_reliable_flow_speed_mps,
                        ),
                    )
                    self.merge_gap_commit_progress_m = 0.0
                    self.merge_gap_commit_last_timestamp = timestamp
                    # Stay on the online route when following the lead. A
                    # lateral path is reserved for a confirmed stopped or
                    # reversing right-side obstacle.
                    self.merge_gap_commit_start_offset_m = 0.0
                conflict_distance = flow["conflict_cluster_distance_m"]
                if conflict_distance > 0.0:
                    self.merge_gap_consecutive_empty_frames = 0
                    if conflict_distance <= _number(
                        self.config.get("merge_gap_pass_near_m"), 5.0
                    ):
                        self.merge_gap_near_pass_seen = True
                        self.merge_gap_near_last_seen_timestamp = timestamp
                        self.merge_gap_clear_frames = 0
                        self.merge_gap_safe_following_frames = 0
                elif self.merge_gap_near_pass_seen:
                    self.merge_gap_clear_frames += 1
                    self.merge_gap_consecutive_empty_frames += 1
                near_clearance_elapsed = (
                    timestamp - self.merge_gap_near_last_seen_timestamp
                    if self.merge_gap_near_last_seen_timestamp is not None
                    else 0.0
                )
                # Oblique radial speed is only the projection toward the
                # sensor.  On route3785 it was ~2.3 m/s while the already
                # observed road-flow speed was ~9 m/s, inflating an unsafe
                # 20 m gap to a fictitious eight-second TTC.  Judge the next
                # vehicle with the online longitudinal flow speed instead.
                conservative_flow_speed = max(
                    _number(
                        self.config.get("merge_gap_min_flow_speed_mps"), 8.0
                    ),
                    self.last_reliable_flow_speed_mps,
                )
                next_conflict_ttc = (
                    conflict_distance / conservative_flow_speed
                    if conflict_distance > 0.0 else 0.0
                )
                following_gap_sample = bool(
                    conflict_distance > _number(
                        self.config.get("merge_gap_pass_near_m"), 5.0
                    )
                    and conflict_distance >= _number(
                        self.config.get("merge_gap_next_min_distance_m"),
                        25.0,
                    )
                    and next_conflict_ttc >= _number(
                        self.config.get("merge_gap_next_min_ttc_s"), 3.0
                    )
                    and near_clearance_elapsed >= _number(
                        self.config.get("merge_gap_body_clearance_s"), 1.2
                    )
                )
                if following_gap_sample:
                    self.merge_gap_safe_following_frames += 1
                elif conflict_distance > 0.0:
                    self.merge_gap_safe_following_frames = 0
                sustained_empty_gap = bool(
                    self.merge_gap_consecutive_empty_frames >= max(
                        1, int(_number(self.config.get(
                            "merge_gap_empty_confirm_frames"
                        ), 15))
                    )
                    and near_clearance_elapsed >= _number(
                        self.config.get("merge_gap_empty_min_s"), 2.5
                    )
                )
                # A stopped ego needs more than a detector debounce interval
                # to enter a moving stream.  Require enough rear-corridor
                # clearance to accelerate to the online flow speed and then
                # clear one vehicle body.  This is evaluated from live speed
                # and flow evidence; it is not a route-specific wait value.
                merge_gap_entry_target_speed = max(
                    _number(
                        self.config.get("merge_gap_commit_target_speed_mps"),
                        8.2,
                    ),
                    self.last_reliable_flow_speed_mps,
                )
                merge_gap_required_rear_clear_s = max(
                    _number(
                        self.config.get("merge_gap_rear_clear_min_s"), 1.5
                    ),
                    max(0.0, merge_gap_entry_target_speed - speed)
                    / max(
                        0.5,
                        _number(
                            self.config.get("merge_gap_assumed_accel_mps2"),
                            3.0,
                        ),
                    )
                    + _number(
                        self.config.get("merge_gap_body_clearance_s"), 1.5
                    ),
                )
                # A side vehicle can stop behind the yielding ego and remain
                # inside the rear-envelope detector forever.  Requiring that
                # actor to disappear creates a circular deadlock.  After the
                # crossing vehicle has passed, authorize the measured pocket
                # only when the adjacent rear actor is persistently stationary
                # and the forward body gap is large and non-closing.
                # The compact radar conflict sector and detector tracks cover
                # different geometry.  An empty conflict sector must never
                # override a currently tracked lead inside the body corridor.
                # Use the empty-horizon alternative only when there is no
                # tracked front vehicle at all.
                stationary_front_room_ready = bool(
                    (
                        front_tracked_vehicle["active"]
                        and front_tracked_vehicle["distance_m"] >= _number(
                            self.config.get(
                                "merge_gap_stationary_front_min_gap_m"), 10.0
                        )
                        and front_track_closing_rate_mps <= _number(
                            self.config.get(
                                "merge_gap_stationary_front_max_closing_mps"),
                            0.5,
                        )
                    )
                    or (
                        not front_tracked_vehicle["active"]
                        and sustained_empty_gap
                    )
                )
                stationary_rear_pocket_ready = bool(
                    rear_stationary_confirmed
                    and rear_side_pressure["lifecycle_observed"]
                    and rear_side_pressure["nearest_rear_m"] >= _number(
                        self.config.get(
                            "merge_gap_stationary_rear_min_gap_m"), 5.0
                    )
                    and stationary_front_room_ready
                )
                if (
                    not self.right_obstacle_latched
                    and not self.merge_gap_committed
                    and self.merge_gap_near_pass_seen
                    and self.merge_gap_clear_frames >= max(1, int(_number(
                        self.config.get("merge_gap_clear_confirm_frames"), 3
                    )))
                    and (
                        self.merge_gap_safe_following_frames >= max(
                            1, int(_number(self.config.get(
                                "merge_gap_safe_following_confirm_frames"
                            ), 2))
                        )
                        or sustained_empty_gap
                    )
                    and (
                        stationary_rear_pocket_ready
                        or (
                            not rear_side_pressure["active"]
                            and not rear_side_pressure["lifecycle_observed"]
                            and merge_gap_rear_clear_elapsed
                            >= merge_gap_required_rear_clear_s
                        )
                    )
                ):
                    # Commit only after observing the following vehicle twice
                    # at a gap feasible under the measured road-flow speed.
                    # Empty radar frames never authorize entry by themselves.
                    self.merge_gap_committed = True
                    self.merge_gap_wait_active = False
                    # Freeze the online road-flow estimate at the physical
                    # commit boundary.  The reliable-flow freshness window is
                    # intentionally short for normal driving, but a gap
                    # commit remains a live manoeuvre after the actor leaves
                    # the front radar.  Dropping this target after one second
                    # let the model coast below the surrounding stream while
                    # a rear-quarter vehicle closed on the ego body.
                    self.merge_gap_commit_flow_target_speed_mps = min(
                        _number(self.config.get("flow_speed_hard_mps"), 9.0),
                        max(
                            _number(
                                self.config.get(
                                    "merge_gap_commit_target_speed_mps"
                                ), 8.2,
                            ),
                            self.last_reliable_flow_speed_mps,
                        ),
                    )
                    self.merge_gap_commit_progress_m = 0.0
                    self.merge_gap_commit_last_timestamp = timestamp
                    if self.merge_gap_entry_offset_history:
                        measured_entry_offset = float(np.median(
                            list(self.merge_gap_entry_offset_history)
                        ))
                        maximum_entry_offset = abs(_number(
                            self.config.get(
                                "merge_gap_entry_offset_max_abs_m"
                            ), 4.5
                        ))
                        measured_entry_offset = max(
                            -maximum_entry_offset,
                            min(maximum_entry_offset, measured_entry_offset),
                        )
                        if abs(measured_entry_offset) >= _number(
                            self.config.get(
                                "merge_gap_entry_offset_min_abs_m"
                            ), 1.0
                        ):
                            self.merge_gap_commit_start_offset_m = (
                                measured_entry_offset
                            )
                    self.merge_gap_lateral_rejoin_started = False
                    self.merge_gap_lateral_rejoin_start_progress_m = None
        if self.merge_gap_committed:
            if self.merge_gap_commit_last_timestamp is not None:
                commit_dt = max(
                    0.0,
                    min(0.20, timestamp - self.merge_gap_commit_last_timestamp),
                )
                self.merge_gap_commit_progress_m += speed * commit_dt
            self.merge_gap_commit_last_timestamp = timestamp
            commit_target = max(
                _number(
                    self.config.get("merge_gap_commit_target_speed_mps"), 8.2
                ),
                self.merge_gap_commit_flow_target_speed_mps,
            )
            if (
                not self.merge_gap_lateral_rejoin_started
                and speed >= commit_target - _number(
                    self.config.get(
                        "merge_gap_rejoin_speed_margin_mps"
                    ), 0.5
                )
                and not rear_side_pressure["active"]
                and not rear_side_pressure["lifecycle_observed"]
                and merge_gap_rear_clear_elapsed >= _number(
                    self.config.get("merge_gap_rejoin_rear_clear_s"), 1.5
                )
                and not bool(getattr(features, "immediate_hazard", False))
            ):
                self.merge_gap_lateral_rejoin_started = True
                self.merge_gap_lateral_rejoin_start_progress_m = (
                    self.merge_gap_commit_progress_m
                )
        if not rear_side_pressure["active"]:
            self.merge_flow_target_speed_mps = 0.0
        reliable_flow_age_s = (
            timestamp - self.last_reliable_flow_timestamp
            if self.last_reliable_flow_timestamp is not None else float("inf")
        )
        retained_flow_speed_mps = (
            self.merge_flow_target_speed_mps
            if rear_side_pressure["active"]
            and self.merge_flow_target_speed_mps > 0.0
            else self.last_reliable_flow_speed_mps
            if reliable_flow_age_s <= _number(
                self.config.get("rear_side_flow_hold_s"), 1.0
            ) else 0.0
        )
        input_throttle = _number(getattr(control, "throttle", 0.0))
        input_brake = _number(getattr(control, "brake", 0.0))
        flow_target_speed_mps = min(
            _number(self.config.get("flow_speed_soft_mps"), 8.2),
            retained_flow_speed_mps,
        ) if retained_flow_speed_mps > 0.0 else 0.0
        flow_recovery_error_mps = max(0.0, flow_target_speed_mps - speed)
        flow_recovery_throttle = min(
            1.0,
            _number(self.config.get("flow_recovery_throttle_base"), 0.65)
            + _number(self.config.get("flow_recovery_throttle_gain"), 0.15)
            * flow_recovery_error_mps,
        )
        if self.right_obstacle_latched or self.right_obstacle_plan_completed:
            # The dedicated obstacle chain owns this physical event from
            # first stop through either local-plan completion or dynamic
            # clearance.  Residual sparse returns may still satisfy the
            # legacy oblique-conflict detector, but must never create a
            # second lateral controller in parallel.
            self.fast_crossing_conflict = False
            self.post_fast_follow_active = False
            self.post_fast_follow_miss_frames = 0
        state = "FLOW_MATCH"
        left_clear = bool(getattr(features, "left_clear", False)) and not bool(
            getattr(features, "side_risk", False)
        )
        right_clear = bool(getattr(features, "right_clear", False)) and not bool(
            getattr(features, "side_risk", False)
        )
        # Lock a physically drivable corridor before evaluating the lateral
        # commit gate.  A merging actor crosses the radar centreline, so the
        # sign of its lateral centroid is not a stable road-side label.  The
        # static boundary estimator is already ego-compensated and smoothed;
        # use repeated evidence for the wider online corridor instead.
        if (
            flow["conflict_active"]
            and not self.right_obstacle_latched
            and not self.right_obstacle_plan_completed
            and not self.merge_entry_follow_lifecycle_observed
            and (slow_queue_bypass or not self.longitudinal_merge_latched)
            and self.avoidance_side_sign == 0.0
        ):
            left_boundary = flow["static_left_clearance_m"]
            right_boundary = flow["static_right_clearance_m"]
            clearance_margin = 0.0
            if left_clear and not right_clear and left_boundary > 0.0:
                clearance_margin = left_boundary
            elif right_clear and not left_clear and right_boundary > 0.0:
                clearance_margin = -right_boundary
            elif left_clear and right_clear:
                if left_boundary > 0.0 and right_boundary > 0.0:
                    clearance_margin = left_boundary - right_boundary
            if clearance_margin != 0.0:
                self.clearance_side_history.append(clearance_margin)
        if (
            self.avoidance_side_sign == 0.0
            and slow_queue_corridor_preferred_sign != 0.0
            and not self.right_obstacle_latched
            and not self.right_obstacle_plan_completed
        ):
            self.avoidance_side_sign = slow_queue_corridor_preferred_sign
        if self.avoidance_side_sign == 0.0:
            required = max(3, int(_number(
                self.config.get("avoidance_clearance_window_samples"), 5
            )))
            if len(self.clearance_side_history) >= required:
                clearance_score = sum(
                    list(self.clearance_side_history)[-required:]
                )
                if clearance_score > 0.0:
                    self.avoidance_side_sign = 1.0
                elif clearance_score < 0.0:
                    self.avoidance_side_sign = -1.0
        selected_side_clear = (
            left_clear if self.avoidance_side_sign > 0.0
            else right_clear if self.avoidance_side_sign < 0.0
            else False
        )
        selected_boundary_clearance = (
            flow["static_left_clearance_m"]
            if self.avoidance_side_sign > 0.0
            else flow["static_right_clearance_m"]
            if self.avoidance_side_sign < 0.0
            else 0.0
        )
        if selected_boundary_clearance > 0.0:
            if (
                self.last_selected_boundary_clearance_m is not None
                and self.last_selected_boundary_timestamp is not None
                and timestamp > self.last_selected_boundary_timestamp
            ):
                dt = max(
                    0.05,
                    min(0.30, timestamp - self.last_selected_boundary_timestamp),
                )
                self.selected_boundary_closing_rate_mps = max(
                    0.0,
                    min(
                        _number(
                            self.config.get("boundary_max_closing_mps"), 1.5
                        ),
                        (
                            self.last_selected_boundary_clearance_m
                            - selected_boundary_clearance
                        ) / dt,
                    ),
                )
            self.last_selected_boundary_clearance_m = selected_boundary_clearance
            self.last_selected_boundary_timestamp = timestamp
        projected_boundary_clearance = selected_boundary_clearance
        if projected_boundary_clearance > 0.0:
            projected_boundary_clearance = max(
                0.0,
                projected_boundary_clearance
                - self.selected_boundary_closing_rate_mps * _number(
                    self.config.get("boundary_reaction_horizon_s"), 0.5
                ),
            )
        if (
            flow["conflict_active"]
            and not self.right_obstacle_latched
            and not self.right_obstacle_plan_completed
            and not self.merge_entry_follow_lifecycle_observed
            and (slow_queue_bypass or not self.longitudinal_merge_latched)
            and not self.avoidance_committed
        ):
            self.stop_confirm_frames = (
                self.stop_confirm_frames + 1
                if speed <= _number(
                    self.config.get("avoidance_stop_max_speed_mps"), 0.15
                )
                else 0
            )
            # Accumulate both online corridors from the first confirmed
            # conflict frame.  Side selection needs several coherent radar
            # samples and must not serially add another clearance delay.  If
            # both corridors are clear, preserve longitudinal momentum on the
            # centre route while the lateral sign is still resolving; publish
            # the offset directive only after the sign locks below.
            self.left_clear_confirm_frames = (
                self.left_clear_confirm_frames + 1 if left_clear else 0
            )
            self.right_clear_confirm_frames = (
                self.right_clear_confirm_frames + 1 if right_clear else 0
            )
            required_clear = max(1, int(_number(
                self.config.get("avoidance_clear_confirm_frames"), 3
            )))
            left_ready = self.left_clear_confirm_frames >= required_clear
            right_ready = self.right_clear_confirm_frames >= required_clear
            selected_ready = (
                left_ready if self.avoidance_side_sign > 0.0
                else right_ready if self.avoidance_side_sign < 0.0
                else False
            )
            if (
                not self.fast_crossing_conflict
                and ((left_ready and right_ready) or selected_ready)
            ):
                self.avoidance_committed = True
                self.avoidance_last_timestamp = timestamp
        elif (
            (
                not flow["conflict_active"]
                or self.merge_entry_follow_lifecycle_observed
            )
            and not self.avoidance_committed
        ):
            self.left_clear_confirm_frames = 0
            self.right_clear_confirm_frames = 0

        if self.avoidance_committed:
            if self.avoidance_last_timestamp is not None:
                dt = max(0.0, min(0.20, timestamp - self.avoidance_last_timestamp))
                self.avoidance_progress_m += speed * dt
            self.avoidance_last_timestamp = timestamp
            if (
                not flow["conflict_active"]
                and self.avoidance_clear_start_progress_m is None
            ):
                self.avoidance_clear_start_progress_m = self.avoidance_progress_m

        if self.right_obstacle_handoff_pending:
            result.throttle = 0.0
            result.brake = max(input_brake, 0.85)
            state = "RIGHT_OBSTACLE_HANDOFF_OBSERVE"
        elif self.right_obstacle_latched and not self.avoidance_committed:
            result.throttle = 0.0
            result.brake = max(input_brake, 0.85)
            state = (
                "RIGHT_OBSTACLE_DYNAMIC_CLEAR_WAIT"
                if self.lead_reversal_dynamic_wait_active
                else "RIGHT_OBSTACLE_STOP_AND_PLAN"
            )
        elif self._traffic_hold(
            features,
            ignore_vehicle_immediate=(
                self.merge_entry_follow_active
                or self.right_obstacle_plan is not None
            ),
        ):
            result.throttle = 0.0
            result.brake = max(input_brake, 0.85)
            state = "TRAFFIC_HOLD"
        elif self.right_obstacle_plan is not None:
            combined_half_width = (
                _number(self.config.get(
                    "right_obstacle_ego_half_width_m"), 1.0
                )
                + _number(self.config.get(
                    "right_obstacle_actor_half_width_m"), 1.05
                )
                + _number(self.config.get(
                    "right_obstacle_lateral_body_margin_m"), 0.35
                )
            )
            front_body_overlap = bool(
                self.right_obstacle_plan_phase == "pass"
                and front_tracked_vehicle["active"]
                and abs(front_tracked_vehicle["lateral_m"])
                <= combined_half_width
            )
            intrusion_ttc_s = (
                front_tracked_vehicle["distance_m"]
                / max(0.1, front_track_closing_rate_mps)
                if front_track_closing_rate_mps > 0.1
                else float("inf")
            )
            detour_escape_progress = (
                _number(local_path_feedback.get("x_m"), 0.0)
                if feedback_matches_plan else 0.0
            )
            detour_escape_in_progress = bool(
                self.right_obstacle_intrusion_detour_active
                and detour_escape_progress < _number(
                    self.config.get("right_obstacle_detour_shift_end_m"), 7.0
                )
            )
            initial_pass_escape_in_progress = bool(
                self.right_obstacle_plan_phase == "initial_shift"
                or (
                    self.right_obstacle_plan_phase == "pass"
                    and self.right_obstacle_latched
                    and not self.right_obstacle_pass_rear_seen
                )
            )
            raw_corridor_intrusion = bool(
                front_body_overlap
                and not detour_escape_in_progress
                # The original obstacle necessarily occupies the swept-body
                # envelope at the start of the plan built to avoid it.  Until
                # the initial lateral shift is complete it is expected input,
                # not a new intrusion.  Immediate re-latching here previously
                # stopped the plan before its first metre and handed control
                # to an unrelated detour.
                and not initial_pass_escape_in_progress
                and (
                    front_tracked_vehicle["distance_m"] <= _number(
                        self.config.get(
                            "right_obstacle_intrusion_hard_stop_m"), 10.0
                    )
                    or intrusion_ttc_s <= _number(
                        self.config.get(
                            "right_obstacle_intrusion_stop_ttc_s"), 3.2
                    )
                )
            )
            if raw_corridor_intrusion:
                self.right_obstacle_intrusion_latched = True
                self.right_obstacle_intrusion_clear_frames = 0
            elif self.right_obstacle_intrusion_latched:
                still_occupies_swept_corridor = bool(
                    front_body_overlap
                    and front_tracked_vehicle["distance_m"] <= _number(
                        self.config.get(
                            "right_obstacle_intrusion_release_m"), 25.0
                    )
                )
                if still_occupies_swept_corridor:
                    self.right_obstacle_intrusion_clear_frames = 0
                else:
                    self.right_obstacle_intrusion_clear_frames += 1
                    if self.right_obstacle_intrusion_clear_frames >= max(
                        1, int(_number(self.config.get(
                            "right_obstacle_intrusion_clear_frames"
                        ), 10))
                    ):
                        self.right_obstacle_intrusion_latched = False
                        self.right_obstacle_intrusion_clear_frames = 0
            self.right_obstacle_intrusion_hold_frames = (
                self.right_obstacle_intrusion_hold_frames + 1
                if self.right_obstacle_intrusion_latched
                and speed <= _number(
                    self.config.get("avoidance_stop_max_speed_mps"), 0.15
                )
                else 0
            )
            intrusion_replan_frames = max(1, int(_number(
                self.config.get("right_obstacle_intrusion_replan_stop_frames"),
                20,
            )))
            if front_track_closing_rate_mps >= _number(
                self.config.get(
                    "right_obstacle_intrusion_dynamic_closing_mps"), 0.5
            ):
                intrusion_replan_frames = max(1, int(_number(
                    self.config.get(
                        "right_obstacle_intrusion_dynamic_stop_frames"), 3
                )))
            detour_ready = bool(
                self.right_obstacle_intrusion_latched
                and self.right_obstacle_intrusion_hold_frames
                >= intrusion_replan_frames
                and front_tracked_vehicle["active"]
                and front_tracked_vehicle["distance_m"] >= _number(
                    self.config.get(
                        "right_obstacle_intrusion_detour_min_gap_m"), 5.0
                )
                and bool(getattr(features, "left_clear", False))
                and bool(getattr(features, "right_clear", False))
                and not bool(getattr(features, "side_risk", False))
                and self.right_obstacle_intrusion_detour_count < max(
                    1, int(_number(self.config.get(
                        "right_obstacle_intrusion_max_detours"
                    ), 2))
                )
            )
            if detour_ready:
                self.right_obstacle_intrusion_detour_count += 1
                self.right_obstacle_plan = (
                    self._build_right_obstacle_intrusion_detour(
                        route_lateral,
                        front_tracked_vehicle["lateral_m"],
                        combined_half_width,
                    )
                )
                self.right_obstacle_plan_phase = "detour_shift"
                self.right_obstacle_intrusion_detour_active = True
                self.right_obstacle_pass_progress_m = 0.0
                self.right_obstacle_pass_rear_seen = False
                self.right_obstacle_pass_clear_frames = 0
                self.right_obstacle_intrusion_latched = False
                self.right_obstacle_intrusion_clear_frames = 0
                self.right_obstacle_intrusion_hold_frames = 0
            if self.right_obstacle_intrusion_latched:
                result.throttle = 0.0
                result.brake = max(input_brake, 0.85)
                state = "RIGHT_OBSTACLE_PASS_INTRUSION_HOLD"
            else:
                detour_feedback_progress = (
                    _number(local_path_feedback.get("x_m"), 0.0)
                    if feedback_matches_plan else 0.0
                )
                detour_shift_active = bool(
                    (
                        self.right_obstacle_intrusion_detour_active
                        and detour_feedback_progress < _number(
                            self.config.get(
                                "right_obstacle_detour_shift_end_m"), 7.0
                        )
                    )
                    or self.right_obstacle_plan_phase == "initial_shift"
                )
                target_speed = _number(
                    self.config.get(
                        "right_obstacle_detour_target_speed_mps"
                        if detour_shift_active
                        else "right_obstacle_target_speed_mps"
                    ), 2.5 if detour_shift_active else 5.5
                )
                overspeed = speed - target_speed
                if overspeed > 0.5:
                    result.throttle = 0.0
                    result.brake = min(
                        _number(
                            self.config.get("right_obstacle_max_brake"), 0.55
                        ),
                        _number(
                            self.config.get("right_obstacle_brake_base"), 0.12
                        ) + _number(
                            self.config.get("right_obstacle_brake_gain"), 0.10
                        ) * overspeed,
                    )
                elif overspeed > 0.0:
                    result.throttle = 0.0
                    result.brake = _number(
                        self.config.get("right_obstacle_brake_base"), 0.12
                    )
                else:
                    result.brake = 0.0
                    result.throttle = min(
                        max(input_throttle, _number(
                            self.config.get(
                                "right_obstacle_throttle_floor"), 0.35
                        )),
                        _number(self.config.get(
                            "right_obstacle_throttle_cap"), 0.65
                        ),
                    )
                state = (
                    "RIGHT_OBSTACLE_INTRUSION_DETOUR"
                    if self.right_obstacle_intrusion_detour_active
                    else "RIGHT_OBSTACLE_LEFT_PASS"
                    if self.right_obstacle_plan_phase in {
                        "initial_shift", "pass"
                    }
                    else "RIGHT_OBSTACLE_EVENT_REJOIN"
                )
        elif self.merge_entry_follow_active:
            distance = front_tracked_vehicle["distance_m"]
            if front_tracked_vehicle["active"] and distance <= _number(
                self.config.get("merge_entry_follow_hard_gap_m"), 4.0
            ):
                result.throttle = 0.0
                result.brake = _number(
                    self.config.get("merge_entry_follow_hard_brake"), 0.45
                )
            elif front_tracked_vehicle["active"] and distance <= _number(
                self.config.get("merge_entry_follow_soft_gap_m"), 5.5
            ):
                result.throttle = 0.0
                result.brake = _number(
                    self.config.get("merge_entry_follow_soft_brake"), 0.28
                )
            elif front_tracked_vehicle["active"]:
                result.brake = 0.0
                result.throttle = min(
                    max(input_throttle, 0.18),
                    _number(
                        self.config.get("merge_entry_follow_throttle_cap"), 0.35
                    ),
                )
            elif bool(getattr(features, "immediate_hazard", False)):
                # The box has just left the centre envelope but the physical
                # hazard is still asserted.  Briefly coast/decelerate without
                # committing to a stop in the live traffic stream.
                result.throttle = 0.0
                result.brake = _number(
                    self.config.get("merge_entry_follow_dropout_brake"), 0.20
                )
            else:
                # Once the lead actor has visibly cleared, promptly recover
                # toward the measured flow speed.  Holding the earlier 0.35
                # cap made ego decay to ~3 m/s in a 9 m/s stream and caused a
                # rear impact despite a clear forward path.
                if (
                    flow_target_speed_mps > 0.0
                    and speed >= flow_target_speed_mps + _number(
                        self.config.get("flow_recovery_overspeed_margin_mps"),
                        0.4,
                    )
                ):
                    result.throttle = 0.0
                    result.brake = _number(
                        self.config.get("flow_recovery_overspeed_brake"), 0.12
                    )
                else:
                    result.brake = 0.0
                    result.throttle = min(
                        max(input_throttle, 0.35, flow_recovery_throttle),
                        _number(
                            self.config.get(
                                "merge_entry_follow_clear_throttle_cap"
                            ), 1.0
                        ),
                    )
            state = "FLOW_ENTRY_FOLLOW"
        elif self.merge_entry_follow_lifecycle_observed:
            # The detector box can disappear while the same lead actor is
            # still a dense radar cluster across the merge.  Keep that actor
            # under longitudinal merge ownership until its radar episode has
            # cleared; do not hand it to the crossing-obstacle side planner.
            if (
                flow_target_speed_mps > 0.0
                and speed >= flow_target_speed_mps + _number(
                    self.config.get("flow_recovery_overspeed_margin_mps"), 0.4
                )
            ):
                result.throttle = 0.0
                result.brake = _number(
                    self.config.get("flow_recovery_overspeed_brake"), 0.12
                )
            else:
                result.brake = 0.0
                result.throttle = min(
                    max(input_throttle, 0.35, flow_recovery_throttle),
                    _number(
                        self.config.get(
                            "merge_entry_follow_clear_throttle_cap"
                        ), 1.0
                    ),
                )
            state = "FLOW_ENTRY_CLEAR"
        else:
            speed_soft = _number(self.config.get("flow_speed_soft_mps"), 8.2)
            speed_hard = _number(self.config.get("flow_speed_hard_mps"), 9.0)
            minimum_gap = _number(self.config.get("flow_min_gap_m"), 5.0)
            desired_gap = minimum_gap + _number(
                self.config.get("flow_time_headway_s"), 0.8
            ) * speed
            distance = flow["front_distance_m"]
            closing = flow["closing_rate_mps"]
            conflict_distance = flow["conflict_cluster_distance_m"]
            stopping_envelope = (
                _number(self.config.get("conflict_base_stop_margin_m"), 3.0)
                + speed * speed / max(
                    1.0,
                    2.0 * _number(
                        self.config.get("conflict_assumed_decel_mps2"), 4.0
                    ),
                )
                + _number(
                    self.config.get("conflict_reaction_time_s"), 0.25
                ) * speed
            )

            if self.merge_gap_wait_active and not self.merge_gap_committed:
                result.throttle = 0.0
                result.brake = max(input_brake, _number(
                    self.config.get("merge_gap_yield_brake"), 0.75
                ))
                state = "MERGE_GAP_YIELD"
            elif (
                flow["conflict_active"]
                and (slow_queue_bypass or not self.longitudinal_merge_latched)
                and not self.avoidance_committed
                and conflict_distance > 0.0
                and conflict_distance <= stopping_envelope
            ):
                result.throttle = 0.0
                result.brake = max(input_brake, _number(
                    self.config.get("conflict_hard_brake"), 0.75
                ))
                state = "FLOW_CONFLICT_HARD_BRAKE"
            elif (
                flow["conflict_active"]
                and (slow_queue_bypass or not self.longitudinal_merge_latched)
                and not self.avoidance_committed
            ):
                result.throttle = 0.0
                result.brake = max(input_brake, _number(
                    self.config.get("conflict_yield_brake"), 0.45
                ))
                state = (
                    "FLOW_CONFLICT_YIELD"
                    if conflict_distance > 0.0
                    else "FLOW_CONFLICT_CLEAR_CONFIRM"
                )
            elif (
                distance > 0.0
                and (
                    distance <= _number(
                        self.config.get("flow_hard_stop_distance_m"), 4.0
                    )
                    or (
                        distance <= minimum_gap
                        and closing >= _number(
                            self.config.get("flow_hard_closing_mps"), 0.5
                        )
                    )
                )
            ):
                result.throttle = 0.0
                result.brake = max(input_brake, _number(
                    self.config.get("flow_hard_brake"), 0.70
                ))
                state = "FLOW_HARD_BRAKE"
            elif (
                distance > 0.0
                and distance < desired_gap
                and closing >= _number(
                    self.config.get("flow_follow_closing_mps"), 0.15
                )
            ):
                gap_error = max(0.0, desired_gap - distance)
                result.throttle = min(input_throttle, 0.08)
                result.brake = max(input_brake, min(
                    _number(self.config.get("flow_follow_max_brake"), 0.35),
                    0.08 + 0.04 * gap_error + 0.04 * closing,
                ))
                state = "FLOW_HEADWAY_BRAKE"
            elif (
                self.merge_gap_committed
                and front_tracked_vehicle["active"]
            ):
                # A selected traffic gap can still contain its leading
                # vehicle.  v56 launched toward the captured 8.2 m/s road
                # speed although that lead was only 9.8 m ahead and moving
                # much more slowly, producing a direct rear-end collision.
                # Use a proportional low-level follower throughout the
                # commit.  The target combines measured lead speed with the
                # signed time-headway error, so it accelerates into a growing
                # gap but brakes before the ego body reaches the lead.
                tracked_distance = front_tracked_vehicle["distance_m"]
                tracked_desired_gap = minimum_gap + _number(
                    self.config.get("flow_time_headway_s"), 0.8
                ) * speed
                estimated_lead_speed = max(
                    0.0, speed - front_track_closing_rate_mps
                )
                tracked_target_speed = max(
                    0.0,
                    min(
                        _number(
                            self.config.get("merge_gap_commit_target_speed_mps"),
                            8.2,
                        ),
                        estimated_lead_speed
                        + _number(
                            self.config.get("merge_commit_lead_gap_gain"), 0.5
                        ) * (tracked_distance - tracked_desired_gap),
                    ),
                )
                tracked_speed_error = tracked_target_speed - speed
                if tracked_speed_error < -_number(
                    self.config.get("merge_commit_lead_deadband_mps"), 0.25
                ):
                    result.throttle = 0.0
                    result.brake = min(
                        _number(
                            self.config.get("merge_commit_lead_max_brake"), 0.85
                        ),
                        _number(
                            self.config.get("merge_commit_lead_brake_base"), 0.18
                        )
                        + _number(
                            self.config.get("merge_commit_lead_brake_gain"), 0.12
                        ) * (-tracked_speed_error),
                    )
                    state = "MERGE_GAP_COMMIT_LEAD_BRAKE"
                else:
                    result.brake = 0.0
                    result.throttle = min(
                        _number(
                            self.config.get("merge_commit_lead_max_throttle"),
                            0.65,
                        ),
                        max(
                            0.0,
                            _number(
                                self.config.get(
                                    "merge_commit_lead_throttle_base"
                                ), 0.25
                            )
                            + _number(
                                self.config.get(
                                    "merge_commit_lead_throttle_gain"
                                ), 0.15
                            ) * max(0.0, tracked_speed_error),
                        ),
                    )
                    state = "MERGE_GAP_COMMIT_LEAD_TRACK"
            elif (
                self.merge_gap_committed
                and (
                    self.merge_gap_commit_progress_m < _number(
                        self.config.get("merge_gap_commit_control_distance_m"),
                        25.0,
                    )
                    or rear_side_pressure["active"]
                )
            ):
                # Own the complete merge, not only the launch from rest.  A
                # fixed 7.5 m/s launch threshold released control while the
                # online stream was travelling near 9 m/s; model zero-throttle
                # frames then recreated a rear-end target.  Track the road
                # speed captured at commit until ego has travelled clear of
                # the merge envelope and the rear quarter is unoccupied.
                commit_target = max(
                    _number(
                        self.config.get("merge_gap_commit_target_speed_mps"),
                        8.2,
                    ),
                    self.merge_gap_commit_flow_target_speed_mps,
                )
                commit_target = min(speed_hard, commit_target)
                commit_error = commit_target - speed
                if commit_error <= -_number(
                    self.config.get("flow_recovery_overspeed_margin_mps"), 0.4
                ):
                    result.throttle = 0.0
                    # A committed local merge plan must own both axes.  The
                    # learned policy produced full-brake pulses here, which
                    # interrupted the lateral route-PID and turned ego into a
                    # rear-collision target.  Use the class controller's
                    # bounded speed correction; higher-priority real hazards
                    # have already returned through the branches above.
                    result.brake = _number(
                        self.config.get("flow_recovery_overspeed_brake"), 0.12
                    )
                    state = "MERGE_GAP_COMMIT_SPEED_BRAKE"
                else:
                    result.brake = 0.0
                    required_throttle = min(
                        1.0,
                        _number(
                            self.config.get("flow_recovery_throttle_base"), 0.65
                        )
                        + _number(
                            self.config.get("flow_recovery_throttle_gain"), 0.15
                        ) * max(0.0, commit_error),
                    )
                    if commit_error >= _number(
                        self.config.get("rear_side_min_speed_error_mps"), 0.2
                    ):
                        required_throttle = max(
                            required_throttle,
                            _number(
                                self.config.get(
                                    "merge_gap_commit_throttle_floor"
                                ), 0.90,
                            ),
                        )
                    else:
                        required_throttle = max(required_throttle, 0.35)
                    result.throttle = max(input_throttle, required_throttle)
                    state = (
                        "MERGE_GAP_COMMIT_ACCEL"
                        if speed < _number(
                            self.config.get(
                                "merge_gap_commit_target_speed_mps"
                            ), 8.2,
                        )
                        else "MERGE_GAP_COMMIT_FLOW_TRACK"
                    )
            elif self.post_fast_follow_active:
                tracked_distance = front_tracked_vehicle["distance_m"]
                tracked_desired_gap = minimum_gap + _number(
                    self.config.get("post_fast_follow_time_headway_s"), 0.8
                ) * speed
                if not front_tracked_vehicle["active"]:
                    result.throttle = 0.0
                    result.brake = max(input_brake, _number(
                        self.config.get("post_fast_follow_lost_brake"), 0.25
                    ))
                    state = "POST_FAST_TAIL_LOST_HOLD"
                elif tracked_distance <= _number(
                    self.config.get("post_fast_follow_hard_stop_m"), 4.5
                ):
                    result.throttle = 0.0
                    result.brake = max(input_brake, _number(
                        self.config.get("flow_hard_brake"), 0.70
                    ))
                    state = "POST_FAST_TAIL_HARD_BRAKE"
                elif tracked_distance < tracked_desired_gap:
                    gap_error = tracked_desired_gap - tracked_distance
                    result.throttle = 0.0
                    result.brake = max(input_brake, min(
                        _number(
                            self.config.get("post_fast_follow_max_brake"), 0.35
                        ),
                        0.08 + 0.06 * gap_error,
                    ))
                    state = "POST_FAST_TAIL_HEADWAY_BRAKE"
                elif tracked_distance <= tracked_desired_gap + _number(
                    self.config.get("post_fast_follow_near_band_m"), 2.0
                ):
                    result.throttle = min(
                        input_throttle,
                        _number(
                            self.config.get("post_fast_follow_near_throttle"),
                            0.15,
                        ),
                    )
                    result.brake = 0.0
                    state = "POST_FAST_TAIL_NEAR"
                else:
                    result.throttle = min(
                        max(
                            input_throttle,
                            _number(self.config.get(
                                "post_fast_follow_throttle_floor"
                            ), 0.20),
                        ),
                        _number(self.config.get(
                            "post_fast_follow_throttle_cap"
                        ), 0.35),
                    )
                    result.brake = 0.0
                    state = "POST_FAST_TAIL_FOLLOW"
            elif speed >= speed_hard:
                result.throttle = 0.0
                result.brake = max(input_brake, _number(
                    self.config.get("flow_speed_brake"), 0.18
                ))
                state = "FLOW_SPEED_BRAKE"
            elif speed >= speed_soft:
                result.throttle = min(
                    input_throttle,
                    _number(self.config.get("flow_speed_throttle_cap"), 0.12),
                )
                result.brake = 0.0
                state = "FLOW_SPEED_CAP"
            else:
                result.throttle = min(
                    input_throttle,
                    _number(self.config.get("flow_accel_throttle_cap"), 0.65),
                )
                result.brake = 0.0

                # The normal branch used to be only a throttle ceiling.  On
                # route 3785 that let ego remain roughly 2 m/s slower than a
                # clearly observed same-direction flow while an adjacent car
                # closed from the rear quarter.  In this narrowly supported
                # merge state, track the (radar-derived) flow speed with a
                # proportional throttle floor.  Every hazard, headway,
                # conflict, speed and traffic-control branch above retains
                # priority and can never be bypassed by this floor.
                flow_target_speed = min(speed_soft, retained_flow_speed_mps)
                flow_speed_error = max(0.0, flow_target_speed - speed)
                if (
                    rear_side_pressure["active"]
                    and not self.avoidance_committed
                    and not flow["conflict_active"]
                    and retained_flow_speed_mps > 0.0
                    and flow_speed_error >= _number(
                        self.config.get("rear_side_min_speed_error_mps"), 0.2
                    )
                ):
                    pressure_throttle = min(
                        1.0,
                        _number(
                            self.config.get("rear_side_throttle_base"), 0.65
                        )
                        + _number(
                            self.config.get("rear_side_throttle_gain"), 0.15
                        ) * flow_speed_error,
                    )
                    result.throttle = max(result.throttle, pressure_throttle)
                    state = "FLOW_REAR_SIDE_SPEED_TRACK"
                elif (
                    (
                        flow["front_count"] >= minimum_reliable_flow_points
                        or self.longitudinal_merge_latched
                    )
                    and not self.avoidance_committed
                    and retained_flow_speed_mps > 0.0
                    and flow_recovery_error_mps >= _number(
                        self.config.get("flow_recovery_min_error_mps"), 0.5
                    )
                ):
                    result.throttle = max(
                        result.throttle, flow_recovery_throttle
                    )
                    state = "FLOW_SPEED_RECOVERY"

            if (
                self.avoidance_committed
                and state == "FLOW_MATCH"
                and speed < speed_soft
            ):
                result.throttle = max(
                    result.throttle,
                    _number(self.config.get("avoidance_throttle_floor"), 0.65),
                )
                result.brake = 0.0
                state = "FLOW_AVOIDANCE_ROUTE"

        if result.brake > 0.01:
            result.throttle = 0.0
        debug = {
            "source": "ordinary_class_16_flow_follow",
            "state": state,
            "flow_observation": flow,
            "flow_seen": bool(self.flow_seen),
            "flow_clear_frames": int(self.flow_clear_frames),
            "rear_side_pressure": rear_side_pressure,
            "rear_stationary_confirmed": bool(rear_stationary_confirmed),
            "rear_stationary_sample_count": int(
                len(self.rear_side_stationary_distance_history)
            ),
            "rear_stationary_span_m": (
                float(rear_stationary_span_m)
                if math.isfinite(rear_stationary_span_m) else None
            ),
            "stationary_rear_pocket_ready": bool(
                locals().get("stationary_rear_pocket_ready", False)
            ),
            "stationary_front_room_ready": bool(
                locals().get("stationary_front_room_ready", False)
            ),
            "retained_flow_speed_mps": float(retained_flow_speed_mps),
            "flow_target_speed_mps": float(flow_target_speed_mps),
            "flow_recovery_error_mps": float(flow_recovery_error_mps),
            "flow_recovery_throttle": float(flow_recovery_throttle),
            "merge_flow_target_speed_mps": float(
                self.merge_flow_target_speed_mps
            ),
            "reliable_flow_age_s": (
                float(reliable_flow_age_s)
                if math.isfinite(reliable_flow_age_s) else None
            ),
            "fast_crossing_conflict": bool(self.fast_crossing_conflict),
            "conflict_motion_mode": self.conflict_motion_mode,
            "slow_queue_bypass": bool(slow_queue_bypass),
            "flow_corridor_margin_sampled": bool(
                flow_corridor_margin_sampled
            ),
            "flow_corridor_margin_sample_count": int(
                len(self.flow_corridor_margin_history)
            ),
            "slow_queue_corridor_median_m": float(
                slow_queue_corridor_median_m
            ),
            "slow_queue_corridor_preferred_sign": float(
                slow_queue_corridor_preferred_sign
            ),
            "slow_queue_momentum_min_speed_mps": float(
                slow_queue_momentum_min_speed_mps
            ),
            "conflict_speed_history": [
                float(value) for value in self.conflict_speed_history
            ],
            "post_fast_follow_active": bool(self.post_fast_follow_active),
            "post_fast_follow_miss_frames": int(
                self.post_fast_follow_miss_frames
            ),
            "front_tracked_vehicle": front_tracked_vehicle,
            "front_track_closing_rate_mps": float(
                front_track_closing_rate_mps
            ),
            "right_obstacle_candidate": bool(right_obstacle_candidate),
            "fast_flow_merge_owns_right_actor": bool(
                fast_flow_merge_owns_right_actor
            ),
            "right_side_vehicle": right_side_vehicle,
            "startup_vehicle_handoff": startup_vehicle_handoff,
            "stopped_handoff_candidate": stopped_handoff_candidate,
            "right_obstacle_latched": bool(self.right_obstacle_latched),
            "right_obstacle_handoff_pending": bool(
                self.right_obstacle_handoff_pending
            ),
            "right_obstacle_handoff_observe_frames": int(
                self.right_obstacle_handoff_observe_frames
            ),
            "right_obstacle_handoff_resolved_as_flow": bool(
                self.right_obstacle_handoff_resolved_as_flow
            ),
            "lead_reversal_obstacle_confirm_frames": int(
                self.lead_reversal_obstacle_confirm_frames
            ),
            "lead_reversal_obstacle_latched": bool(
                self.lead_reversal_obstacle_latched
            ),
            "lead_reversal_dynamic_wait_active": bool(
                self.lead_reversal_dynamic_wait_active
            ),
            "lead_reversal_front_clear_frames": int(
                self.lead_reversal_front_clear_frames
            ),
            "lead_reversal_stationary_frames": int(
                self.lead_reversal_stationary_frames
            ),
            "estimated_lead_speed_mps": (
                float(estimated_lead_speed_mps)
                if math.isfinite(estimated_lead_speed_mps) else None
            ),
            "right_obstacle_stop_frames": int(
                self.right_obstacle_stop_frames
            ),
            "right_obstacle_left_clear_frames": int(
                self.right_obstacle_left_clear_frames
            ),
            "right_obstacle_plan_active": bool(
                self.right_obstacle_plan is not None
            ),
            "right_obstacle_plan_completed": bool(
                self.right_obstacle_plan_completed
            ),
            "right_obstacle_plan_phase": str(
                self.right_obstacle_plan_phase
            ),
            "right_obstacle_pass_rear_seen": bool(
                self.right_obstacle_pass_rear_seen
            ),
            "right_obstacle_pass_clear_frames": int(
                self.right_obstacle_pass_clear_frames
            ),
            "right_obstacle_pass_progress_m": float(
                self.right_obstacle_pass_progress_m
            ),
            "right_obstacle_intrusion_latched": bool(
                self.right_obstacle_intrusion_latched
            ),
            "right_obstacle_intrusion_clear_frames": int(
                self.right_obstacle_intrusion_clear_frames
            ),
            "right_obstacle_intrusion_hold_frames": int(
                self.right_obstacle_intrusion_hold_frames
            ),
            "right_obstacle_intrusion_detour_active": bool(
                self.right_obstacle_intrusion_detour_active
            ),
            "right_obstacle_intrusion_detour_count": int(
                self.right_obstacle_intrusion_detour_count
            ),
            "right_obstacle_intrusion_replan_frames": int(
                intrusion_replan_frames
                if self.right_obstacle_plan is not None else 0
            ),
            "right_obstacle_gap_required_s": float(
                self.right_obstacle_gap_required_s
            ),
            "right_obstacle_side_clear_elapsed_s": (
                float(timestamp - self.right_obstacle_last_side_occupied_timestamp)
                if self.right_obstacle_last_side_occupied_timestamp is not None
                else 0.0
            ),
            "right_obstacle_local_feedback": dict(local_path_feedback),
            "lead_departure_commit": bool(self.lead_departure_commit),
            "merge_entry_follow_active": bool(self.merge_entry_follow_active),
            "merge_entry_follow_lifecycle_observed": bool(
                self.merge_entry_follow_lifecycle_observed
            ),
            "merge_entry_follow_clear_frames": int(
                self.merge_entry_follow_clear_frames
            ),
            "longitudinal_merge_latched": bool(
                self.longitudinal_merge_latched
            ),
            "merge_gap_wait_active": bool(self.merge_gap_wait_active),
            "merge_gap_near_pass_seen": bool(self.merge_gap_near_pass_seen),
            "merge_gap_near_last_seen_timestamp": (
                float(self.merge_gap_near_last_seen_timestamp)
                if self.merge_gap_near_last_seen_timestamp is not None
                else None
            ),
            "merge_gap_clear_frames": int(self.merge_gap_clear_frames),
            "merge_gap_consecutive_empty_frames": int(
                self.merge_gap_consecutive_empty_frames
            ),
            "merge_gap_safe_following_frames": int(
                self.merge_gap_safe_following_frames
            ),
            "merge_gap_committed": bool(self.merge_gap_committed),
            "merge_gap_commit_flow_target_speed_mps": float(
                self.merge_gap_commit_flow_target_speed_mps
            ),
            "merge_gap_commit_progress_m": float(
                self.merge_gap_commit_progress_m
            ),
            "merge_gap_commit_start_offset_m": float(
                self.merge_gap_commit_start_offset_m
            ),
            "merge_gap_lateral_rejoin_started": bool(
                self.merge_gap_lateral_rejoin_started
            ),
            "merge_gap_lateral_rejoin_start_progress_m": (
                float(self.merge_gap_lateral_rejoin_start_progress_m)
                if self.merge_gap_lateral_rejoin_start_progress_m is not None
                else None
            ),
            "merge_gap_last_rear_side_timestamp": (
                float(self.merge_gap_last_rear_side_timestamp)
                if self.merge_gap_last_rear_side_timestamp is not None
                else None
            ),
            "merge_gap_rear_clear_elapsed_s": (
                float(merge_gap_rear_clear_elapsed)
                if math.isfinite(merge_gap_rear_clear_elapsed) else None
            ),
            "merge_gap_required_rear_clear_s": float(
                merge_gap_required_rear_clear_s
                if self.longitudinal_merge_latched
                and self.merge_gap_wait_active
                and not self.merge_gap_committed
                else 0.0
            ),
            "merge_entry_follow_last_seen_timestamp": (
                float(self.merge_entry_follow_last_seen_timestamp)
                if self.merge_entry_follow_last_seen_timestamp is not None
                else None
            ),
            "avoidance_committed": bool(self.avoidance_committed),
            "avoidance_side_sign": float(self.avoidance_side_sign),
            "conflict_lateral_history": [
                float(value) for value in self.conflict_lateral_history
            ],
            "clearance_side_history": [
                float(value) for value in self.clearance_side_history
            ],
            "stop_confirm_frames": int(self.stop_confirm_frames),
            "left_clear": bool(left_clear),
            "right_clear": bool(right_clear),
            "selected_side_clear": bool(selected_side_clear),
            "left_clear_confirm_frames": int(self.left_clear_confirm_frames),
            "right_clear_confirm_frames": int(self.right_clear_confirm_frames),
            "selected_boundary_clearance_m": float(
                selected_boundary_clearance
            ),
            "selected_boundary_closing_rate_mps": float(
                self.selected_boundary_closing_rate_mps
            ),
            "projected_boundary_clearance_m": float(
                projected_boundary_clearance
            ),
            "avoidance_progress_m": float(self.avoidance_progress_m),
            "avoidance_clear_start_progress_m": self.avoidance_clear_start_progress_m,
            "conflict_history": [bool(value) for value in self.conflict_history],
            "ego_speed": speed,
            "static_background_ignored": bool(
                _number(radar.get("front_min", 0.0)) > 0.0
                and flow["front_count"] == 0
            ),
            "input_control": {
                "throttle": input_throttle,
                "steer": _number(getattr(control, "steer", 0.0)),
                "brake": input_brake,
            },
            "output_control": {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0), 0.0),
            },
        }
        # The measured class-16 failure is a reverse-rolling queue with fast
        # traffic closing from behind.  Brake while the route is unresolved;
        # once a side is supported by repeated radar evidence, retain the
        # remaining momentum and create only the vehicle-envelope clearance
        # that fits inside the measured merge-road corridor.
        # The directive contains no route identity, XML/map coordinate,
        # actor id, or evaluator metadata.
        if (
            self.right_obstacle_plan is not None
            and result.brake < 0.50
        ):
            debug["ordinary_local_path"] = dict(self.right_obstacle_plan)
        elif (
            self.longitudinal_merge_latched
            and not slow_queue_bypass
            and not self.avoidance_committed
            and (
                result.brake < 0.50
                or self.merge_gap_committed
            )
        ):
            merge_route_offset = 0.0
            merge_phase = "class16_longitudinal_merge_center"
            if (
                self.merge_gap_committed
                and abs(self.merge_gap_commit_start_offset_m) > 0.0
            ):
                merge_route_offset = self.merge_gap_commit_start_offset_m
                merge_phase = "class16_acceleration_lane_hold"
                if (
                    self.merge_gap_lateral_rejoin_started
                    and self.merge_gap_lateral_rejoin_start_progress_m
                    is not None
                ):
                    rejoin_progress = max(
                        0.0,
                        self.merge_gap_commit_progress_m
                        - self.merge_gap_lateral_rejoin_start_progress_m,
                    )
                    rejoin_fraction = min(
                        1.0,
                        rejoin_progress / max(
                            1.0,
                            _number(
                                self.config.get(
                                    "merge_gap_rejoin_distance_m"
                                ), 20.0
                            ),
                        ),
                    )
                    merge_route_offset *= 0.5 * (
                        1.0 + math.cos(math.pi * rejoin_fraction)
                    )
                    merge_phase = "class16_speed_matched_rejoin"
            debug["ordinary_dense_route_pid"] = {
                "active": True,
                "phase": merge_phase,
                "lookahead_base_m": _number(
                    self.config.get("avoidance_dense_lookahead_base_m"), 5.0
                ),
                "lookahead_speed_s": _number(
                    self.config.get("avoidance_dense_lookahead_speed_s"), 0.45
                ),
                "lookahead_min_m": 5.0,
                "lookahead_max_m": 9.0,
                "steer_limit": _number(
                    self.config.get("avoidance_dense_steer_limit"), 0.48
                ),
                "steer_rate_per_frame": _number(
                    self.config.get("avoidance_dense_steer_rate"), 0.10
                ),
                "pose_filter_frames": int(_number(
                    self.config.get("avoidance_dense_pose_filter_frames"), 7
                )),
                "target_route_offset_m": merge_route_offset,
            }
        elif (
            self.avoidance_committed
            and self.avoidance_side_sign != 0.0
            and result.brake < 0.50
        ):
            full_corridor_offset = self.avoidance_side_sign * _number(
                self.config.get("avoidance_clearance_offset_m"), 1.1
            )
            # A lateral waypoint is not itself a drivable lane-change path.
            # Grow the offset with zero slope at both ends so a moving commit
            # cannot turn a 2.5 m target step into a saturated steering pulse.
            entry_distance = max(1.0, _number(
                self.config.get("avoidance_entry_distance_m"), 18.0
            ))
            entry_fraction = min(
                1.0, max(0.0, self.avoidance_progress_m / entry_distance)
            )
            corridor_offset = full_corridor_offset * 0.5 * (
                1.0 - math.cos(math.pi * entry_fraction)
            )
            # Turn the nominal offset into a radar-constrained local path.
            # The correction equals the predicted geometric clearance deficit
            # and can only pull the target back toward the online route; it
            # cannot invent a move to the opposite side.
            boundary_required = _number(
                self.config.get("boundary_required_clearance_m"), 1.27
            )
            boundary_correction = 0.0
            if projected_boundary_clearance > 0.0:
                raw_boundary_deficit = max(
                    0.0, boundary_required - projected_boundary_clearance
                )
                self.boundary_predicted_deficit_frames = (
                    self.boundary_predicted_deficit_frames + 1
                    if raw_boundary_deficit > 0.0 else 0
                )
                if (
                    selected_boundary_clearance < boundary_required
                    or self.boundary_predicted_deficit_frames >= max(
                        1, int(_number(self.config.get(
                            "boundary_deficit_confirm_frames"
                        ), 2))
                    )
                ):
                    boundary_correction = raw_boundary_deficit
                    corridor_offset = self.avoidance_side_sign * max(
                        0.0, abs(corridor_offset) - boundary_correction
                    )
            else:
                self.boundary_predicted_deficit_frames = 0
            rejoin_distance = _number(
                self.config.get("avoidance_rejoin_distance_m"), 15.0
            )
            if self.avoidance_clear_start_progress_m is not None:
                rejoin_progress = max(
                    0.0,
                    self.avoidance_progress_m
                    - self.avoidance_clear_start_progress_m,
                )
                fraction = min(1.0, rejoin_progress / max(1.0, rejoin_distance))
                corridor_offset *= 0.5 * (1.0 + math.cos(math.pi * fraction))
            debug["ordinary_dense_route_pid"] = {
                "active": True,
                "phase": "class16_dense_merge_clearance",
                "lookahead_base_m": _number(
                    self.config.get("avoidance_dense_lookahead_base_m"), 5.0
                ),
                "lookahead_speed_s": _number(
                    self.config.get("avoidance_dense_lookahead_speed_s"), 0.45
                ),
                "lookahead_min_m": 5.0,
                "lookahead_max_m": 9.0,
                "steer_limit": _number(
                    self.config.get("avoidance_dense_steer_limit"), 0.48
                ),
                "steer_rate_per_frame": _number(
                    self.config.get("avoidance_dense_steer_rate"), 0.10
                ),
                "pose_filter_frames": int(_number(
                    self.config.get("avoidance_dense_pose_filter_frames"), 7
                )),
                "target_route_offset_m": corridor_offset,
                "entry_fraction": entry_fraction,
                "boundary_required_clearance_m": boundary_required,
                "boundary_correction_m": boundary_correction,
                "boundary_predicted_deficit_frames": int(
                    self.boundary_predicted_deficit_frames
                ),
            }
        return result, debug


def build_rule():
    return Rule16(get_config(16))
