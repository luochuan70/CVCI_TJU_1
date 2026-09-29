"""Class-33 signalized left turn across continuous opposing flow.

Approach on the live route, stop at the turn entry, wait for a sensor-confirmed
green phase, then accept a measured opening in the opposing stream. Runtime
decisions consume online sensors/navigation only; route identity, XML and map
metadata are deliberately absent.
"""

from __future__ import absolute_import

import math

import numpy as np

from .common import clone_control, feature_value, _number
from .configs import get_config


class Rule33(object):
    APPROACH_JUNCTION = "APPROACH_JUNCTION"
    WAIT_FOR_GREEN = "WAIT_FOR_GREEN"
    WAIT_FOR_GAP = "WAIT_FOR_GAP"
    ROLLING_PREPARE = "ROLLING_PREPARE"
    COMMITTED_CROSSING = "COMMITTED_CROSSING"
    ROUTE_FOLLOW = "ROUTE_FOLLOW"

    def __init__(self, config):
        self.config = dict(config)
        self.reset()

    def reset(self):
        self.state = self.APPROACH_JUNCTION
        self.green_frames = 0
        self.stop_frames = 0
        self.pass_seen = False
        self.gap_frames = 0
        self.current_gap_safe_frames = 0
        self.empty_frames = 0
        self.verified_empty_tail_frames = 0
        self.committed_distance_m = 0.0
        self.last_timestamp = None
        self.signal_region = None
        self.signal_target_xy = None
        self.signal_red_armed = False
        self.signal_red_frames = 0
        self.signal_green_frames = 0
        self.direct_green_target_xy = None
        self.signal_red_tracks = []
        self.signal_track_serial = 0
        self.signal_observation_index = 0
        self.signal_last_observation_timestamp = None
        self.signal_pending_green_track_id = None
        self.signal_pending_green_frames = 0
        self.signal_direct_green_frames = 0
        self.signal_flow_seen = False
        self.signal_flow_empty_frames = 0
        self.signal_governing_phase = "unknown"
        self.signal_governing_green_xy = None
        self.prelock_gap_frames = 0
        self.prelock_gap_last_timestamp = None
        self.prelock_gap_observation = None
        self.prelock_crossing_armed = False
        self.prelock_flow_pass_handoff = False
        self.prelock_approach_handoff = False
        self.prelock_turn_distance_m = None
        self.prelock_turn_last_timestamp = None
        self.prelock_turn_unobserved_progress_m = 0.0
        self.prelock_approach_progress_m = 0.0
        # Reaching the current junction stop line is a discrete spatial event.
        # The online navigation horizon may advance to the *next* turn while
        # the ego is still decelerating, so a later, larger turn distance must
        # not revoke the hard stop before the vehicle is stationary.
        self.junction_reached_latched = False
        self.junction_wait_distance_m = None
        self.wait_for_gap_started_at = None
        # Preserve the online geometry seen on approach.  Once the route
        # horizon advances beyond the current junction, its far heading can
        # describe the *next* turn and must not move a sharp junction's safe
        # waiting line into the opposing lane.
        self.sharp_crossing_latched = False
        self.sharp_safe_wait_latched = False
        self.timed_slot_frames = 0
        self.timed_slot_observation = None
        self.timed_slot_mode = "none"
        self.rolling_prepare_frames = 0
        self.rolling_mode = "none"
        self.mild_rolling_attempts = 0
        self.mild_rolling_cumulative_distance_m = 0.0
        self.mild_rolling_abort_braking = False
        self.mild_rolling_turn_chord_latched = False
        self.mild_rolling_candidate_observation = None
        self.mild_rolling_candidate_timestamp = None
        self.mild_rolling_candidate_frames = 0
        self.rolling_commit_frames = 0
        self.rolling_distance_m = 0.0
        self.rolling_started_at = None
        self.rolling_attempted = False
        self.rolling_pair_observation = None
        self.rolling_pair_last_timestamp = None
        self.rolling_pair_last_dual_timestamp = None
        self.rolling_pair_predicted = False
        self.rolling_pair_leader_supported = False
        self.rolling_pair_track_frames = 0
        self.rolling_pair_acquired = False
        self.rolling_pair_continuous = False
        self.rolling_pair_residuals = None
        self.rolling_commit_last_timestamp = None
        self.rolling_staged = False
        self.rolling_leader_tail_seen = False
        self.rolling_post_pass_authorized = False
        self.rolling_turn_arc_started = False
        self.rolling_turn_arc_start_distance_m = None
        self.sharp_commit_leader_tail_near = False
        self.sharp_commit_next_actor_seen = False
        self.sharp_commit_pass_frames = 0
        self.sharp_commit_leader_passed = False
        self.sharp_arc_started_distance_m = None
        self.local_bypass_active = False
        self.local_bypass_side = 0.0
        self.local_bypass_clear_frames = 0
        self.local_bypass_started_distance_m = None
        self.local_bypass_recovery_active = False
        self.local_bypass_recovery_until_distance_m = None
        self.local_bypass_completed = False
        self.local_bypass_hazard_hold_frames = 0
        self.committed_conflict_frames = 0
        self.committed_replan_attempts = 0

    def _centered_slow_blocker(self, features):
        """Return true for a near centred vehicle that can be passed locally."""
        max_distance = _number(
            self.config.get("class33_local_bypass_max_obstacle_distance_m"), 6.0
        )
        max_speed = _number(
            self.config.get("class33_local_bypass_max_obstacle_speed_mps"), 1.0
        )
        min_score = _number(
            self.config.get("class33_crossing_detector_min_score"), 0.20
        )
        for item in list(getattr(features, "detector_objects", None) or []):
            try:
                if str(item.get("class_name", "")).lower() not in {
                    "car", "truck", "bus", "van",
                }:
                    continue
                if float(item.get("score", 1.0)) < min_score:
                    continue
                box = item.get("box_lidar") or {}
                # DriveTransformer's legacy boxes are x-lateral/y-forward.
                # The previous swap interpreted an adjacent vehicle as a
                # centred blocker and could launch a needless bypass.
                lateral = abs(float(box.get("x")))
                longitudinal = float(box.get("y"))
                velocity = math.hypot(float(box.get("vx", 0.0)), float(box.get("vy", 0.0)))
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                0.5 <= longitudinal <= max_distance
                and lateral <= 1.5
                and velocity <= max_speed
            ):
                return True
        return False

    def _local_bypass_clear(self, features, opposing):
        """Require repeated left clearance and no imminent opposing conflict."""
        left_clear = bool(getattr(features, "left_clear", False)) and not bool(
            getattr(features, "side_risk", False)
        )
        safe_flow = bool(
            opposing.get("count", 0) == 0
            or (
                opposing.get("ttc_s", 0.0) >= 2.2
                and opposing.get("follower_ttc_s", 0.0) >= 3.8
            )
        )
        if left_clear and safe_flow:
            self.local_bypass_clear_frames += 1
        else:
            self.local_bypass_clear_frames = 0
        return self.local_bypass_clear_frames >= int(_number(
            self.config.get("class33_local_bypass_min_clear_frames"), 3
        ))

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

    def _opposing_observation(self, radar, speed):
        empty = {
            "count": 0, "distance_m": 0.0, "ttc_s": 0.0,
            "closing_speed_mps": 0.0, "azimuth_deg": 0.0,
            "cluster_count": 0, "follower_distance_m": 0.0,
            "follower_ttc_s": 0.0, "headway_s": 0.0,
        }
        points = self._packet_points(radar.get("_online_radar_packet"))
        if len(points) == 0:
            return empty
        depth, azimuth, velocity = points[:, 0], points[:, 2], points[:, 3]
        relative_velocity = velocity + abs(speed) * np.cos(azimuth)
        mask = (
            (depth > 0.0)
            & (depth <= _number(self.config.get("opposing_radar_max_m"), 60.0))
            & (np.abs(azimuth) <= math.radians(_number(
                self.config.get("opposing_radar_half_angle_deg"), 60.0
            )))
            & (relative_velocity <= -_number(
                self.config.get("opposing_min_closing_speed_mps"), 0.5
            ))
        )
        if not np.any(mask):
            return empty
        indices = np.where(mask)[0]
        index = indices[int(np.argmin(depth[indices]))]
        distance = float(depth[index])
        closing_speed = max(0.1, -float(relative_velocity[index]))
        # A safe moving-slot plan needs both edges of the opening, not only
        # the next actor.  Cluster radar returns by depth while the ego is
        # stopped: points on one vehicle span roughly two metres, whereas
        # consecutive vehicles in this class are separated by 15--28 m.
        ordered = sorted(indices, key=lambda value: float(depth[value]))
        clusters = []
        split_m = _number(self.config.get("opposing_cluster_split_m"), 4.0)
        for candidate in ordered:
            if (
                not clusters
                or float(depth[candidate]) - float(depth[clusters[-1][-1]]) > split_m
            ):
                clusters.append([candidate])
            else:
                clusters[-1].append(candidate)

        def cluster_state(cluster):
            nearest = min(cluster, key=lambda value: float(depth[value]))
            cluster_distance = float(depth[nearest])
            cluster_closing = max(0.1, -float(relative_velocity[nearest]))
            return cluster_distance, cluster_distance / cluster_closing

        follower_distance = 0.0
        follower_ttc = 0.0
        if len(clusters) >= 2:
            follower_distance, follower_ttc = cluster_state(clusters[1])
        return {
            "count": int(len(indices)),
            "distance_m": distance,
            "ttc_s": distance / closing_speed,
            "closing_speed_mps": closing_speed,
            "azimuth_deg": math.degrees(float(azimuth[index])),
            "cluster_count": int(len(clusters)),
            "follower_distance_m": follower_distance,
            "follower_ttc_s": follower_ttc,
            "headway_s": max(0.0, follower_ttc - distance / closing_speed),
        }

    @staticmethod
    def _component_list(vote, name, minimum_area=8, maximum_area=500):
        result = []
        for item in vote.get(name + "_components", []) or []:
            try:
                area = int(item.get("area", 0))
                point = item.get("centroid")
                point = (float(point[0]), float(point[1]))
            except (AttributeError, TypeError, ValueError, IndexError):
                continue
            if minimum_area <= area <= maximum_area:
                result.append((area, point))
        return result

    def observe_signal_phase(self, features, timestamp=None):
        """Observe all visible heads before class lock and bind on red->green.

        A turn direction is not an image-space traffic-light identity.  The
        governing head is therefore selected only when one stable red blob is
        replaced by a vertically aligned green blob in the *same* camera
        region.  This observer is safe to run before Class33 is locked: it
        records sensor history but never emits control.
        """
        try:
            timestamp_key = float(timestamp) if timestamp is not None else None
        except (TypeError, ValueError):
            timestamp_key = None
        if (
            timestamp_key is not None
            and self.signal_last_observation_timestamp is not None
            and abs(timestamp_key - self.signal_last_observation_timestamp) < 1e-6
        ):
            return
        self.signal_last_observation_timestamp = timestamp_key
        self.signal_observation_index += 1

        detector_signal = False
        for item in list(getattr(features, "detector_objects", None) or []):
            try:
                detector_signal = bool(
                    str(item.get("class_name", "")).lower() == "traffic_light"
                    and float(item.get("score", 0.0) or 0.0) >= 0.45
                )
            except (AttributeError, TypeError, ValueError):
                detector_signal = False
            if detector_signal:
                break
        votes = getattr(features, "traffic_light_region_votes", {}) or {}

        def candidates(active_vote, color_name):
            strict = self._component_list(active_vote, color_name)
            dim = self._component_list(active_vote, "dim_" + color_name)
            return strict or dim

        visible_reds = []
        visible_greens = []
        for region in ("left", "center", "right"):
            vote = votes.get(region, {}) or {}
            visible_reds.extend(
                (region, area, point) for area, point in candidates(vote, "red")
            )
            visible_greens.extend(
                (region, area, point) for area, point in candidates(vote, "green")
            )

        # Match green against red history from earlier frames.  Keeping all
        # heads until a real phase transition avoids choosing the largest or
        # route-side red lamp, both of which can belong to another approach.
        green_matches = []
        for region, area, point in visible_greens:
            for track in self.signal_red_tracks:
                age = self.signal_observation_index - track["last_seen"]
                dx = abs(point[0] - track["xy"][0])
                dy = point[1] - track["xy"][1]
                # A detector box is useful corroboration, but it is not a
                # prerequisite for a real lamp.  Town12 fog intermittently
                # drops every traffic-light box while the RGB lamp remains
                # stable.  In that case require a substantially longer red
                # history before accepting the same-head red->green change.
                # This rejects a persistent unrelated green blob without
                # weakening the shared detector or colour thresholds.
                detector_supported = track.get("detector_frames", 0) >= int(
                    self.config.get("fog_signal_detector_confirm_frames", 2)
                )
                required_red_frames = int(self.config.get(
                    "fog_signal_red_confirm_frames", 2
                )) if detector_supported else int(self.config.get(
                    "fog_signal_no_detector_red_confirm_frames", 5
                ))
                if (
                    track["region"] == region
                    and track["frames"] >= required_red_frames
                    and age <= 30
                    and dx <= 16.0
                    and -5.0 <= dy <= 35.0
                ):
                    green_matches.append((track, area, point, dx, dy))
        if green_matches:
            match = min(
                green_matches,
                key=lambda item: (item[3], abs(item[4] - 10.0), -item[0]["frames"]),
            )
            track, area, point, dx, dy = match
            if self.signal_pending_green_track_id == track["id"]:
                self.signal_pending_green_frames += 1
            else:
                self.signal_pending_green_track_id = track["id"]
                self.signal_pending_green_frames = 1
            if self.signal_pending_green_frames >= int(
                self.config.get("fog_signal_green_confirm_frames", 3)
            ):
                self.signal_region = track["region"]
                self.signal_target_xy = track["xy"]
                self.signal_governing_green_xy = point
                self.signal_red_armed = True
                self.signal_governing_phase = "green"
                self.signal_green_frames = self.signal_pending_green_frames
        else:
            self.signal_pending_green_track_id = None
            self.signal_pending_green_frames = 0

        for region, area, point in visible_reds:
            matches = []
            for track in self.signal_red_tracks:
                if track["region"] != region:
                    continue
                distance = math.hypot(
                    point[0] - track["xy"][0], point[1] - track["xy"][1]
                )
                if distance <= 35.0:
                    matches.append((distance, track))
            if matches:
                track = min(matches, key=lambda item: item[0])[1]
                track["xy"] = point
                track["frames"] += 1
                track["last_seen"] = self.signal_observation_index
                track["area"] = area
                if detector_signal:
                    track["detector_frames"] = track.get("detector_frames", 0) + 1
            else:
                self.signal_track_serial += 1
                self.signal_red_tracks.append({
                    "id": self.signal_track_serial,
                    "region": region,
                    "xy": point,
                    "frames": 1,
                    "last_seen": self.signal_observation_index,
                    "area": area,
                    "detector_frames": 1 if detector_signal else 0,
                })
        self.signal_red_tracks = [
            track for track in self.signal_red_tracks
            if self.signal_observation_index - track["last_seen"] <= 30
        ]

    def _signal_color(self, features, radar):
        """Report a governing phase learned from online temporal evidence."""
        self.observe_signal_phase(features, radar.get("_timestamp"))
        color = str(
            getattr(features, "traffic_light_color", "unknown") or "unknown"
        ).lower()
        confidence = _number(
            getattr(features, "traffic_light_color_confidence", 0.0), 0.0
        )
        if getattr(features, "red_light_active", False):
            return "red", max(confidence, 1.0), {"source": "shared_red"}

        detector_signal = False
        for item in list(getattr(features, "detector_objects", None) or []):
            try:
                detector_signal = bool(
                    str(item.get("class_name", "")).lower() == "traffic_light"
                    and float(item.get("score", 0.0) or 0.0) >= 0.45
                )
            except (AttributeError, TypeError, ValueError):
                detector_signal = False
            if detector_signal:
                break
        if color in {"red", "yellow", "green"} and detector_signal:
            return color, confidence, {"source": "shared_strict"}
        if self.signal_governing_phase == "green":
            return "green", 0.90, {
                "source": "class33_prelock_red_to_green",
                "region": self.signal_region,
                "target_xy": self.signal_target_xy,
                "green_xy": self.signal_governing_green_xy,
                "green_confirm_frames": self.signal_green_frames,
            }

        # In dense fog the RGB lamp can remain visible while the 3-D detector
        # intermittently loses the exact box-to-pixel association. Use online
        # turn geometry only to choose the expected image region (never a
        # route identity), and require three strict green, red-free frames
        # plus a traffic-light detector before accepting a direct green.
        expected_region = (
            "right" if self.sharp_crossing_latched else "center"
        )
        expected_vote = (
            getattr(features, "traffic_light_region_votes", {}) or {}
        ).get(expected_region, {}) or {}
        expected_reds = self._component_list(expected_vote, "red")
        expected_greens = self._component_list(expected_vote, "green")
        direct_green = bool(
            detector_signal and expected_greens and not expected_reds
        )
        self.signal_direct_green_frames = (
            self.signal_direct_green_frames + 1 if direct_green else 0
        )
        if self.signal_direct_green_frames >= int(
            self.config.get("fog_signal_direct_green_confirm_frames", 3)
        ):
            self.signal_governing_phase = "green"
            self.signal_region = expected_region
            self.signal_governing_green_xy = expected_greens[0][1]
            return "green", 0.85, {
                "source": "class33_geometry_direct_green",
                "region": expected_region,
                "green_xy": self.signal_governing_green_xy,
                "green_confirm_frames": self.signal_direct_green_frames,
            }

        stable_tracks = [
            track for track in self.signal_red_tracks
            if track["frames"] >= int(
                self.config.get("fog_signal_red_confirm_frames", 2)
            )
            and track["region"] == expected_region
            and self.signal_observation_index - track["last_seen"] <= 2
        ]
        if detector_signal and stable_tracks:
            return "red", 0.70, {
                "source": "class33_fog_red_unbound",
                "regions": sorted(set(track["region"] for track in stable_tracks)),
                "stable_red_tracks": len(stable_tracks),
                "pending_green_frames": self.signal_pending_green_frames,
            }
        return "unknown", confidence, {
            "source": "class33_signal_unknown",
            "region": self.signal_region,
            "detector_signal": bool(detector_signal),
            "stable_red_tracks": len(stable_tracks),
            "pending_green_frames": self.signal_pending_green_frames,
        }

    def _at_junction(self, radar, features, distance=None, stop_distance=None):
        if distance is None:
            turn = radar.get("_route_turn_observation") or {}
            distance = turn.get("distance_to_turn_m")
        if distance is not None:
            try:
                if stop_distance is None:
                    stop_distance = _number(
                        self.config.get("junction_stop_distance_m"), 6.0
                    )
                # A valid online route-turn distance is the authoritative
                # white-line reference.  Do not fall through to the broader
                # traffic-light range and stop several metres early.
                return float(distance) <= float(stop_distance)
            except (TypeError, ValueError):
                pass
        red_distance = getattr(features, "red_stop_distance", None)
        try:
            return red_distance is not None and float(red_distance) <= _number(
                self.config.get("signal_stop_distance_m"), 7.5
            )
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _hard_stop(result):
        result.throttle = 0.0
        result.brake = max(_number(getattr(result, "brake", 0.0)), 0.85)

    def _nonflow_hazard(self, features, opposing):
        """Separate an ego-lane blocker from the expected opposing stream.

        The shared adapter intentionally uses a wide +/-2.6 m vehicle band,
        which is conservative for generic driving but includes the adjacent
        opposing lane at this junction.  Rule33 already measures that stream
        explicitly with radar.  A bounded stop-line approach may therefore
        ignore only the broad vehicle flag when tracked vehicle centres are
        outside the ego corridor and moving opposing-flow evidence exists.
        Pedestrians, construction/obstacle evidence, or any centred tracked
        vehicle remain unconditional blockers.
        """
        if getattr(features, "front_pedestrian_distance", None) is not None:
            return True
        obstacle_distance = getattr(features, "front_obstacle_distance", None)
        try:
            if (
                obstacle_distance is not None
                and 0.0 < float(obstacle_distance) <= _number(
                    self.config.get("approach_obstacle_stop_distance_m"), 6.0
                )
            ):
                return True
        except (TypeError, ValueError):
            return True
        if int(getattr(features, "construction_object_count", 0) or 0) > 0:
            return True
        if not bool(getattr(features, "immediate_hazard", False)):
            return False
        tracks = list(getattr(features, "tracked_objects", None) or [])
        if not tracks or opposing.get("count", 0) <= 0:
            return True
        corridor_half_width = _number(
            self.config.get("approach_corridor_half_width_m"), 1.8
        )
        hazard_distance = _number(
            self.config.get("approach_vehicle_stop_distance_m"), 5.0
        )
        for track in tracks:
            if str(track.get("class_name", "")).lower() not in {
                "car", "van", "truck", "bus", "motorcycle", "bicycle",
            }:
                continue
            try:
                longitudinal = float(track.get("x"))
                lateral = abs(float(track.get("y")))
            except (TypeError, ValueError):
                continue
            if (
                0.0 <= longitudinal <= hazard_distance
                and lateral <= corridor_half_width
            ):
                return True
        return False

    def _blocking_nonflow_hazard(self, features):
        """Return only physical blockers in the planned ego corridor.

        ``immediate_hazard`` and ``front_vehicle_distance`` deliberately use
        a wider generic band.  At Class33's sharp stop pose that band also
        contains a parked vehicle beside the route and must not cancel an
        otherwise valid rolling gap.  Pedestrians, explicit obstacle depth,
        construction objects, and a detector-confirmed centred slow vehicle
        remain hard blockers and are handled by the existing stop/bypass
        lifecycle.
        """
        if getattr(features, "front_pedestrian_distance", None) is not None:
            return True
        obstacle_distance = getattr(features, "front_obstacle_distance", None)
        vehicle_distance = getattr(features, "front_vehicle_distance", None)
        try:
            obstacle_distance = (
                float(obstacle_distance)
                if obstacle_distance is not None else None
            )
            vehicle_distance = (
                float(vehicle_distance)
                if vehicle_distance is not None else None
            )
            # The shared adapter aliases a vehicle detected anywhere in its
            # broad front band into both distance fields.  At this junction
            # that band includes the parked car beside the sharp-turn path.
            # Treat only a distinct obstacle-only return as unconditional;
            # vehicle geometry is checked below in the narrower Class33
            # corridor and can therefore enter the stop/left-bypass planner.
            aliased_vehicle_return = bool(
                obstacle_distance is not None
                and vehicle_distance is not None
                and abs(obstacle_distance - vehicle_distance) <= 0.25
            )
            if (
                obstacle_distance is not None
                and not aliased_vehicle_return
                and 0.0 < obstacle_distance <= _number(
                    self.config.get("approach_obstacle_stop_distance_m"), 6.0
                )
            ):
                return True
        except (TypeError, ValueError):
            return True
        if int(getattr(features, "construction_object_count", 0) or 0) > 0:
            return True
        return bool(self._centered_slow_blocker(features))

    def _waypoint_directive(self):
        near_switch = _number(
            self.config.get("crossing_waypoint_near_switch_distance_m"), 2.0
        )
        sharp_rolling_arc = bool(
            self.state == self.ROLLING_PREPARE
            and self.rolling_mode == "sharp"
            and self.rolling_turn_arc_started
        )
        sharp_committed_arc = bool(
            self.sharp_crossing_latched
            and (
                sharp_rolling_arc
                or (
                    self.state == self.COMMITTED_CROSSING
                    and self.sharp_commit_leader_passed
                )
            )
        )
        directive = {
            "active": True,
            "phase": (
                "class33_sharp_committed_arc"
                if sharp_committed_arc else "class33_committed_crossing"
            ),
            "steer_kp": _number(
                self.config.get("crossing_waypoint_steer_kp"), 0.72
            ),
            "model_steer_blend": _number(
                self.config.get(
                    "crossing_waypoint_model_blend_sharp"
                    if sharp_committed_arc
                    else "crossing_waypoint_model_blend_mild",
                    self.config.get("crossing_waypoint_model_blend"),
                ),
            ),
            "steer_limit": _number(
                self.config.get("crossing_waypoint_steer_limit"), 0.65
            ),
            "steer_rate_per_frame": _number(
                self.config.get("crossing_waypoint_steer_rate"), 0.10
            ),
            "high_speed_start_mps": 7.5,
            "high_speed_steer_limit": 0.34,
            "near_switch_distance_m": near_switch,
        }
        if sharp_committed_arc:
            # v41 proved that waiting for the leader tail is necessary, but a
            # fixed 55% chord still let the short near point pull the car back
            # to the right several times during the first six metres.  Shape
            # an actual local arc: establish lateral clearance with a stronger
            # (still bounded) far contribution, then taper to the v41 blend so
            # the exit follows the near route and cannot recreate v22's
            # full-far cut into roadside furniture.
            if sharp_rolling_arc:
                arc_start = _number(
                    self.rolling_turn_arc_start_distance_m,
                    self.rolling_distance_m,
                )
                arc_progress = max(0.0, self.rolling_distance_m - arc_start)
            else:
                arc_start = _number(
                    self.sharp_arc_started_distance_m,
                    self.committed_distance_m,
                )
                arc_progress = max(0.0, self.committed_distance_m - arc_start)
            strong_distance = max(
                0.1,
                _number(
                    self.config.get("sharp_arc_strong_distance_m"), 6.0
                ),
            )
            taper_distance = max(
                0.1,
                _number(
                    self.config.get("sharp_arc_taper_distance_m"), 3.0
                ),
            )
            base_blend = _number(
                self.config.get("sharp_crossing_heading_blend_far"), 0.55
            )
            strong_blend = _number(
                self.config.get("sharp_arc_strong_heading_blend_far"), 0.75
            )
            if arc_progress <= strong_distance:
                heading_blend_far = strong_blend
            elif arc_progress < strong_distance + taper_distance:
                taper = (arc_progress - strong_distance) / taper_distance
                heading_blend_far = (
                    strong_blend + taper * (base_blend - strong_blend)
                )
            else:
                heading_blend_far = base_blend
            directive["heading_blend_far"] = heading_blend_far
            directive["sharp_arc_progress_m"] = arc_progress
        elif self.mild_rolling_turn_chord_latched:
            # Blend only after the online near point has rotated farther than
            # the far chord, which is the sensor-visible signature that the
            # vehicle has entered the turn.  Applying this at commit time made
            # v60 turn across traffic before reaching the junction.
            directive["heading_blend_far"] = _number(
                self.config.get(
                    "mild_rolling_crossing_heading_blend_far"
                ),
                0.55,
            )
        return directive

    def _dense_route_directive(self):
        """Follow the live dense navigation horizon through the junction.

        The sparse near/far navigation pair is a safe fallback, but blending
        those two headings draws a chord across a tight junction.  The router
        already exposes the online dense route and the shared controller uses
        CARLA's public PID callback on it; Class33 opts into that existing
        path without reading route XML or route identity at runtime.
        """
        return {
            "active": True,
            "phase": "class33_dense_junction_follow",
            "target_route_offset_m": 0.0,
            "lookahead_base_m": _number(
                self.config.get("dense_pid_lookahead_base_m"), 4.5
            ),
            "lookahead_speed_s": _number(
                self.config.get("dense_pid_lookahead_speed_s"), 0.40
            ),
            "lookahead_min_m": _number(
                self.config.get("dense_pid_lookahead_min_m"), 4.0
            ),
            "lookahead_max_m": _number(
                self.config.get("dense_pid_lookahead_max_m"), 8.0
            ),
            "steer_limit": _number(
                self.config.get("dense_pid_steer_limit"), 0.52
            ),
            "steer_rate_per_frame": _number(
                self.config.get("dense_pid_steer_rate"), 0.10
            ),
            "pose_filter_frames": int(
                self.config.get("dense_pid_pose_filter_frames", 5)
            ),
        }

    def observe_prelock_context(self, radar, features, flow_reason):
        """Latch a safe Class33 handoff before the classifier finishes.

        Class33 and Class44 need several close-flow frames to separate, but a
        signalized left turn can expose its safest opening immediately after
        the governing head turns green.  Keep this gate entirely sensor based:
        the ego must already be stopped at the live turn entry, the same visual
        lamp must have completed a red-to-green transition, and two consecutive
        radar frames must show the large initial opening.  This method only
        latches ownership; it never changes control by itself.
        """
        timestamp = _number(radar.get("_timestamp"), 0.0)
        previous_timestamp = self.prelock_gap_last_timestamp
        if (
            previous_timestamp is not None
            and abs(timestamp - previous_timestamp) < 1e-6
        ):
            return bool(
                self.prelock_crossing_armed
                or self.prelock_flow_pass_handoff
                or self.prelock_approach_handoff
            )
        self.prelock_gap_last_timestamp = timestamp
        speed = max(0.0, feature_value(features, "ego_speed", 0.0))
        if (
            self.prelock_turn_distance_m is not None
            and previous_timestamp is not None
            and not (
                self.prelock_crossing_armed
                or self.prelock_flow_pass_handoff
                or self.prelock_approach_handoff
            )
        ):
            self.prelock_turn_unobserved_progress_m += speed * min(
                0.2, max(0.0, timestamp - previous_timestamp)
            )
        turn = radar.get("_route_turn_observation") or {}
        try:
            observed_turn_distance = float(turn.get("distance_to_turn_m"))
            if not math.isfinite(observed_turn_distance):
                raise ValueError("non-finite turn distance")
            self.prelock_turn_distance_m = observed_turn_distance
            self.prelock_turn_last_timestamp = timestamp
            self.prelock_turn_unobserved_progress_m = 0.0
        except (TypeError, ValueError):
            observed_turn_distance = None
        turn_distance = observed_turn_distance
        if (
            turn_distance is None
            and self.prelock_turn_distance_m is not None
            and self.prelock_turn_last_timestamp is not None
            # A stopped vehicle does not lose its physical relation to the
            # turn merely because a wall-clock timer elapsed.  Keep the last
            # online geometry while odometry says the turn is still ahead;
            # invalidate it after the ego has consumed that distance.
            and self.prelock_turn_unobserved_progress_m
            <= float(self.prelock_turn_distance_m) + 1.0
            and speed <= _number(
                self.config.get("prelock_max_ego_speed_mps"), 0.75
            )
        ):
            turn_distance = max(
                0.0,
                float(self.prelock_turn_distance_m)
                - float(self.prelock_turn_unobserved_progress_m),
            )
        if turn_distance is None:
            turn_distance = float("inf")
        opposing = self._opposing_observation(radar, speed)
        nonflow_hazard = self._nonflow_hazard(features, opposing)
        eligible = bool(
            str(flow_reason or "") in {
                "semantic_flow_prehold", "semantic_flow_disambiguating",
            }
            and speed <= _number(
                self.config.get("prelock_max_ego_speed_mps"), 0.75
            )
            and turn_distance <= _number(
                self.config.get("prelock_turn_max_distance_m"), 7.0
            )
            and not getattr(features, "immediate_hazard", False)
            and opposing["count"] > 0
            and opposing["distance_m"] >= _number(
                self.config.get("initial_gap_min_distance_m"), 27.0
            )
            and opposing["ttc_s"] >= _number(
                self.config.get("initial_gap_min_ttc_s"), 3.0
            )
        )
        self.prelock_gap_frames = self.prelock_gap_frames + 1 if eligible else 0
        self.prelock_gap_observation = opposing
        if (
            eligible
            and self.signal_governing_phase == "green"
            and self.prelock_gap_frames >= int(
                self.config.get("initial_gap_confirm_frames", 2)
            )
        ):
            # This is ownership evidence, not a crossing permit.  The v29
            # route3936 run proved that directly jumping from a one-actor
            # 3 s observation bypasses the two-edge gap planner.  Reuse the
            # bounded stop-line approach and let WAIT_FOR_GAP make the only
            # irreversible crossing decision.
            self.prelock_approach_handoff = True
            self.prelock_turn_distance_m = float(turn_distance)
            self.prelock_turn_unobserved_progress_m = 0.0
            self.prelock_approach_progress_m = 0.0
            self.state = self.APPROACH_JUNCTION
        flow_pass_handoff = bool(
            radar.get("_classifier_flow_pass_seen", False)
            and str(flow_reason or "") in {
                "semantic_flow_prehold", "semantic_flow_disambiguating",
                "semantic_turning_flow_after_pass",
                "semantic_specific_lock_persisted",
            }
            and speed <= _number(
                self.config.get("prelock_max_ego_speed_mps"), 0.75
            )
            # A measured pass is stronger ownership evidence than the early
            # broad flow state.  The shared sharp-turn hold can stop just
            # outside the old 7 m boundary (7.316 m in route4183); acquire
            # anywhere in the already bounded 6--10 m approach band, then
            # *still* drive only to the existing 3 m waiting line.
            and turn_distance <= _number(
                self.config.get("confirmed_red_stop_distance_m"), 10.0
            )
            and self.signal_governing_phase == "green"
            and not nonflow_hazard
            and getattr(features, "front_pedestrian_distance", None) is None
        )
        if flow_pass_handoff:
            self.prelock_flow_pass_handoff = True
        # The shared sharp-turn guard can bring the ego to rest just outside
        # Rule33's physical stop zone.  Once the online flow state and the
        # same-head red-to-green sequence agree that this is a signalized
        # flow junction, hand ownership to Rule33 for the remaining bounded
        # approach.  This is deliberately *not* a crossing authorization:
        # APPROACH_JUNCTION moves only to the existing stop zone and then the
        # normal green/gap state machine starts from rest.
        approach_handoff = bool(
            str(flow_reason or "") in {
                "semantic_flow_prehold", "semantic_flow_disambiguating",
            }
            and speed <= _number(
                self.config.get("prelock_max_ego_speed_mps"), 0.75
            )
            and turn_distance > _number(
                self.config.get("junction_stop_distance_m"), 6.0
            )
            and turn_distance <= _number(
                self.config.get("confirmed_red_stop_distance_m"), 10.0
            )
            and self.signal_governing_phase == "green"
            and not nonflow_hazard
            and getattr(features, "front_pedestrian_distance", None) is None
        )
        if approach_handoff:
            if not self.prelock_approach_handoff:
                self.prelock_turn_distance_m = float(turn_distance)
                self.prelock_turn_unobserved_progress_m = 0.0
                self.prelock_approach_progress_m = 0.0
            self.prelock_approach_handoff = True
        return bool(
            self.prelock_flow_pass_handoff
            or self.prelock_approach_handoff
        )

    def _committed_crossing_near_vehicle_hazard(self, features, speed):
        """Detect a close moving vehicle in the ego crossing corridor."""
        # At rest the independent TTC/occupied-corridor guards own the stop;
        # detector motion is only used once the ego has begun to enter the
        # crossing.  This avoids treating a low-confidence duplicate as a
        # moving conflict during a planned launch.
        if float(speed) <= 0.5:
            return False
        for item in list(getattr(features, "detector_objects", None) or []):
            try:
                if str(item.get("class_name", "")).lower() not in {
                    "car", "truck", "bus", "van",
                }:
                    continue
                box = item.get("box_lidar") or {}
                score = float(item.get("score", 0.0) or 0.0)
                lateral = abs(float(box.get("x", 999.0)))
                longitudinal = float(box.get("y", -999.0))
                closing = float(box.get("vy", 0.0))
                yaw = float(box.get("yaw", 0.0))
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                score >= _number(
                    self.config.get("class33_crossing_hazard_min_score"), 0.20
                )
                and (
                    lateral <= _number(
                        self.config.get(
                            "class33_crossing_hazard_core_lateral_m"
                        ),
                        2.5,
                    )
                    or (
                        lateral <= _number(
                            self.config.get(
                                "class33_crossing_hazard_lateral_m"
                            ),
                            5.0,
                        )
                        and abs(math.sin(yaw)) >= _number(
                            self.config.get(
                                "class33_crossing_hazard_transverse_sine"
                            ),
                            0.50,
                        )
                    )
                )
                and 2.0 <= longitudinal <= _number(
                    self.config.get(
                        "class33_crossing_hazard_longitudinal_max_m"
                    ),
                    12.0,
                )
                and closing <= -2.0
            ):
                return True
        return False

    def apply(self, control, radar, features):
        result = clone_control(control)
        speed = max(0.0, feature_value(features, "ego_speed", 0.0))
        timestamp = _number(radar.get("_timestamp"), 0.0)
        if self.last_timestamp is not None and timestamp > self.last_timestamp:
            dt = min(0.2, timestamp - self.last_timestamp)
            if self.state in (self.COMMITTED_CROSSING, self.ROUTE_FOLLOW):
                self.committed_distance_m += speed * dt
            if self.state == self.ROLLING_PREPARE:
                rolling_progress_m = speed * dt
                self.rolling_distance_m += rolling_progress_m
                if self.rolling_mode == "mild":
                    self.mild_rolling_cumulative_distance_m += (
                        rolling_progress_m
                    )
            elif self.mild_rolling_abort_braking:
                # Include the physical braking tail after a rejected rolling
                # slot.  Otherwise a retry could reset odometry while the car
                # was still coasting and exceed the total safe staging bound.
                self.mild_rolling_cumulative_distance_m += speed * dt
                if speed <= 0.10:
                    self.mild_rolling_abort_braking = False
            if (
                self.state == self.APPROACH_JUNCTION
                and (
                    self.prelock_approach_handoff
                    or self.prelock_flow_pass_handoff
                )
            ):
                self.prelock_approach_progress_m += speed * dt
        self.last_timestamp = timestamp
        opposing = self._opposing_observation(radar, speed)
        waypoint_observation = radar.get("_route_waypoint_observation") or {}
        try:
            far_heading_error = float(
                (waypoint_observation.get("far") or {}).get(
                    "heading_error_rad"
                )
            )
            if not math.isfinite(far_heading_error):
                raise ValueError("non-finite far heading error")
        except (AttributeError, TypeError, ValueError):
            far_heading_error = 0.0
        try:
            near_heading_error = float(
                (waypoint_observation.get("near") or {}).get(
                    "heading_error_rad"
                )
            )
            if not math.isfinite(near_heading_error):
                raise ValueError("non-finite near heading error")
        except (AttributeError, TypeError, ValueError):
            near_heading_error = 0.0
        if (
            not self.mild_rolling_turn_chord_latched
            and self.state == self.COMMITTED_CROSSING
            and self.rolling_mode == "mild"
            and self.committed_distance_m >= _number(
                self.config.get(
                    "mild_rolling_turn_chord_min_progress_m"
                ),
                8.0,
            )
            and near_heading_error * far_heading_error > 0.0
            and abs(near_heading_error) >= _number(
                self.config.get(
                    "mild_rolling_turn_chord_min_near_error_rad"
                ),
                0.40,
            )
            and abs(near_heading_error) - abs(far_heading_error)
            >= _number(
                self.config.get(
                    "mild_rolling_turn_chord_error_margin_rad"
                ),
                0.25,
            )
        ):
            self.mild_rolling_turn_chord_latched = True
        observed_sharp_crossing = bool(
            abs(far_heading_error) >= _number(
                self.config.get("sharp_crossing_heading_error_rad"), 0.70
            )
        )
        if (
            self.state == self.APPROACH_JUNCTION
            and not self.junction_reached_latched
            and observed_sharp_crossing
        ):
            self.sharp_crossing_latched = True
        sharp_crossing = bool(
            observed_sharp_crossing or self.sharp_crossing_latched
        )
        signal_color, signal_confidence, signal_tracking = self._signal_color(
            features, radar
        )
        # The Town12 fog route can keep the governing lamp unreadable even
        # though radar observes the complete signal-controlled opposing flow.
        # A normal inter-vehicle opening in this class is at most about 3 s;
        # only treat five continuous seconds of empty, healthy radar after a
        # previously observed stream as phase-clearance evidence.  This is a
        # sensor-only fallback and is deliberately unavailable before the ego
        # has stopped at the junction.
        if self.state == self.WAIT_FOR_GREEN:
            if opposing["count"] > 0:
                self.signal_flow_seen = True
                self.signal_flow_empty_frames = 0
            elif self.signal_flow_seen:
                self.signal_flow_empty_frames += 1
            if (
                self.signal_flow_seen
                and self.signal_flow_empty_frames >= int(
                    self.config.get("signal_flow_clear_confirm_frames", 50)
                )
                and signal_color != "green"
            ):
                signal_color = "green"
                signal_confidence = 0.85
                signal_tracking = {
                    "source": "class33_opposing_flow_phase_clear",
                    "empty_frames": int(self.signal_flow_empty_frames),
                }
        else:
            self.signal_flow_empty_frames = 0
        turn_observation = radar.get("_route_turn_observation") or {}
        turn_distance = turn_observation.get("distance_to_turn_m")
        try:
            turn_distance = float(turn_distance)
            if not math.isfinite(turn_distance):
                raise ValueError("non-finite turn distance")
            self.prelock_turn_distance_m = turn_distance
            self.prelock_turn_last_timestamp = timestamp
            self.prelock_turn_unobserved_progress_m = 0.0
            # A fresh online distance is a new spatial reference.  Retaining
            # progress accumulated before this observation would subtract the
            # same motion twice when the route-turn detector drops out again.
            self.prelock_approach_progress_m = 0.0
        except (TypeError, ValueError):
            turn_distance = None
        if turn_distance is None and self.prelock_turn_distance_m is not None:
            turn_distance = max(
                0.0,
                float(self.prelock_turn_distance_m)
                - float(self.prelock_approach_progress_m),
            )

        # A sensor flow handoff must wait from one repeatable physical pose.
        # If the pass evidence arrives after the initial 6 m stop, return to
        # APPROACH_JUNCTION and finish the bounded approach to the closer
        # flow-wait line before evaluating any crossing gap.
        flow_handoff_stop_distance = _number(
            self.config.get(
                "sharp_flow_handoff_stop_distance_m"
                if sharp_crossing
                else "flow_handoff_stop_distance_m"
            ),
            6.0 if sharp_crossing else 3.0,
        )
        late_flow_handoff = bool(
            self.prelock_approach_handoff
            and not self.prelock_flow_pass_handoff
            and bool(radar.get("_classifier_flow_pass_seen", False))
            and self.signal_governing_phase == "green"
            and turn_distance is not None
            and float(turn_distance) <= _number(
                self.config.get("junction_stop_distance_m"), 6.0
            )
        )
        if late_flow_handoff:
            self.prelock_flow_pass_handoff = True
            if (
                float(turn_distance) > flow_handoff_stop_distance
                and self.state in (self.WAIT_FOR_GREEN, self.WAIT_FOR_GAP)
            ):
                self.state = self.APPROACH_JUNCTION
                self.stop_frames = 0
                self.green_frames = 0
                self.gap_frames = 0
        # Once Rule33 owns the approach, reach the online geometry's actual
        # stop-line target before observing traffic.  Making this depend on
        # the classifier's asynchronous flow-pass flag caused otherwise
        # identical routes to stop at different poses: v90 route3936 entered
        # WAIT_FOR_GAP at 5.99 m, well before the visible white line, whereas
        # route4183 reached the intended 3--4 m conflict-entry pose.  The
        # class lock and live turn geometry are sufficient for this bounded
        # approach; the flow-pass flag remains mandatory for sharp crossing
        # authorization, not for reaching the planning pose.
        junction_stop_distance = flow_handoff_stop_distance
        try:
            confirmed_red_stop_zone = bool(
                signal_color in {"red", "yellow"}
                and turn_distance is not None
                and float(turn_distance) <= junction_stop_distance
            )
        except (TypeError, ValueError):
            confirmed_red_stop_zone = False
        at_junction = bool(
            self._at_junction(
                radar, features, turn_distance, junction_stop_distance
            )
            or confirmed_red_stop_zone
        )
        if (
            self.state == self.APPROACH_JUNCTION
            and at_junction
            and not self.junction_reached_latched
        ):
            try:
                self.junction_wait_distance_m = float(turn_distance)
            except (TypeError, ValueError):
                self.junction_wait_distance_m = None
            self.junction_reached_latched = True
            # The online turn-distance scalar can jump by about one metre
            # while the stopped vehicle has not moved (v35: 5.007 m, v36:
            # 4.045 m at effectively the same GNSS pose).  Safety therefore
            # follows the control lifecycle that stopped at the configured
            # sharp 6 m line, not that discontinuous scalar.  Subsequent
            # rolling distance is measured by ego odometry from this pose.
            self.sharp_safe_wait_latched = bool(
                sharp_crossing
                and self.prelock_flow_pass_handoff
                and junction_stop_distance >= _number(
                    self.config.get("sharp_flow_handoff_stop_distance_m"), 6.0
                ) - 1.0e-6
            )
        # The sharp stop and the online flow handoff are independent sensor
        # lifecycle events.  In route4183 v37 the ego reached and held the
        # configured safe stop before `_classifier_flow_pass_seen` arrived;
        # evaluating their conjunction only on the first stop frame therefore
        # made the rolling planner permanently unreachable.  Latch the safety
        # event monotonically whenever both facts have become true, regardless
        # of their arrival order.  This remains a preparation permission only:
        # the live two-actor gap checks below still authorize all movement.
        if (
            not self.sharp_safe_wait_latched
            and self.junction_reached_latched
            and sharp_crossing
            and self.prelock_flow_pass_handoff
            and junction_stop_distance >= _number(
                self.config.get("sharp_flow_handoff_stop_distance_m"), 6.0
            ) - 1.0e-6
        ):
            self.sharp_safe_wait_latched = True
        if self.state == self.APPROACH_JUNCTION and self.junction_reached_latched:
            at_junction = True
        immediate_hazard = bool(getattr(features, "immediate_hazard", False))
        nonflow_hazard = self._nonflow_hazard(features, opposing)
        blocking_nonflow_hazard = self._blocking_nonflow_hazard(features)
        mild_timed_slot = False
        sharp_timed_slot = False
        sharp_wait_distance = self.junction_wait_distance_m
        sharp_conflict_eta_s = 0.0
        sharp_lead_ttc_min_s = 0.0
        sharp_lead_ttc_max_s = 0.0
        sharp_follower_min_ttc_s = 0.0
        near_vehicle_hazard = False
        sharp_continuous_window = False
        validated_continuous_window = False

        # A prelock approach handoff acquires control before the classifier's
        # pass flag necessarily exists.  Once the ego has reached the real
        # stop zone, preserve that later online pass evidence so the same
        # measured two-edge slot planner is available; merely approaching the
        # line never sets this crossing-capable flag.
        if self.state == self.APPROACH_JUNCTION:
            if at_junction:
                self._hard_stop(result)
                self.stop_frames = self.stop_frames + 1 if speed <= 0.45 else 0
                if self.stop_frames >= int(
                    self.config.get("junction_stop_confirm_frames", 3)
                ):
                    self.state = self.WAIT_FOR_GREEN
                    self.green_frames = 0
            elif nonflow_hazard:
                self._hard_stop(result)
            else:
                red_distance = getattr(features, "red_stop_distance", None)
                try:
                    red_distance = float(red_distance)
                except (TypeError, ValueError):
                    red_distance = None
                if (
                    signal_color in {"red", "yellow"}
                    and turn_distance is None
                    and red_distance is not None
                    and red_distance <= 13.0
                ):
                    result.throttle = 0.0
                    result.brake = max(
                        _number(getattr(result, "brake", 0.0)),
                        min(0.75, max(0.18, (13.0 - red_distance) / 10.0)),
                    )
                elif speed < _number(
                    self.config.get("approach_target_speed_mps"), 4.5
                ):
                    result.throttle = max(
                        _number(getattr(result, "throttle", 0.0)), 0.30
                    )
                    result.brake = 0.0
                elif speed > _number(
                    self.config.get("approach_max_speed_mps"), 5.5
                ):
                    result.throttle = 0.0
                    result.brake = max(
                        _number(getattr(result, "brake", 0.0)), 0.20
                    )

        elif self.state == self.WAIT_FOR_GREEN:
            self._hard_stop(result)
            green = bool(
                signal_color == "green"
                and signal_confidence >= _number(
                    self.config.get("green_min_confidence"), 0.30
                )
            )
            self.green_frames = self.green_frames + 1 if green else 0
            if self.green_frames >= int(self.config.get("green_confirm_frames", 3)):
                self.state = self.WAIT_FOR_GAP
                self.wait_for_gap_started_at = timestamp
                self.pass_seen = False
                self.gap_frames = 0
                self.empty_frames = 0

        elif self.state == self.WAIT_FOR_GAP:
            self._hard_stop(result)
            if self.wait_for_gap_started_at is None:
                self.wait_for_gap_started_at = timestamp
            if signal_color in {"red", "yellow"}:
                self.state = self.WAIT_FOR_GREEN
                self.wait_for_gap_started_at = None
                self.green_frames = 0
            else:
                if (
                    opposing["count"] > 0
                    and opposing["distance_m"] <= _number(
                        self.config.get("opposing_pass_near_m"), 4.5
                    )
                ):
                    self.pass_seen = True
                if bool(radar.get("_classifier_flow_pass_seen", False)):
                    self.pass_seen = True
                self.empty_frames = (
                    self.empty_frames + 1 if opposing["count"] == 0 else 0
                )
                radar_point_count = len(self._packet_points(
                    radar.get("_online_radar_packet")
                ))
                verified_empty_tail_sample = bool(
                    not sharp_crossing
                    and self.pass_seen
                    and opposing["count"] == 0
                    and radar_point_count >= int(_number(
                        self.config.get(
                            "mild_verified_empty_tail_min_radar_points"
                        ),
                        5,
                    ))
                )
                self.verified_empty_tail_frames = (
                    self.verified_empty_tail_frames + 1
                    if verified_empty_tail_sample else 0
                )
                post_pass_gap = bool(
                    not sharp_crossing
                    and self.pass_seen
                    and (
                        self.empty_frames >= int(
                            self.config.get("post_pass_empty_confirm_frames", 3)
                        )
                        or (
                            opposing["count"] > 0
                            and opposing["distance_m"] >= _number(
                                self.config.get("opposing_gap_min_distance_m"), 22.0
                            )
                            and opposing["ttc_s"] >= _number(
                                self.config.get("opposing_gap_min_ttc_s"), 3.20
                            )
                        )
                    )
                )
                no_flow_gap = bool(
                    not self.pass_seen
                    and self.empty_frames >= int(
                        self.config.get("no_flow_empty_confirm_frames", 6)
                    )
                )
                # Historical empty frames are not sufficient evidence for a
                # launch: the opposing stream may reappear after a radar
                # occlusion on the very next tick.  Mild crossings therefore
                # require the *current* observation to be either empty or a
                # fully distant/low-closing cluster before post-pass/no-flow
                # authorization can survive to the commit branch.
                current_gap_safe = bool(
                    opposing["count"] == 0
                    or (
                        opposing["distance_m"] >= _number(
                            self.config.get("opposing_gap_min_distance_m"), 22.0
                        )
                        and opposing["ttc_s"] >= _number(
                            self.config.get("opposing_gap_min_ttc_s"), 3.20
                        )
                    )
                )
                self.current_gap_safe_frames = (
                    self.current_gap_safe_frames + 1
                    if current_gap_safe else 0
                )
                # When the flow interval itself is shorter than a launch from
                # rest, waiting until the leader has passed throws away the
                # usable part of the gap.  For the sensor-handoff branch only,
                # plan a moving slot: launch while the leader is still
                # approaching so the ego reaches the conflict line just
                # behind it, but only when radar also sees a sufficiently
                # distant follower.  This uses both measured time edges and
                # never stages or stops inside the opposing lane.
                mild_timed_slot = bool(
                    bool(self.config.get("timed_slot_enabled", False))
                    and not sharp_crossing
                    and self.pass_seen
                    and speed <= _number(
                        self.config.get("timed_slot_max_ego_speed_mps"), 0.75
                    )
                    and signal_color == "green"
                    and not nonflow_hazard
                    and getattr(features, "front_pedestrian_distance", None) is None
                    and opposing["cluster_count"] >= 2
                    and opposing["ttc_s"] >= _number(
                        self.config.get("timed_slot_lead_ttc_min_s"), 0.20
                    )
                    # On the sharper online geometry, waiting until the lead
                    # actor disappears consumed about 0.6 s of a 2.5 s
                    # interval.  Commit only at the measured trailing edge:
                    # the leader is almost clear, while the follower still
                    # has the full observed gap.  This replaces the disproved
                    # fixed ego-conflict ETA used by v17--v21.
                    and opposing["ttc_s"] <= _number(
                        self.config.get("timed_slot_lead_ttc_max_s"), 0.50
                    )
                    and opposing["headway_s"] >= _number(
                        self.config.get("timed_slot_min_headway_s"), 2.30
                    )
                    and opposing["follower_ttc_s"] >= _number(
                        self.config.get("timed_slot_follower_min_ttc_s"), 3.60
                    )
                )
                try:
                    sharp_wait_distance = float(sharp_wait_distance)
                    sharp_min_wait_distance = _number(
                        self.config.get(
                            "sharp_timed_slot_min_wait_distance_m"
                        ),
                        4.5,
                    )
                    if sharp_wait_distance >= sharp_min_wait_distance:
                        conflict_boundary_m = _number(
                            self.config.get(
                                "sharp_conflict_boundary_distance_m"
                            ),
                            3.0,
                        )
                        reference_progress_m = max(
                            0.1,
                            _number(
                                self.config.get(
                                    "sharp_conflict_reference_wait_distance_m"
                                ),
                                6.0,
                            ) - conflict_boundary_m,
                        )
                        progress_m = max(
                            0.0,
                            sharp_wait_distance - conflict_boundary_m,
                        )
                        sharp_conflict_eta_s = _number(
                            self.config.get(
                                "sharp_conflict_reference_eta_s"
                            ),
                            1.30,
                        ) * math.sqrt(progress_m / reference_progress_m)
                        sharp_lead_ttc_max_s = max(
                            0.0,
                            sharp_conflict_eta_s - _number(
                                self.config.get(
                                    "sharp_leader_tail_margin_s"
                                ),
                                0.25,
                            ),
                        )
                        sharp_lead_ttc_max_s = min(
                            sharp_lead_ttc_max_s,
                            _number(
                                self.config.get(
                                    "sharp_lead_ttc_absolute_max_s"
                                ),
                                1.05,
                            ),
                        )
                        sharp_lead_ttc_min_s = max(
                            _number(
                                self.config.get(
                                    "sharp_lead_ttc_absolute_min_s"
                                ),
                                0.25,
                            ),
                            sharp_lead_ttc_max_s - _number(
                                self.config.get(
                                    "sharp_lead_ttc_window_s"
                                ),
                                0.30,
                            ),
                        )
                        sharp_clear_eta_s = (
                            _number(
                                self.config.get("sharp_clear_eta_base_s"),
                                2.45,
                            )
                            + _number(
                                self.config.get(
                                    "sharp_clear_eta_per_wait_m_s"
                                ),
                                0.10,
                            ) * sharp_wait_distance
                        )
                        sharp_follower_min_ttc_s = (
                            sharp_clear_eta_s
                            + _number(
                                self.config.get(
                                    "sharp_follower_clear_margin_s"
                                ),
                                0.40,
                            )
                        )
                except (TypeError, ValueError):
                    sharp_wait_distance = None
                # A sharp junction launches from the earlier 6 m wait line.
                # Runtime v32 exposed a radar-occlusion trap: a purported
                # 4.77 s follower became a hidden 2.47 s middle actor one
                # frame after commit.  The official class stream's *online*
                # adjacent-vehicle headways remain in a stable 2.6--3.1 s
                # band.  Plan on those real consecutive edges instead of an
                # anomalously large cluster skip.  v34 then showed that a
                # 0.63 s launch from the 5.0 m pose
                # reached the downstream conflict at 2.4 s, too late for the
                # follower.  Keep the proven absolute ceiling of 1.05 s, but
                # use the last 0.20 s of the pose-specific conflict ETA so the
                # launch occurs about 0.3 s earlier without repeating v17's
                # unsafe 1.56 s leader entry.
                sharp_timed_slot = bool(
                    bool(self.config.get("sharp_timed_slot_enabled", False))
                    and not bool(
                        self.config.get("sharp_rolling_stage_enabled", False)
                    )
                    and sharp_crossing
                    and sharp_wait_distance is not None
                    and sharp_follower_min_ttc_s > 0.0
                    and self.pass_seen
                    and speed <= _number(
                        self.config.get("timed_slot_max_ego_speed_mps"), 0.75
                    )
                    and signal_color == "green"
                    and not blocking_nonflow_hazard
                    and getattr(features, "front_pedestrian_distance", None) is None
                    and opposing["cluster_count"] >= 2
                    # TTCs are reconstructed from distance/velocity and the
                    # configured decimal boundary can differ by one binary
                    # ulp (for example 0.85 vs 0.8500000000000001).
                    and opposing["ttc_s"] >= sharp_lead_ttc_min_s - 1.0e-6
                    and opposing["ttc_s"] <= sharp_lead_ttc_max_s + 1.0e-6
                    and opposing["headway_s"] >= _number(
                        self.config.get("sharp_timed_slot_min_headway_s"), 2.60
                    )
                    and opposing["headway_s"] <= _number(
                        self.config.get("sharp_timed_slot_max_headway_s"), 3.10
                    )
                    and opposing["follower_ttc_s"] >= sharp_follower_min_ttc_s
                    and opposing["follower_ttc_s"] <= _number(
                        self.config.get("sharp_timed_slot_follower_max_ttc_s"), 4.20
                    )
                )
                sharp_rolling_prepare_candidate = bool(
                    bool(self.config.get("sharp_rolling_stage_enabled", False))
                    and sharp_crossing
                    and self.sharp_safe_wait_latched
                    and self.pass_seen
                    and not self.rolling_attempted
                    and speed <= _number(
                        self.config.get("timed_slot_max_ego_speed_mps"), 0.75
                    )
                    and signal_color == "green"
                    and not blocking_nonflow_hazard
                    and getattr(features, "front_pedestrian_distance", None) is None
                    and opposing["cluster_count"] >= 2
                    and opposing["ttc_s"] >= _number(
                        self.config.get("rolling_prepare_lead_ttc_min_s"), 1.40
                    )
                    and opposing["ttc_s"] <= _number(
                        self.config.get("rolling_prepare_lead_ttc_max_s"), 1.80
                    )
                    and opposing["headway_s"] >= _number(
                        self.config.get("sharp_timed_slot_min_headway_s"), 2.60
                    )
                    and opposing["headway_s"] <= _number(
                        self.config.get("sharp_timed_slot_max_headway_s"), 3.10
                    )
                    and opposing["follower_ttc_s"] >= _number(
                        self.config.get("rolling_prepare_follower_min_ttc_s"), 4.0
                    )
                    and opposing["follower_ttc_s"] <= _number(
                        self.config.get("rolling_prepare_follower_max_ttc_s"), 4.8
                    )
                )
                # The mild variants 4669/4683 expose a different root case:
                # the real ActorFlow interval is only about 2.0--2.7 s, so a
                # launch from rest can never clear it, while apparent 4--5 s
                # gaps repeatedly collapse on the next frame when an occluded
                # middle actor becomes visible.  After a long safe wait, track
                # the *ordinary* adjacent pair over several consecutive live
                # samples and build bounded momentum behind its leader.  This
                # is deliberately separate from both the disproved one-frame
                # mild timed-slot and the sharp-junction rolling calibration.
                mild_wait_elapsed_s = max(
                    0.0,
                    timestamp - _number(
                        self.wait_for_gap_started_at, timestamp
                    ),
                )
                mild_pair_sample = bool(
                    bool(self.config.get("mild_rolling_stage_enabled", False))
                    and not sharp_crossing
                    and self.mild_rolling_cumulative_distance_m <= _number(
                        self.config.get(
                            "mild_rolling_retry_max_start_progress_m"
                        ),
                        0.75,
                    )
                    and mild_wait_elapsed_s >= _number(
                        self.config.get("mild_rolling_min_wait_s"), 10.0
                    )
                    and speed <= _number(
                        self.config.get("timed_slot_max_ego_speed_mps"), 0.75
                    )
                    and signal_color == "green"
                    and not nonflow_hazard
                    and getattr(features, "front_pedestrian_distance", None) is None
                    and opposing["cluster_count"] >= 2
                    and opposing["ttc_s"] >= _number(
                        self.config.get("mild_rolling_prepare_lead_ttc_min_s"),
                        1.60,
                    )
                    and opposing["ttc_s"] <= _number(
                        self.config.get("mild_rolling_prepare_lead_ttc_max_s"),
                        2.40,
                    )
                    and opposing["headway_s"] >= _number(
                        self.config.get("mild_rolling_prepare_min_headway_s"),
                        1.80,
                    )
                    and opposing["headway_s"] <= _number(
                        self.config.get("mild_rolling_prepare_max_headway_s"),
                        2.70,
                    )
                    and opposing["follower_ttc_s"] >= _number(
                        self.config.get("mild_rolling_prepare_follower_min_ttc_s"),
                        3.70,
                    )
                    and opposing["follower_ttc_s"] <= _number(
                        self.config.get("mild_rolling_candidate_follower_max_ttc_s"),
                        5.00,
                    )
                )
                previous_mild = self.mild_rolling_candidate_observation or {}
                previous_mild_timestamp = self.mild_rolling_candidate_timestamp
                mild_pair_continuous = False
                if mild_pair_sample:
                    if previous_mild_timestamp is None:
                        mild_pair_continuous = True
                    else:
                        mild_dt = max(0.0, timestamp - previous_mild_timestamp)
                        mild_pair_continuous = bool(
                            mild_dt <= _number(
                                self.config.get(
                                    "rolling_commit_observation_gap_s"
                                ),
                                0.35,
                            )
                            and abs(
                                opposing["ttc_s"]
                                - max(
                                    0.0,
                                    _number(previous_mild.get("ttc_s"), 0.0)
                                    - mild_dt,
                                )
                            ) <= _number(
                                self.config.get(
                                    "mild_rolling_pair_max_lead_residual_s"
                                ),
                                0.35,
                            )
                            and abs(
                                opposing["follower_ttc_s"]
                                - max(
                                    0.0,
                                    _number(
                                        previous_mild.get("follower_ttc_s"),
                                        0.0,
                                    ) - mild_dt,
                                )
                            ) <= _number(
                                self.config.get(
                                    "mild_rolling_pair_max_follower_residual_s"
                                ),
                                0.35,
                            )
                            and abs(
                                opposing["headway_s"]
                                - _number(previous_mild.get("headway_s"), 0.0)
                            ) <= _number(
                                self.config.get(
                                    "mild_rolling_pair_max_headway_residual_s"
                                ),
                                0.30,
                            )
                        )
                    self.mild_rolling_candidate_frames = (
                        self.mild_rolling_candidate_frames + 1
                        if mild_pair_continuous else 1
                    )
                    self.mild_rolling_candidate_observation = dict(opposing)
                    self.mild_rolling_candidate_timestamp = timestamp
                else:
                    self.mild_rolling_candidate_frames = 0
                    self.mild_rolling_candidate_observation = None
                    self.mild_rolling_candidate_timestamp = None
                mild_rolling_prepare_candidate = bool(
                    mild_pair_sample
                    and mild_pair_continuous
                    and self.mild_rolling_candidate_frames >= int(
                        self.config.get(
                            "mild_rolling_prepare_confirm_frames", 4
                        )
                    )
                )
                if sharp_rolling_prepare_candidate:
                    self.rolling_prepare_frames += 1
                elif mild_pair_sample:
                    self.rolling_prepare_frames = int(
                        self.mild_rolling_candidate_frames
                    )
                else:
                    self.rolling_prepare_frames = 0
                rolling_prepare_candidate = bool(
                    (
                        sharp_rolling_prepare_candidate
                        and self.rolling_prepare_frames >= int(
                            self.config.get("rolling_prepare_confirm_frames", 2)
                        )
                    )
                    or mild_rolling_prepare_candidate
                )
                if rolling_prepare_candidate:
                    # A 2.6--3.1 s ActorFlow interval is shorter than the
                    # measured 2.45 s launch-to-impact plus rear-body
                    # clearance from rest.  Build bounded momentum while the
                    # leader still owns the conflict point, but stay at least
                    # four metres from the old 3 m intrusive stop pose.  The
                    # next state must either commit behind that same leader or
                    # stop before the conflict band; it never parks in-flow.
                    self.state = self.ROLLING_PREPARE
                    self.rolling_mode = (
                        "sharp" if sharp_rolling_prepare_candidate else "mild"
                    )
                    self.rolling_attempted = True
                    if self.rolling_mode == "mild":
                        self.mild_rolling_attempts += 1
                        self.mild_rolling_abort_braking = False
                    self.rolling_started_at = timestamp
                    self.rolling_distance_m = 0.0
                    self.rolling_commit_frames = 0
                    self.rolling_commit_last_timestamp = None
                    self.rolling_staged = False
                    self.rolling_leader_tail_seen = False
                    self.rolling_post_pass_authorized = False
                    self.rolling_pair_observation = dict(opposing)
                    self.rolling_pair_last_timestamp = timestamp
                    self.rolling_pair_last_dual_timestamp = timestamp
                    self.rolling_pair_predicted = False
                    self.rolling_pair_leader_supported = False
                    self.rolling_pair_track_frames = int(
                        self.rolling_prepare_frames
                    )
                    self.rolling_pair_acquired = False
                    self.rolling_pair_continuous = True
                    self.rolling_pair_residuals = {
                        "lead_ttc_s": 0.0,
                        "follower_ttc_s": 0.0,
                        "headway_s": 0.0,
                    }
                    self.rolling_turn_arc_started = False
                    self.rolling_turn_arc_start_distance_m = None
                    result.throttle = _number(
                        self.config.get(
                            "mild_rolling_prepare_throttle"
                            if self.rolling_mode == "mild"
                            else "rolling_prepare_throttle"
                        ),
                        0.55,
                    )
                    result.brake = 0.0
                timed_slot = bool(mild_timed_slot or sharp_timed_slot)
                self.timed_slot_frames = (
                    self.timed_slot_frames + 1 if timed_slot else 0
                )
                self.timed_slot_observation = opposing if timed_slot else None
                self.timed_slot_mode = (
                    "sharp_6m" if sharp_timed_slot
                    else "mild_3m" if mild_timed_slot
                    else "none"
                )
                timed_slot_ready = self.timed_slot_frames >= int(
                    self.config.get("timed_slot_confirm_frames", 2)
                )
                # The v30 route4183 run collided only 0.30 s after launch and
                # 0.039 m from rest: its 3 m wait pose already placed the
                # vehicle nose in the transverse conflict band.  A sharp
                # crossing therefore waits at the earlier 6 m line.  It may
                # use only the longer, separately calibrated sharp slot above
                # or six radar-empty frames over the full 60 m opposing range.
                # Milder geometry keeps the route3936-validated planner.
                sharp_clear_gap = bool(
                    sharp_crossing
                    and self.empty_frames >= int(
                        self.config.get("no_flow_empty_confirm_frames", 6)
                    )
                )
                mild_history_gap = bool(
                    self.config.get("mild_history_gap_enabled", True)
                    and not sharp_crossing
                    and (post_pass_gap or no_flow_gap)
                    and current_gap_safe
                )
                mild_verified_empty_tail_ready = bool(
                    self.config.get(
                        "mild_verified_empty_tail_enabled", False
                    )
                    and verified_empty_tail_sample
                    and self.verified_empty_tail_frames >= int(
                        self.config.get(
                            "mild_verified_empty_tail_confirm_frames", 50
                        )
                    )
                    and signal_color == "green"
                    and not nonflow_hazard
                    and getattr(
                        features, "front_pedestrian_distance", None
                    ) is None
                )
                gap_candidate = (
                    sharp_clear_gap
                    if sharp_crossing
                    else (
                        mild_history_gap
                        or mild_verified_empty_tail_ready
                        or timed_slot_ready
                    )
                )
                # A timed slot can become stale between the observation that
                # arms it and the frame that would launch.  Re-check the
                # near conflict edge on the actual commit frame; otherwise a
                # lead actor that was 10 m away one frame earlier can already
                # be inside the ego nose clearance on this frame.  This is a
                # category-level safety invariant, not a route-specific rule.
                launch_hazard = bool(
                    opposing["count"] > 0
                    and opposing["distance_m"] < _number(
                        self.config.get("crossing_launch_min_distance_m"), 8.0
                    )
                    and opposing["ttc_s"] < _number(
                        self.config.get("crossing_launch_min_ttc_s"), 1.5
                    )
                )
                if launch_hazard:
                    self.gap_frames = 0
                    gap_candidate = False
                mild_history_gap_ready = bool(
                    mild_history_gap
                    and self.current_gap_safe_frames >= int(
                        self.config.get("current_gap_confirm_frames", 1)
                    )
                )
                self.gap_frames = self.gap_frames + 1 if gap_candidate else 0
                if (
                    sharp_clear_gap
                    or timed_slot_ready
                    or mild_history_gap_ready
                    or mild_verified_empty_tail_ready
                ):
                    self.state = self.COMMITTED_CROSSING
                    self.committed_distance_m = 0.0
                    self.sharp_commit_leader_tail_near = False
                    self.sharp_commit_next_actor_seen = False
                    self.sharp_commit_pass_frames = 0
                    self.sharp_commit_leader_passed = False
                    self.sharp_arc_started_distance_m = None
                    result.throttle = max(
                        _number(getattr(result, "throttle", 0.0)),
                        _number(
                            self.config.get("crossing_throttle_floor"), 0.75
                        ),
                    )
                    result.brake = 0.0

        elif self.state == self.ROLLING_PREPARE:
            rolling_prefix = "mild_" if self.rolling_mode == "mild" else ""
            def rolling_value(name, default):
                return _number(
                    self.config.get(rolling_prefix + name),
                    _number(self.config.get(name), default),
                )

            stage_elapsed_s = max(
                0.0,
                timestamp - _number(self.rolling_started_at, timestamp),
            )
            rolling_pair_observed_valid = bool(
                opposing["cluster_count"] >= 2
                and opposing["headway_s"] >= rolling_value(
                    "rolling_track_min_headway_s", 2.30
                )
                and opposing["headway_s"] <= rolling_value(
                    "rolling_track_max_headway_s", 3.35
                )
                and opposing["follower_ttc_s"] >= rolling_value(
                    "rolling_commit_follower_min_ttc_s", 3.20
                )
                and opposing["follower_ttc_s"] <= rolling_value(
                    "rolling_prepare_follower_max_ttc_s", 4.8
                )
            )
            # Compare a restored live pair against the previous live sample
            # before replacing the cache.  The leader may hide the follower
            # for a frame, so identity is carried by bounded TTC/headway
            # residuals rather than by demanding two live samples inside a
            # narrow terminal-TTC interval.
            pair_age_s = max(
                0.0,
                timestamp - _number(
                    self.rolling_pair_last_timestamp, timestamp + 999.0
                ),
            )
            cached_pair = self.rolling_pair_observation or {}
            cached_lead_ttc_s = max(
                0.0, _number(cached_pair.get("ttc_s"), 0.0) - pair_age_s
            )
            cached_follower_ttc_s = max(
                0.0,
                _number(cached_pair.get("follower_ttc_s"), 0.0) - pair_age_s,
            )
            rolling_pair_continuous = bool(
                rolling_pair_observed_valid
                and (
                    self.rolling_pair_observation is None
                    or (
                        pair_age_s <= _number(
                            self.config.get(
                                "rolling_commit_observation_gap_s"
                            ),
                            0.35,
                        )
                        and abs(
                            opposing["ttc_s"] - cached_lead_ttc_s
                        ) <= _number(
                            self.config.get(
                                "rolling_pair_max_lead_residual_s"
                            ),
                            0.45,
                        )
                        and abs(
                            opposing["follower_ttc_s"]
                            - cached_follower_ttc_s
                        ) <= _number(
                            self.config.get(
                                "rolling_pair_max_follower_residual_s"
                            ),
                            0.45,
                        )
                        and abs(
                            opposing["headway_s"]
                            - _number(cached_pair.get("headway_s"), 0.0)
                        ) <= _number(
                            self.config.get(
                                "rolling_pair_max_headway_residual_s"
                            ),
                            0.40,
                        )
                    )
                )
            )
            self.rolling_pair_continuous = rolling_pair_continuous
            if rolling_pair_observed_valid:
                self.rolling_pair_residuals = {
                    "lead_ttc_s": abs(
                        opposing["ttc_s"] - cached_lead_ttc_s
                    ) if self.rolling_pair_observation is not None else 0.0,
                    "follower_ttc_s": abs(
                        opposing["follower_ttc_s"]
                        - cached_follower_ttc_s
                    ) if self.rolling_pair_observation is not None else 0.0,
                    "headway_s": abs(
                        opposing["headway_s"]
                        - _number(cached_pair.get("headway_s"), 0.0)
                    ) if self.rolling_pair_observation is not None else 0.0,
                }
                self.rolling_pair_track_frames = (
                    self.rolling_pair_track_frames + 1
                    if rolling_pair_continuous else 1
                )
                self.rolling_pair_observation = dict(opposing)
                self.rolling_pair_last_timestamp = timestamp
                self.rolling_pair_last_dual_timestamp = timestamp
            pair_confirm_frames = int(
                self.config.get(
                    rolling_prefix + "rolling_pair_track_confirm_frames",
                    self.config.get("rolling_pair_track_confirm_frames", 6),
                )
            )
            if self.rolling_pair_track_frames >= pair_confirm_frames:
                self.rolling_pair_acquired = True
            # Radar depth clustering can hide the follower for one or two
            # frames behind the leader.  v38 observed 2 -> 1 -> 2 clusters
            # while the recovered TTCs followed the same continuous pair.
            # Carry that pair only for a tightly bounded time and only when
            # there is no contradictory second cluster.  This is temporal
            # tracking, not permission to commit without live observations.
            leader_supported_prediction = bool(
                not rolling_pair_observed_valid
                and opposing["cluster_count"] >= 1
                and self.rolling_pair_observation is not None
                and timestamp - _number(
                    self.rolling_pair_last_dual_timestamp,
                    timestamp + 999.0,
                ) <= _number(
                    self.config.get(
                        "rolling_pair_leader_supported_hold_s"
                    ),
                    0.75,
                )
                and opposing["ttc_s"] > 0.0
                and abs(opposing["ttc_s"] - cached_lead_ttc_s)
                <= _number(
                    self.config.get("rolling_pair_max_lead_residual_s"),
                    0.45,
                )
            )
            blind_dropout_prediction = bool(
                not rolling_pair_observed_valid
                and opposing["cluster_count"] == 0
                and self.rolling_pair_observation is not None
                and pair_age_s <= _number(
                    self.config.get("rolling_pair_dropout_hold_s"), 0.25
                )
            )
            predicted_lead_min_ttc_s = rolling_value(
                "rolling_pair_cached_lead_min_ttc_s", 0.35
            )
            predicted_follower_min_ttc_s = rolling_value(
                "rolling_commit_follower_min_ttc_s", 3.20
            )
            predicted_headway_min_s = rolling_value(
                "rolling_commit_min_headway_s", 2.50
            )
            if self.rolling_pair_acquired:
                # A confirmed pair is allowed to advance into its tail-pass
                # phase inside the existing short prediction horizon.  This
                # prevents one substituted depth cluster from destroying a
                # trajectory that has already been proved over many frames.
                predicted_lead_min_ttc_s = min(
                    predicted_lead_min_ttc_s,
                    rolling_value("rolling_leader_tail_ttc_s", 0.55),
                )
                predicted_follower_min_ttc_s = rolling_value(
                    "rolling_next_actor_min_ttc_s", 2.20
                )
                predicted_headway_min_s = rolling_value(
                    "rolling_track_min_headway_s", 2.30
                )
            rolling_pair_predicted = bool(
                (leader_supported_prediction or blind_dropout_prediction)
                and cached_lead_ttc_s >= predicted_lead_min_ttc_s
                and cached_follower_ttc_s >= predicted_follower_min_ttc_s
                and _number(cached_pair.get("headway_s"), 0.0)
                >= predicted_headway_min_s
                and _number(cached_pair.get("headway_s"), 0.0) <= rolling_value(
                    "rolling_commit_max_headway_s", 3.15
                )
            )
            self.rolling_pair_predicted = rolling_pair_predicted
            self.rolling_pair_leader_supported = bool(
                rolling_pair_predicted and leader_supported_prediction
            )
            if self.rolling_pair_leader_supported:
                # Advance the hidden follower prediction while retaining the
                # timestamp of the last real dual observation as the hard
                # uncertainty bound.  This lets the live leader prove motion
                # continuity without turning an arbitrarily long occlusion
                # into a chain of freshly reset predictions.
                self.rolling_pair_observation = {
                    "count": opposing.get("count", 0),
                    "distance_m": opposing.get("distance_m", 0.0),
                    "ttc_s": opposing["ttc_s"],
                    "closing_speed_mps": opposing.get(
                        "closing_speed_mps", 0.0
                    ),
                    "azimuth_deg": opposing.get("azimuth_deg", 0.0),
                    "cluster_count": 2,
                    "follower_distance_m": 0.0,
                    "follower_ttc_s": cached_follower_ttc_s,
                    "headway_s": max(
                        0.0,
                        cached_follower_ttc_s - opposing["ttc_s"],
                    ),
                }
                self.rolling_pair_last_timestamp = timestamp
            rolling_pair_valid = bool(
                (rolling_pair_observed_valid and rolling_pair_continuous)
                or rolling_pair_predicted
            )
            # A rolling approach is not crossing authorization.  Authorization
            # can come from either (a) the terminal live observation of the
            # same temporally acquired pair, or (b) the later post-pass edge.
            # The former is the v64 route4183 success path: lead/follower TTC
            # 0.85/3.63 s, 2.78 s headway, ego 3.02 m/s.  Removing it in favour
            # of post-pass-only authorization made v86 reproduce the same
            # acquired pair but roll past the proven terminal window and time
            # out.  This is pair identity plus kinematics, never route identity.
            post_pass_hold = bool(
                self.config.get("rolling_post_pass_hold_enabled", False)
            )
            continuous_post_pass = bool(
                self.config.get(
                    "rolling_post_pass_continuous_enabled", False
                )
                and self.rolling_mode in {"sharp", "mild"}
            )
            post_pass_tracking = bool(
                (post_pass_hold or continuous_post_pass)
                and self.rolling_mode in {"sharp", "mild"}
            )
            if post_pass_tracking:
                if (
                    opposing["ttc_s"] > 0.0
                    and opposing["ttc_s"] <= _number(
                        self.config.get(
                            rolling_prefix + "rolling_leader_tail_ttc_s"
                        ),
                        0.55,
                    )
                ):
                    self.rolling_leader_tail_seen = True
                if continuous_post_pass:
                    # This is a sensor event marker, not a stop command.
                    self.rolling_staged = bool(
                        self.rolling_leader_tail_seen
                    )
                elif (
                    self.rolling_distance_m >= _number(
                        self.config.get(
                            rolling_prefix + "rolling_stage_distance_m"
                        ),
                        1.70,
                    )
                    or stage_elapsed_s >= _number(
                        self.config.get(
                            rolling_prefix + "rolling_stage_max_time_s"
                        ),
                        1.60,
                    )
                ):
                    self.rolling_staged = True
                next_actor_min_ttc_s = _number(
                    self.config.get(
                        rolling_prefix + "rolling_next_actor_min_ttc_s"
                    ),
                    2.20,
                )
                next_actor_ttc_s = 0.0
                if opposing["cluster_count"] >= 1:
                    nearest_ttc_s = _number(opposing.get("ttc_s"), 0.0)
                    # Seeing a follower behind a still-live leader is not yet
                    # physical tail passage.  The nearest observed actor must
                    # itself have changed to the following edge of the gap.
                    if nearest_ttc_s > rolling_value(
                        "rolling_leader_tail_ttc_s", 0.55
                    ):
                        next_actor_ttc_s = nearest_ttc_s
                elif self.rolling_pair_acquired:
                    # Empty radar output is not a free gap.  Project the last
                    # confirmed follower through the bounded tracker so a
                    # brief occlusion cannot manufacture clearance.
                    next_actor_ttc_s = cached_follower_ttc_s
                rolling_post_pass_gap = bool(
                    self.rolling_staged
                    and self.rolling_leader_tail_seen
                    and self.rolling_pair_acquired
                    and next_actor_ttc_s >= next_actor_min_ttc_s
                )
                if (
                    rolling_post_pass_gap
                    and self.rolling_mode == "sharp"
                    and not self.rolling_turn_arc_started
                ):
                    # The accepted slot owns one continuous 2-D trajectory:
                    # begin the bounded left arc on the first feasible
                    # post-pass observation, rather than rolling straight and
                    # waiting until COMMITTED_CROSSING to steer.
                    self.rolling_turn_arc_started = True
                    self.rolling_turn_arc_start_distance_m = float(
                        self.rolling_distance_m
                    )
                self.sharp_commit_pass_frames = (
                    self.sharp_commit_pass_frames + 1
                    if rolling_post_pass_gap else 0
                )
            else:
                rolling_post_pass_gap = False
            rolling_commit_candidate = bool(
                rolling_pair_observed_valid
                and rolling_pair_continuous
                and opposing["ttc_s"] >= rolling_value(
                    "rolling_commit_lead_ttc_min_s", 0.45
                )
                and opposing["ttc_s"] <= rolling_value(
                    "rolling_commit_lead_ttc_max_s", 0.90
                )
                and opposing["headway_s"] >= rolling_value(
                    "rolling_commit_min_headway_s", 2.50
                )
                and opposing["headway_s"] <= rolling_value(
                    "rolling_commit_max_headway_s", 3.15
                )
            )
            rolling_prepass_slot = bool(
                self.config.get("rolling_prepass_commit_enabled", False)
                and rolling_commit_candidate
                and self.rolling_pair_acquired
            )
            if rolling_commit_candidate:
                observation_gap_s = max(
                    0.0,
                    timestamp - _number(
                        self.rolling_commit_last_timestamp, timestamp
                    ),
                )
                if (
                    self.rolling_commit_last_timestamp is None
                    or observation_gap_s <= _number(
                        self.config.get("rolling_commit_observation_gap_s"), 0.35
                    )
                ):
                    self.rolling_commit_frames += 1
                else:
                    self.rolling_commit_frames = 1
                self.rolling_commit_last_timestamp = timestamp
            elif (
                not rolling_pair_valid
                or (
                    self.rolling_commit_last_timestamp is not None
                    and timestamp - self.rolling_commit_last_timestamp
                    > _number(
                        self.config.get("rolling_commit_observation_gap_s"), 0.35
                    )
                )
                or rolling_pair_observed_valid
            ):
                self.rolling_commit_frames = 0
                self.rolling_commit_last_timestamp = None
            rolling_abort_nonflow = bool(
                nonflow_hazard
                if self.rolling_mode == "mild"
                else blocking_nonflow_hazard
            )
            rolling_abort = bool(
                signal_color in {"red", "yellow"}
                or rolling_abort_nonflow
                or getattr(features, "front_pedestrian_distance", None) is not None
                or (
                    not rolling_pair_valid
                    and not (
                        post_pass_tracking
                        and self.rolling_leader_tail_seen
                    )
                )
                or (
                    (
                        self.mild_rolling_cumulative_distance_m
                        if self.rolling_mode == "mild"
                        else self.rolling_distance_m
                    ) >= rolling_value("rolling_prepare_max_distance_m", 0.65)
                    and not (
                        post_pass_tracking
                        and self.rolling_leader_tail_seen
                    )
                    and not (
                        continuous_post_pass and rolling_pair_valid
                    )
                )
                or (
                    stage_elapsed_s >= rolling_value(
                        "rolling_prepare_max_time_s", 1.2
                    )
                    and not (
                        post_pass_tracking
                        and self.rolling_leader_tail_seen
                    )
                    and not (
                        continuous_post_pass and rolling_pair_valid
                    )
                )
                or (
                    continuous_post_pass
                    and (
                        stage_elapsed_s >= _number(
                            self.config.get(
                                "rolling_continuous_hard_max_time_s"
                            ),
                            3.0,
                        )
                        or self.rolling_distance_m >= _number(
                            self.config.get(
                                "rolling_continuous_hard_max_distance_m"
                            ),
                            8.0,
                        )
                    )
                )
            )
            rolling_tracked_commit = bool(
                rolling_prepass_slot
                or (
                    post_pass_tracking
                    and rolling_post_pass_gap
                    and self.sharp_commit_pass_frames >= int(
                        self.config.get(
                            rolling_prefix
                            + "rolling_post_pass_confirm_frames",
                            2,
                        )
                    )
                )
                or (
                    not post_pass_tracking
                    and rolling_commit_candidate
                    and self.rolling_pair_track_frames >= int(
                        self.config.get(
                            rolling_prefix + "rolling_pair_track_confirm_frames",
                            self.config.get("rolling_pair_track_confirm_frames", 6),
                        )
                    )
                )
            )
            if rolling_tracked_commit:
                self.state = self.COMMITTED_CROSSING
                self.rolling_post_pass_authorized = bool(
                    continuous_post_pass
                    and (rolling_prepass_slot or rolling_post_pass_gap)
                )
                self.committed_distance_m = 0.0
                self.sharp_commit_leader_tail_near = False
                self.sharp_commit_next_actor_seen = False
                self.sharp_commit_pass_frames = 0
                self.sharp_commit_leader_passed = bool(
                    self.rolling_turn_arc_started
                )
                self.sharp_arc_started_distance_m = (
                    0.0 if self.rolling_turn_arc_started else None
                )
                result.throttle = max(
                    _number(self.config.get("crossing_throttle_floor"), 0.75),
                    _number(getattr(result, "throttle", 0.0)),
                )
                result.brake = 0.0
            elif rolling_abort:
                self.state = self.WAIT_FOR_GAP
                self._hard_stop(result)
                if self.rolling_mode == "mild":
                    self.mild_rolling_abort_braking = True
                self.timed_slot_frames = 0
                self.rolling_commit_frames = 0
                self.rolling_commit_last_timestamp = None
                self.rolling_pair_track_frames = 0
                self.rolling_pair_acquired = False
                self.rolling_turn_arc_started = False
                self.rolling_turn_arc_start_distance_m = None
            else:
                if (
                    post_pass_hold
                    and self.rolling_mode in {"sharp", "mild"}
                    and self.rolling_staged
                ):
                    self._hard_stop(result)
                else:
                    rolling_target_speed = rolling_value(
                        "rolling_prepare_target_speed_mps", 2.2
                    )
                    if speed < rolling_target_speed:
                        result.throttle = rolling_value(
                            "rolling_prepare_throttle", 0.55
                        )
                        result.brake = 0.0
                    else:
                        result.throttle = 0.0
                        result.brake = max(
                            _number(getattr(result, "brake", 0.0)), 0.12
                        )

        elif self.state == self.COMMITTED_CROSSING:
            centered_blocker = self._centered_slow_blocker(features)
            if (
                bool(self.config.get("class33_local_bypass_enabled", True))
                and not self.local_bypass_active
                and not self.local_bypass_recovery_active
                and not self.local_bypass_completed
                and centered_blocker
                and self._local_bypass_clear(features, opposing)
            ):
                # Use the controller's negative-steer convention for the
                # left corridor. This is a bounded local manoeuvre, not a
                # route or scenario lookup.
                self.local_bypass_active = True
                # CARLA/control convention is negative steer for a leftward
                # path.  The previous positive sign cancelled the normal
                # left-turn arc and drove the ego toward right-side layout.
                self.local_bypass_side = -1.0
                self.local_bypass_started_distance_m = float(
                    self.committed_distance_m
                )
            # Complete the local manoeuvre based on geometry/progress before
            # evaluating a transient TTC brake hold.  Otherwise the hazard
            # branch can skip this transition forever and leave the bypass
            # latched while the ego waits for traffic.
            if (
                self.local_bypass_active
                and not centered_blocker
                and self.local_bypass_started_distance_m is not None
                and self.committed_distance_m
                - self.local_bypass_started_distance_m
                >= 2.0
            ):
                self.local_bypass_recovery_active = True
                self.local_bypass_recovery_until_distance_m = (
                    self.committed_distance_m
                    + _number(
                        self.config.get("class33_local_bypass_recovery_distance_m"),
                        8.0,
                    )
                )
                self.local_bypass_active = False
                self.local_bypass_side = 0.0
                self.local_bypass_clear_frames = 0
                self.local_bypass_completed = True
            # Once either geometry has observed the leader tail pass and the
            # following gap remain feasible, preserve momentum through that
            # finite window.  A qualified centred blocker is deliberately
            # excluded so the stop-plan-left-bypass lifecycle can own it.
            validated_continuous_window = bool(
                self.config.get(
                    "rolling_post_pass_continuous_enabled", False
                )
                and self.rolling_post_pass_authorized
                and self.rolling_mode in {"sharp", "mild"}
                and not self.local_bypass_active
                and not self.local_bypass_recovery_active
                and not centered_blocker
            )
            sharp_continuous_window = bool(
                validated_continuous_window
                and self.rolling_mode == "sharp"
            )
            if not validated_continuous_window:
                # Detector geometry remains the final collision guard for
                # mild turns and explicit obstacle bypass execution.
                near_vehicle_hazard = bool(
                    self._committed_crossing_near_vehicle_hazard(
                        features, speed
                    )
                )
            # A detector/TTC hazard can disappear for one frame exactly as a
            # transverse actor crosses behind the model's box. Do not let
            # the local path re-accelerate on that single missing frame.
            if near_vehicle_hazard:
                self.local_bypass_hazard_hold_frames = int(
                    self.config.get("class33_local_bypass_hazard_hold_frames", 5)
                )
            elif (
                (self.local_bypass_active or self.local_bypass_recovery_active)
                and self.local_bypass_hazard_hold_frames > 0
            ):
                self.local_bypass_hazard_hold_frames -= 1
                near_vehicle_hazard = True
            # A crossing authorization is never permission to enter an
            # occupied ego corridor.  In v66 route4183 the online detector
            # exposed an immediate front obstacle at 3.19 m and
            # nonflow_hazard=True, but the committed-crossing branch still
            # applied its throttle floor.  Make the obstacle/non-flow signal
            # a state-independent hard stop; it clears online and therefore
            # does not create a route-specific exception or alter gap logic.
            if (
                immediate_hazard
                and not validated_continuous_window
                and (
                    nonflow_hazard
                    if not self.sharp_crossing_latched
                    else blocking_nonflow_hazard
                )
                and not self.local_bypass_active
                and not self.local_bypass_recovery_active
            ):
                near_vehicle_hazard = True
            # The live adjacent-pair TTC is an independent online safety
            # signal for both mild and sharp crossings.  Formal replay showed
            # that a sharp crossing could also reach COMMITTED_CROSSING with
            # TTC already below one second while detector geometry exposed no
            # usable box.  Hold the ego vehicle until that measured conflict
            # clears; never accelerate through a sub-second crossing conflict.
            if (
                self.rolling_mode == "mild"
                and not validated_continuous_window
                and not self.local_bypass_active
                and not self.local_bypass_recovery_active
                and opposing.get("ttc_s", 0.0) > 0.0
                and opposing.get("ttc_s", 0.0)
                <= _number(
                    self.config.get("mild_committed_min_ttc_hold_s"), 0.80
                )
            ):
                near_vehicle_hazard = True
            # A launch can become invalid immediately after commit when the
            # radar pair association was briefly optimistic. If the ego has
            # made almost no crossing progress and the conflict persists,
            # return to the safe waiting state and obtain a new online gap
            # instead of spending the whole evaluator budget in COMMITTED.
            if not self.local_bypass_active and not self.local_bypass_recovery_active:
                if near_vehicle_hazard:
                    self.committed_conflict_frames += 1
                else:
                    self.committed_conflict_frames = 0
                if (
                    self.committed_conflict_frames >= int(
                        self.config.get("class33_committed_replan_confirm_frames", 8)
                    )
                    and self.committed_distance_m <= _number(
                        self.config.get("class33_committed_replan_max_progress_m"),
                        2.0,
                    )
                    and self.committed_replan_attempts < int(
                        self.config.get(
                            "class33_committed_replan_max_attempts", 3
                        )
                    )
                ):
                    self.committed_replan_attempts += 1
                    self.committed_conflict_frames = 0
                    self.state = self.WAIT_FOR_GAP
                    self.wait_for_gap_started_at = timestamp
                    self.pass_seen = True
                    self.gap_frames = 0
                    self.current_gap_safe_frames = 0
                    self.timed_slot_frames = 0
                    self.rolling_mode = "none"
                    self.rolling_attempted = False
                    self.rolling_post_pass_authorized = False
                    self._hard_stop(result)
                    near_vehicle_hazard = True
            if near_vehicle_hazard:
                # Preserve the crossing state and resume once the live object
                # clears; do not convert this transient hazard into a retry.
                result.throttle = 0.0
                result.brake = 1.0
            else:
                result.brake = min(_number(getattr(result, "brake", 0.0)), 0.10)
            if self.sharp_crossing_latched and not self.sharp_commit_leader_passed:
                if (
                    opposing["cluster_count"] >= 1
                    and opposing["ttc_s"] > 0.0
                    and opposing["ttc_s"] <= _number(
                        self.config.get("sharp_commit_leader_tail_near_ttc_s"),
                        0.50,
                    )
                ):
                    self.sharp_commit_leader_tail_near = True
                if (
                    self.sharp_commit_leader_tail_near
                    and opposing["cluster_count"] >= 1
                    and opposing["ttc_s"] >= _number(
                        self.config.get("sharp_commit_next_actor_min_ttc_s"),
                        1.20,
                    )
                ):
                    self.sharp_commit_next_actor_seen = True
                pass_evidence = bool(
                    self.sharp_commit_leader_tail_near
                    and self.sharp_commit_next_actor_seen
                    and (
                        opposing["cluster_count"] == 0
                        or opposing["ttc_s"] >= _number(
                            self.config.get(
                                "sharp_commit_next_actor_min_ttc_s"
                            ),
                            1.20,
                        )
                    )
                )
                self.sharp_commit_pass_frames = (
                    self.sharp_commit_pass_frames + 1
                    if pass_evidence else 0
                )
                if self.sharp_commit_pass_frames >= int(
                    self.config.get(
                        "sharp_commit_leader_pass_confirm_frames", 2
                    )
                ):
                    if not self.sharp_commit_leader_passed:
                        self.sharp_arc_started_distance_m = float(
                            self.committed_distance_m
                        )
                    self.sharp_commit_leader_passed = True
            target_speed = _number(
                self.config.get(
                    "sharp_crossing_target_speed_mps"
                    if self.sharp_crossing_latched
                    else "crossing_target_speed_mps"
                ),
                10.0 if self.sharp_crossing_latched else 7.5,
            )
            if near_vehicle_hazard:
                pass
            elif self.local_bypass_active:
                # A confirmed gap is an execution commitment, not another
                # WAIT_FOR_GAP state.  Clear the short conflict corridor at a
                # bounded target speed without letting the generic radar TTC
                # hold repeatedly reset the manoeuvre.  A real transverse
                # actor is still able to stop it through the detector-geometry
                # guard above.
                bypass_target_speed = _number(
                    self.config.get("class33_local_bypass_target_speed_mps"),
                    2.5,
                )
                result.throttle = (
                    _number(
                        self.config.get("class33_local_bypass_throttle"), 0.45
                    )
                    if speed < bypass_target_speed else 0.20
                )
                result.brake = 0.0
                result.steer = max(
                    -0.48,
                    min(
                        0.48,
                        _number(getattr(result, "steer", 0.0))
                        + self.local_bypass_side
                        * _number(
                            self.config.get("class33_local_bypass_steer_offset"),
                            0.32,
                        ),
                    ),
                )
                if (
                    not centered_blocker
                    and self.local_bypass_started_distance_m is not None
                    and self.committed_distance_m
                    - self.local_bypass_started_distance_m
                    >= 2.0
                ) or (
                    self.local_bypass_started_distance_m is not None
                    and self.committed_distance_m
                    - self.local_bypass_started_distance_m
                    >= _number(
                        self.config.get("class33_local_bypass_max_progress_m"),
                        7.0,
                    )
                ):
                    self.local_bypass_recovery_active = True
                    self.local_bypass_recovery_until_distance_m = (
                        self.committed_distance_m
                        + _number(
                            self.config.get(
                                "class33_local_bypass_recovery_distance_m"
                            ),
                            8.0,
                        )
                    )
                    self.local_bypass_active = False
                    self.local_bypass_side = 0.0
                    self.local_bypass_clear_frames = 0
                    self.local_bypass_completed = True
            elif self.local_bypass_recovery_active:
                # Keep a cautious post-bypass speed while the vehicle clears
                # the conflict corridor; do not immediately reapply the
                # normal 1.0 throttle floor.
                recovery_speed = _number(
                    self.config.get(
                        "class33_local_bypass_recovery_target_speed_mps"
                    ),
                    4.0,
                )
                if speed < recovery_speed:
                    result.throttle = _number(
                        self.config.get("class33_local_bypass_recovery_throttle"),
                        0.28,
                    )
                    result.brake = 0.0
                else:
                    result.throttle = min(
                        _number(getattr(result, "throttle", 0.0)), 0.25
                    )
                    result.brake = min(
                        _number(getattr(result, "brake", 0.0)), 0.10
                    )
                if (
                    self.local_bypass_recovery_until_distance_m is not None
                    and self.committed_distance_m
                    >= self.local_bypass_recovery_until_distance_m
                ):
                    self.local_bypass_recovery_active = False
                    self.local_bypass_recovery_until_distance_m = None
            elif speed < target_speed:
                result.throttle = max(
                    _number(getattr(result, "throttle", 0.0)),
                    _number(self.config.get("crossing_throttle_floor"), 0.75),
                )
                result.brake = 0.0
            else:
                result.throttle = min(
                    _number(getattr(result, "throttle", 0.0)), 0.35
                )
                result.brake = min(_number(getattr(result, "brake", 0.0)), 0.10)
            if self.committed_distance_m >= _number(
                self.config.get("crossing_commit_distance_m"), 30.0
            ):
                self.state = self.ROUTE_FOLLOW

        debug = {
            "source": "ordinary_class_33",
            "state": "CLASS33_{}".format(self.state),
            "at_junction": bool(at_junction),
            "junction_reached_latched": bool(self.junction_reached_latched),
            "junction_wait_distance_m": self.junction_wait_distance_m,
            "junction_stop_distance_m": float(junction_stop_distance),
            "confirmed_red_stop_zone": bool(confirmed_red_stop_zone),
            "immediate_hazard": bool(immediate_hazard),
            "nonflow_hazard": bool(nonflow_hazard),
            "blocking_nonflow_hazard": bool(blocking_nonflow_hazard),
            "near_vehicle_hazard": bool(near_vehicle_hazard),
            "sharp_continuous_window": bool(sharp_continuous_window),
            "validated_continuous_window": bool(
                validated_continuous_window
            ),
            "centered_slow_blocker": bool(
                self._centered_slow_blocker(features)
                if self.state == self.COMMITTED_CROSSING else False
            ),
            "local_bypass_active": bool(self.local_bypass_active),
            "local_bypass_side": float(self.local_bypass_side),
            "local_bypass_clear_frames": int(self.local_bypass_clear_frames),
            "local_bypass_recovery_active": bool(
                self.local_bypass_recovery_active
            ),
            "local_bypass_completed": bool(self.local_bypass_completed),
            "local_bypass_hazard_hold_frames": int(
                self.local_bypass_hazard_hold_frames
            ),
            "committed_conflict_frames": int(self.committed_conflict_frames),
            "committed_replan_attempts": int(self.committed_replan_attempts),
            "local_bypass_recovery_until_distance_m": (
                self.local_bypass_recovery_until_distance_m
            ),
            "route_turn_observation": turn_observation,
            "effective_turn_distance_m": turn_distance,
            "prelock_turn_distance_m": self.prelock_turn_distance_m,
            "prelock_turn_unobserved_progress_m": float(
                self.prelock_turn_unobserved_progress_m
            ),
            "prelock_approach_progress_m": float(
                self.prelock_approach_progress_m
            ),
            "stop_confirm_frames": int(self.stop_frames),
            "signal_color": signal_color,
            "signal_confidence": float(signal_confidence),
            "signal_tracking": signal_tracking,
            "signal_flow_seen": bool(self.signal_flow_seen),
            "signal_flow_empty_frames": int(self.signal_flow_empty_frames),
            "green_confirm_frames": int(self.green_frames),
            "pass_seen": bool(self.pass_seen),
            "gap_confirm_frames": int(self.gap_frames),
            "empty_confirm_frames": int(self.empty_frames),
            "verified_empty_tail_frames": int(
                self.verified_empty_tail_frames
            ),
            "timed_slot_confirm_frames": int(self.timed_slot_frames),
            "timed_slot_observation": self.timed_slot_observation,
            "timed_slot_mode": self.timed_slot_mode,
            "rolling_prepare_frames": int(self.rolling_prepare_frames),
            "rolling_mode": self.rolling_mode,
            "mild_rolling_attempts": int(self.mild_rolling_attempts),
            "mild_rolling_cumulative_distance_m": float(
                self.mild_rolling_cumulative_distance_m
            ),
            "mild_rolling_abort_braking": bool(
                self.mild_rolling_abort_braking
            ),
            "mild_rolling_turn_chord_latched": bool(
                self.mild_rolling_turn_chord_latched
            ),
            "wait_for_gap_elapsed_s": (
                None if self.wait_for_gap_started_at is None else max(
                    0.0, timestamp - self.wait_for_gap_started_at
                )
            ),
            "mild_rolling_candidate_frames": int(
                self.mild_rolling_candidate_frames
            ),
            "rolling_commit_frames": int(self.rolling_commit_frames),
            "rolling_distance_m": float(self.rolling_distance_m),
            "rolling_attempted": bool(self.rolling_attempted),
            "rolling_pair_predicted": bool(self.rolling_pair_predicted),
            "rolling_pair_leader_supported": bool(
                self.rolling_pair_leader_supported
            ),
            "rolling_pair_observation": self.rolling_pair_observation,
            "rolling_staged": bool(self.rolling_staged),
            "rolling_leader_tail_seen": bool(
                self.rolling_leader_tail_seen
            ),
            "rolling_post_pass_authorized": bool(
                self.rolling_post_pass_authorized
            ),
            "rolling_turn_arc_started": bool(
                self.rolling_turn_arc_started
            ),
            # Compatibility aliases keep old experiment parsers readable.
            "sharp_rolling_staged": bool(self.rolling_staged),
            "sharp_rolling_leader_tail_seen": bool(
                self.rolling_leader_tail_seen
            ),
            "rolling_pair_track_frames": int(
                self.rolling_pair_track_frames
            ),
            "rolling_pair_acquired": bool(self.rolling_pair_acquired),
            "rolling_pair_continuous": bool(
                self.rolling_pair_continuous
            ),
            "rolling_pair_residuals": self.rolling_pair_residuals,
            "sharp_commit_leader_tail_near": bool(
                self.sharp_commit_leader_tail_near
            ),
            "sharp_commit_next_actor_seen": bool(
                self.sharp_commit_next_actor_seen
            ),
            "sharp_commit_pass_frames": int(self.sharp_commit_pass_frames),
            "sharp_commit_leader_passed": bool(
                self.sharp_commit_leader_passed
            ),
            "sharp_arc_started_distance_m": self.sharp_arc_started_distance_m,
            "sharp_arc_progress_m": (
                None if self.sharp_arc_started_distance_m is None else max(
                    0.0,
                    self.committed_distance_m
                    - self.sharp_arc_started_distance_m,
                )
            ),
            "sharp_slot_plan": {
                "wait_distance_m": sharp_wait_distance,
                "conflict_eta_s": float(sharp_conflict_eta_s),
                "lead_ttc_min_s": float(sharp_lead_ttc_min_s),
                "lead_ttc_max_s": float(sharp_lead_ttc_max_s),
                "follower_ttc_min_s": float(sharp_follower_min_ttc_s),
            },
            "far_heading_error_rad": float(far_heading_error),
            "sharp_crossing_observed": bool(observed_sharp_crossing),
            "sharp_crossing_latched": bool(self.sharp_crossing_latched),
            "sharp_safe_wait_latched": bool(self.sharp_safe_wait_latched),
            "sharp_crossing": bool(sharp_crossing),
            "timed_slot_ego_conflict_eta_s": _number(
                self.config.get("timed_slot_ego_conflict_eta_s"), 1.30
            ),
            "committed_distance_m": float(self.committed_distance_m),
            "opposing": opposing,
            "prelock_crossing_armed": bool(self.prelock_crossing_armed),
            "prelock_gap_frames": int(self.prelock_gap_frames),
            "prelock_gap_observation": self.prelock_gap_observation,
            "prelock_flow_pass_handoff": bool(
                self.prelock_flow_pass_handoff
            ),
            "prelock_approach_handoff": bool(
                self.prelock_approach_handoff
            ),
            "input_control": {
                "throttle": _number(getattr(control, "throttle", 0.0)),
                "steer": _number(getattr(control, "steer", 0.0)),
                "brake": _number(getattr(control, "brake", 0.0)),
            },
            "output_control": {
                "throttle": _number(getattr(result, "throttle", 0.0)),
                "steer": _number(getattr(result, "steer", 0.0)),
                "brake": _number(getattr(result, "brake", 0.0)),
            },
        }
        if self.state in (
            self.ROLLING_PREPARE, self.COMMITTED_CROSSING, self.ROUTE_FOLLOW
        ):
            debug["ordinary_route_waypoint"] = self._waypoint_directive()
        dense_route_owned = bool(
            self.config.get("dense_route_enabled", False)
            and (
                self.state == self.ROUTE_FOLLOW
                or (
                    self.state == self.COMMITTED_CROSSING
                    and not (
                        self.rolling_mode == "sharp"
                        and self.rolling_post_pass_authorized
                        and self.committed_distance_m < _number(
                            self.config.get("sharp_dense_handoff_distance_m"),
                            16.0,
                        )
                    )
                )
                or (
                    self.state == self.ROLLING_PREPARE
                    and self.rolling_mode == "sharp"
                    and self.rolling_turn_arc_started
                )
            )
        )
        if dense_route_owned:
            debug["ordinary_dense_route_pid"] = self._dense_route_directive()
        return result, debug


def build_rule():
    return Rule33(get_config(33))
