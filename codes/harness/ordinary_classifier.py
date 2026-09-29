"""Offline-trained, temporally smoothed 44-class radar classifier."""

from __future__ import absolute_import

import json
import math
import os
from collections import deque

import numpy as np

from .radar_features import RadarFeatureExtractor, vector_feature_names


UNKNOWN_CLASS = 0


def _finite(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


class OrdinarySceneClassifier44(object):
    """Classify only from a temporal radar observation window.

    The model file contains centroids learned offline from recorded radar
    vectors.  It contains no route IDs, filenames, XML labels or test-order
    lookup.  With no trained model, the only legal output is UNKNOWN/fallback.
    """

    def __init__(self, model_path=None, confidence_threshold=None, min_history=4):
        self.model_path = model_path or os.environ.get("CVCI_ORDINARY_CLASSIFIER_MODEL", "")
        if confidence_threshold is None:
            confidence_threshold = os.environ.get(
                "CVCI_ORDINARY_CLASSIFIER_CONFIDENCE", "0.30"
            )
        self.confidence_threshold = min(max(_finite(confidence_threshold, 0.60), 0.0), 1.0)
        self.min_history = int(min_history)
        self.extractor = RadarFeatureExtractor()
        self.model = self._load(self.model_path)
        self.candidate_history = deque(maxlen=12)
        self.locked_class_id = UNKNOWN_CLASS
        self.locked_confidence = 0.0
        self.last_decision = None
        self.last_timestamp = None
        # A centroid can be useful for broad screening, but it is not enough
        # to distinguish ordinary background returns from the first observed
        # scenario event.  These accumulators are reset for every route and
        # contain only online radar-derived evidence; they never receive a
        # route id, filename, XML label, or test-order value.
        self.semantic_max_dynamic = 0.0
        self.semantic_dynamic_history = deque(maxlen=16)
        # Keep enough frames for class-specific signatures whose evidence is
        # distributed across a short event window.  Static-close checks below
        # still use their own four-frame suffix, so this does not lengthen the
        # class-02 persistence gate.
        self.semantic_static_history = deque(maxlen=32)
        # Dedicated longer window; changing semantic_static_history would
        # alter already validated semantic gates.
        self.semantic_fast_approach_history = deque(maxlen=48)
        self.semantic_high_dynamic_seen = False
        self.semantic_high_persistence_count = 0
        self.semantic_low_dynamic_seen = False
        self.semantic_static_seen = False
        # A short close/approach burst is not by itself enough to claim the
        # low-specificity static or low-dynamic classes. This countdown is
        # refreshed only by online radar observations and resets per route.
        self.semantic_approach_defer_remaining = 0
        self.semantic_approach_seen = False
        self.semantic_broad_lock_remaining = 0
        self.semantic_broad_lock_started = False
        self.semantic_broad_lock_completed = False
        # A confirmed specific event is stronger than a later broad event in
        # the same route.  Keep this state route-local so an opposing-flow
        # high-dynamic burst cannot replace a previously identified
        # dynamic-closing class with a generic cut-in class.
        self.semantic_specific_lock_class = UNKNOWN_CLASS
        self.semantic_specific_lock_confidence = 0.0
        self.semantic_closing_max_consecutive = 0
        self.semantic_observation_count = 0
        self.semantic_min_depth_seen = float("inf")
        self.semantic_geometry4_evaluated = False
        self.semantic_geometry4_near_level_std = 0.0
        self.semantic_geometry4_left_wide_median = 0.0
        self.semantic_geometry4_matched = False
        self.semantic_opposing_flow_max_qualifying = 0
        self.semantic_merge_flow_max_qualifying = 0
        self.semantic_flow_prehold_started_at = None
        self.semantic_flow_pending_class = UNKNOWN_CLASS
        self.semantic_flow_pass_seen = False
        self.semantic_flow_post_frames = 0
        self.semantic_flow_max_slow_near = 0.0
        self.semantic_flow_episode_active = False
        self.semantic_flow_completed_episodes = 0
        self.semantic_flow_clear_frames = 0
        self.semantic_flow_dense_clear_frames = 0
        self.semantic_flow_approach_speeds = deque(maxlen=40)

    @staticmethod
    def _load(path):
        if not path or not os.path.isfile(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                model = json.load(handle)
            if int(model.get("version", 0)) != 1:
                return None
            centroids = model.get("centroids") or {}
            if not centroids:
                return None
            return {
                "mean": np.asarray(model.get("mean") or [], dtype=np.float64),
                "scale": np.asarray(model.get("scale") or [], dtype=np.float64),
                "centroids": {
                    int(key): np.asarray(value, dtype=np.float64)
                    for key, value in centroids.items()
                },
                "feature_names": list(model.get("feature_names") or []),
                "counts": dict(model.get("counts") or {}),
                "semantic_signatures": dict(model.get("semantic_signatures") or {}),
            }
        except Exception:
            return None

    def reset(self):
        self.extractor.reset()
        self.candidate_history.clear()
        self.locked_class_id = UNKNOWN_CLASS
        self.locked_confidence = 0.0
        self.last_decision = None
        self.last_timestamp = None
        self.semantic_max_dynamic = 0.0
        self.semantic_dynamic_history.clear()
        self.semantic_static_history.clear()
        self.semantic_fast_approach_history.clear()
        self.semantic_high_dynamic_seen = False
        self.semantic_high_persistence_count = 0
        self.semantic_low_dynamic_seen = False
        self.semantic_static_seen = False
        self.semantic_approach_defer_remaining = 0
        self.semantic_approach_seen = False
        self.semantic_broad_lock_remaining = 0
        self.semantic_broad_lock_started = False
        self.semantic_broad_lock_completed = False
        self.semantic_specific_lock_class = UNKNOWN_CLASS
        self.semantic_specific_lock_confidence = 0.0
        self.semantic_closing_max_consecutive = 0
        self.semantic_observation_count = 0
        self.semantic_min_depth_seen = float("inf")
        self.semantic_geometry4_evaluated = False
        self.semantic_geometry4_near_level_std = 0.0
        self.semantic_geometry4_left_wide_median = 0.0
        self.semantic_geometry4_matched = False
        self.semantic_opposing_flow_max_qualifying = 0
        self.semantic_merge_flow_max_qualifying = 0
        self.semantic_flow_prehold_started_at = None
        self.semantic_flow_pending_class = UNKNOWN_CLASS
        self.semantic_flow_pass_seen = False
        self.semantic_flow_post_frames = 0
        self.semantic_flow_max_slow_near = 0.0
        self.semantic_flow_episode_active = False
        self.semantic_flow_completed_episodes = 0
        self.semantic_flow_clear_frames = 0
        self.semantic_flow_dense_clear_frames = 0
        self.semantic_flow_approach_speeds.clear()

    def _resolve_pending_flow(
        self, latest_flow_fast_close, latest_flow_slow_near,
        latest_ego_speed, turning_flow_start, guarded,
    ):
        """Resolve only from complete actor events, never one ambiguous frame."""
        fast_close = latest_flow_fast_close
        slow_near = latest_flow_slow_near
        turn_profile = turning_flow_start[0][0] if turning_flow_start else {}
        speed_samples = list(self.semantic_flow_approach_speeds)
        min_speed_samples = max(3, int(_finite(
            turn_profile.get("flow_speed_min_samples"), 6
        )))
        if len(speed_samples) >= min_speed_samples:
            speed_q90 = float(np.quantile(speed_samples, 0.90))
            if speed_q90 <= _finite(
                turn_profile.get("max_low_speed_turning_flow_mps"), 8.4
            ):
                class_id = turning_flow_start[0][1]
                confidence = turning_flow_start[0][2]
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                self.semantic_flow_pending_class = UNKNOWN_CLASS
                return (
                    class_id, confidence,
                    "semantic_turning_flow_low_speed", guarded,
                )
            straight_low = _finite(
                turn_profile.get("min_unique_straight_flow_mps"), 8.55
            )
            straight_high = _finite(
                turn_profile.get("max_unique_straight_flow_mps"), 9.45
            )
            straight_very_high = _finite(
                turn_profile.get("min_unique_high_straight_flow_mps"), 10.5
            )
            if straight_low <= speed_q90 <= straight_high or speed_q90 >= straight_very_high:
                resolved = self.semantic_flow_pending_class
                self.semantic_specific_lock_class = resolved
                self.semantic_specific_lock_confidence = 0.96
                self.semantic_flow_pending_class = UNKNOWN_CLASS
                return (
                    resolved, 0.96,
                    "semantic_straight_flow_speed", guarded,
                )
        if fast_close >= 1.0:
            self.semantic_flow_pass_seen = True
            self.semantic_flow_post_frames += 1
            self.semantic_flow_episode_active = True
            self.semantic_flow_clear_frames = 0
            self.semantic_flow_dense_clear_frames = 0
            self.semantic_flow_max_slow_near = max(
                self.semantic_flow_max_slow_near, slow_near
            )
            for profile, class_id, confidence in turning_flow_start:
                if slow_near >= _finite(
                    profile.get("min_turning_slow_near_count"), 20.0
                ):
                    self.semantic_specific_lock_class = class_id
                    self.semantic_specific_lock_confidence = confidence
                    self.semantic_flow_pending_class = UNKNOWN_CLASS
                    return (
                        class_id, confidence,
                        "semantic_turning_flow_after_pass", guarded,
                    )
            return UNKNOWN_CLASS, 0.0, "semantic_flow_disambiguating", guarded

        if not self.semantic_flow_pass_seen:
            return UNKNOWN_CLASS, 0.0, "semantic_flow_disambiguating", guarded

        self.semantic_flow_post_frames += 1
        self.semantic_flow_clear_frames += 1
        if self.semantic_flow_episode_active:
            self.semantic_flow_episode_active = False
            self.semantic_flow_completed_episodes += 1
            self.semantic_flow_clear_frames = 1

        dense_clear_min = _finite(
            turn_profile.get("min_turning_clear_slow_near_count"), 30.0
        )
        max_dense_speed = _finite(
            turn_profile.get("max_turning_clear_ego_speed"), 0.75
        )
        if slow_near >= dense_clear_min and latest_ego_speed <= max_dense_speed:
            self.semantic_flow_dense_clear_frames += 1
        else:
            self.semantic_flow_dense_clear_frames = 0
        dense_confirm = max(1, int(_finite(
            turn_profile.get("turning_clear_confirm_frames"), 2
        )))
        if self.semantic_flow_dense_clear_frames >= dense_confirm:
            class_id = turning_flow_start[0][1]
            confidence = turning_flow_start[0][2]
            self.semantic_specific_lock_class = class_id
            self.semantic_specific_lock_confidence = confidence
            self.semantic_flow_pending_class = UNKNOWN_CLASS
            return (
                class_id, confidence,
                "semantic_turning_flow_after_clear", guarded,
            )

        # Event count alone is not a category: both official implementations
        # create a continuous ActorFlow from the same source direction.  If
        # speed overlaps, remain safely UNKNOWN until a future observed-path
        # curvature discriminator exists; never guess class 33 on timeout.
        return UNKNOWN_CLASS, 0.0, "semantic_flow_disambiguating", guarded

    def _semantic_candidate(self, features):
        """Apply optional, learned event-signature guards.

        The signatures are generated offline from labelled radar recordings
        and stored in the model.  Formal-valid recordings are required for
        acceptance; explicitly marked invalid recordings may only supply a
        diagnostic candidate.  They are deliberately narrow:
        a class is not accepted merely because its centroid is nearest.  A
        signature either supplies an observable event gate or leaves the
        centroid prediction in place for classes without a learned guard.
        This makes incremental training safe when only a subset of the 44
        classes has real recordings.
        """
        signatures = (self.model or {}).get("semantic_signatures") or {}
        if not signatures:
            return None

        latest_dynamic = _finite(features.get("front_dynamic_count", 0.0))
        latest_close = _finite(features.get("front_close_count", 0.0))
        latest_left = _finite(features.get("left_nearest", 0.0))
        latest_right = _finite(features.get("right_nearest", 0.0))
        latest_depth = _finite(features.get("depth_min", 0.0))
        latest_flow_fast_close = _finite(
            features.get("flow_fast_close_count", 0.0)
        )
        latest_flow_slow_near = _finite(
            features.get("flow_slow_near_count", 0.0)
        )
        latest_flow_approach_count = _finite(
            features.get("flow_approach_speed_count", 0.0)
        )
        latest_flow_approach_speed = _finite(
            features.get("flow_approach_speed_median_mps", 0.0)
        )
        if (
            latest_flow_approach_count >= 3.0
            and latest_flow_approach_speed > 0.0
            and abs(_finite(features.get("ego_speed"), 0.0)) <= 0.75
        ):
            self.semantic_flow_approach_speeds.append(
                latest_flow_approach_speed
            )
        self.semantic_observation_count += 1
        if latest_depth > 0.0:
            self.semantic_min_depth_seen = min(
                self.semantic_min_depth_seen, latest_depth
            )
        self.semantic_max_dynamic = max(self.semantic_max_dynamic, latest_dynamic)
        self.semantic_dynamic_history.append(latest_dynamic)
        self.semantic_static_history.append(
            {
                "dynamic": latest_dynamic,
                "closing": _finite(features.get("front_closing_count", 0.0)),
                "close": latest_close,
                "front_min": _finite(features.get("front_min", 0.0)),
                "left": latest_left,
                "right": latest_right,
                "points": _finite(features.get("point_count", 0.0)),
                "front": _finite(features.get("front_count", 0.0)),
                "depth_min": _finite(features.get("depth_min", 0.0)),
                "depth_median": _finite(features.get("depth_median", 0.0)),
                "depth_mean": _finite(features.get("depth_mean", 0.0)),
                "depth_std": _finite(features.get("depth_std", 0.0)),
                "near_level": _finite(features.get("near_level_count", 0.0)),
                "left_wide_mid": _finite(features.get("left_wide_mid_count", 0.0)),
                "ego_speed": abs(_finite(features.get("ego_speed", 0.0))),
                "route_heading_error": abs(_finite(
                    features.get("route_heading_error", 1e9), 1e9
                )),
                "velocity_abs_q90": _finite(
                    features.get("velocity_abs_q90", 0.0)
                ),
                "closing_velocity_q90": _finite(
                    features.get("closing_velocity_q90", 0.0)
                ),
                "same_direction_flow_count": _finite(
                    features.get("same_direction_flow_count", 0.0)
                ),
            }
        )
        fast_count = _finite(features.get("fast_approach_count", 0.0))
        fast_depth = _finite(features.get("fast_approach_depth_median", 0.0))
        self.semantic_fast_approach_history.append(
            fast_depth if fast_count >= 3.0 and fast_depth > 0.0 else None
        )

        dynamic_closing = []
        high = []
        low = []
        static = []
        static_background_start = []
        static_background_geometry4 = []
        static_straight_obstacle_start = []
        opposing_flow_start = []
        turning_flow_start = []
        merge_flow_start = []
        guarded = set()
        for key, profile in signatures.items():
            try:
                class_id = int(key)
            except (TypeError, ValueError):
                continue
            profile = profile or {}
            guarded.add(class_id)
            kind = str(profile.get("kind", "")).strip().lower()
            if kind == "dynamic_closing":
                dynamic_closing.append((
                    _finite(profile.get("min_peak_dynamic"), 1.0),
                    _finite(profile.get("max_peak_dynamic"), 1e9),
                    _finite(profile.get("min_peak_closing"), 1.0),
                    _finite(profile.get("min_front_close"), 1.0),
                    _finite(profile.get("max_front_min"), 1e9),
                    max(1, int(_finite(profile.get("min_persistence_frames"), 4))),
                    max(1, int(_finite(profile.get("persistence_window_frames"), 12))),
                    max(1, int(_finite(profile.get("min_consecutive_frames"), 0))),
                    max(0, int(_finite(profile.get("approach_deferral_frames"), 0))),
                    max(0, int(_finite(profile.get("broad_lock_delay_frames"), 0))),
                    class_id,
                    _finite(profile.get("confidence"), 0.90),
                ))
                startup = profile.get("startup_static_straight") or {}
                if startup:
                    static_straight_obstacle_start.append((
                        startup,
                        class_id,
                        _finite(profile.get("confidence"), 0.90),
                    ))
            elif kind == "dynamic_high":
                high.append((
                    _finite(profile.get("min_peak_dynamic"), 1.0),
                    max(1, int(_finite(profile.get("min_persistence_frames"), 3))),
                    max(1, int(_finite(profile.get("persistence_window_frames"), 8))),
                    class_id,
                    _finite(profile.get("confidence"), 0.90),
                ))
            elif kind == "dynamic_low":
                low.append((
                    _finite(profile.get("min_peak_dynamic"), 1.0),
                    _finite(profile.get("max_peak_dynamic"), 1e9),
                    class_id,
                    _finite(profile.get("confidence"), 0.80),
                ))
            elif kind == "static_close":
                static.append((
                    _finite(profile.get("min_current_close"), 1.0),
                    _finite(profile.get("min_side_nearest"), 0.0),
                    max(1, int(_finite(profile.get("min_static_frames"), 3))),
                    class_id,
                    _finite(profile.get("confidence"), 0.80),
                ))
            elif kind == "static_background_start":
                geometry4 = profile.get("startup_geometry4") or {}
                if geometry4:
                    static_background_geometry4.append((
                        geometry4,
                        class_id,
                        _finite(profile.get("confidence"), 0.90),
                    ))
                variants = [profile]
                variants.extend(profile.get("alternates") or [])
                for variant in variants:
                    merged = dict(profile)
                    merged.pop("alternates", None)
                    merged.update(variant or {})
                    static_background_start.append((
                        merged,
                        class_id,
                        _finite(merged.get("confidence"), 0.90),
                    ))
            elif kind == "opposing_flow_start":
                opposing_flow_start.append((
                    profile,
                    class_id,
                    _finite(profile.get("confidence"), 0.94),
                ))
            elif kind == "turning_flow_start":
                turning_flow_start.append((
                    profile,
                    class_id,
                    _finite(profile.get("confidence"), 0.95),
                ))
            elif kind == "merge_flow_start":
                merge_flow_start.append((
                    profile,
                    class_id,
                    _finite(profile.get("confidence"), 0.94),
                ))

        # MergerIntoSlowTraffic exposes a short but distinctive online event:
        # while ego accelerates along the merge lane, several frontal returns
        # share the road-speed of the adjacent slow flow and therefore have a
        # high absolute radial speed but almost no closing component.  This is
        # a surrounding-traffic signature; ego speed merely bounds the event
        # phase.  Evaluate it before broad low/high signatures and before a
        # startup-static class-04 lock is made permanent.  It may correct that
        # startup-only class-04 interpretation, but never a confirmed
        # dynamic-closing class-04 or any other specific semantic lock.
        for profile, class_id, confidence in merge_flow_start:
            deadline = max(1, int(_finite(
                profile.get("startup_deadline_frames"), 80
            )))
            if self.semantic_observation_count > deadline:
                continue
            window = max(1, int(_finite(
                profile.get("persistence_window_frames"), 8
            )))
            persistence = max(1, int(_finite(
                profile.get("min_persistence_frames"), 3
            )))
            recent = list(self.semantic_static_history)[-window:]
            legacy_qualifying = sum(
                1 for item in recent
                if (
                    _finite(profile.get("min_ego_speed"), 4.0)
                    <= item["ego_speed"]
                    <= _finite(profile.get("max_ego_speed"), 10.5)
                    and item["dynamic"] >= _finite(
                        profile.get("min_front_dynamic"), 4.0
                    )
                    and _finite(profile.get("min_velocity_abs_q90"), 7.5)
                    <= item["velocity_abs_q90"]
                    <= _finite(profile.get("max_velocity_abs_q90"), 10.5)
                    and item["closing_velocity_q90"] <= _finite(
                        profile.get("max_closing_velocity_q90"), 0.5
                    )
                    and _finite(profile.get("min_depth_min"), 6.0)
                    <= item["depth_min"]
                    <= _finite(profile.get("max_depth_min"), 13.0)
                )
            )
            direct_window = max(1, int(_finite(
                profile.get("direct_flow_window_frames"), 5
            )))
            direct_persistence = max(1, int(_finite(
                profile.get("direct_flow_min_persistence_frames"), 3
            )))
            direct_recent = list(self.semantic_static_history)[-direct_window:]
            direct_qualifying = sum(
                1 for item in direct_recent
                if item["same_direction_flow_count"] >= _finite(
                    profile.get("direct_flow_min_count"), 6.0
                )
            )
            qualifying = max(legacy_qualifying, direct_qualifying)
            self.semantic_merge_flow_max_qualifying = max(
                self.semantic_merge_flow_max_qualifying, qualifying
            )
            correctable_startup_class04 = (
                self.semantic_specific_lock_class == 4
                and self.semantic_closing_max_consecutive == 0
            )
            if (
                (
                    legacy_qualifying >= persistence
                    or direct_qualifying >= direct_persistence
                )
                and (
                    self.semantic_specific_lock_class == UNKNOWN_CLASS
                    or correctable_startup_class04
                )
            ):
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return class_id, confidence, "semantic_merge_flow_start", guarded

        # The early opposing-flow envelope is shared by classes 33 and 44.
        # Once that ambiguity is observed, hold classification until the
        # first actor reaches the merge point.  A turning actor creates a
        # dense, slow-near radar footprint as it enters the ego's target
        # flow; a straight opposing actor does not.  This post-pass topology
        # is the class distinction, whereas an earlier distance threshold is
        # not.  The router's UNKNOWN fallback explicitly stops during this
        # bounded arbitration interval.
        if self.semantic_flow_pending_class != UNKNOWN_CLASS:
            return self._resolve_pending_flow(
                latest_flow_fast_close, latest_flow_slow_near,
                _finite(features.get("ego_speed"), 0.0),
                turning_flow_start, guarded,
            )

        # Some featureless junctions share broad point/depth envelopes with
        # other ordinary classes, but have a stable physical arrangement in
        # the first four raw-radar frames: a low-variance count of near,
        # level returns together with a dense left-wide background.  Evaluate
        # this once, at route startup, then preserve it through the existing
        # specific-lock and hysteresis path.  A rolling four-frame check would
        # be unsafe because unrelated events can later create the same shape.
        for profile, class_id, confidence in static_background_geometry4:
            frames = max(2, int(_finite(profile.get("evaluation_frames"), 4)))
            if self.semantic_observation_count != frames:
                continue
            recent = list(self.semantic_static_history)[:frames]
            if len(recent) != frames:
                continue
            near_level_std = float(np.std([item["near_level"] for item in recent]))
            left_wide_median = float(np.median([
                item["left_wide_mid"] for item in recent
            ]))
            self.semantic_geometry4_evaluated = True
            self.semantic_geometry4_near_level_std = near_level_std
            self.semantic_geometry4_left_wide_median = left_wide_median
            if (
                near_level_std <= _finite(profile.get("max_near_level_std"), 0.0)
                and left_wide_median >= _finite(profile.get("min_left_wide_median"), 1e9)
            ):
                self.semantic_geometry4_matched = True
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return (
                    class_id, confidence,
                    "semantic_static_background_geometry4", guarded,
                )

        # A featureless vanilla junction must be identified before its short
        # initial green phase expires.  Its evidence is the *static radar
        # background geometry*, not the absence of an actor alone.  Limit the
        # decision to a startup window: the same point-count band can occur
        # much later in unrelated routes after their actors have moved away.
        for profile, class_id, confidence in static_background_start:
            min_points = _finite(profile.get("min_point_count"), 0.0)
            max_points = _finite(profile.get("max_point_count"), 1e9)
            min_front = _finite(profile.get("min_front_count"), 0.0)
            max_front = _finite(profile.get("max_front_count"), 1e9)
            max_close = _finite(profile.get("max_front_close"), 1e9)
            max_dynamic = _finite(profile.get("max_dynamic"), 0.0)
            max_min_depth = _finite(profile.get("max_min_depth_seen"), 1e9)
            min_depth = _finite(profile.get("min_depth_min"), 0.0)
            max_depth = _finite(profile.get("max_depth_min"), 1e9)
            min_depth_median = _finite(profile.get("min_depth_median"), 0.0)
            max_depth_median = _finite(profile.get("max_depth_median"), 1e9)
            min_depth_mean = _finite(profile.get("min_depth_mean"), 0.0)
            max_depth_mean = _finite(profile.get("max_depth_mean"), 1e9)
            min_depth_std = _finite(profile.get("min_depth_std"), 0.0)
            max_depth_std = _finite(profile.get("max_depth_std"), 1e9)
            persistence = max(1, int(_finite(profile.get("min_persistence_frames"), 3)))
            window = max(1, int(_finite(profile.get("persistence_window_frames"), 4)))
            deadline = max(1, int(_finite(profile.get("startup_deadline_frames"), 16)))
            if self.semantic_observation_count > deadline:
                continue
            recent = list(self.semantic_static_history)[-window:]
            qualifying = sum(
                1 for item in recent
                if (
                    min_points <= item["points"] <= max_points
                    and min_front <= item["front"] <= max_front
                    and item["close"] <= max_close
                    and item["dynamic"] <= max_dynamic
                    and min_depth <= item["depth_min"] <= max_depth
                    and min_depth_median <= item["depth_median"] <= max_depth_median
                    and min_depth_mean <= item["depth_mean"] <= max_depth_mean
                    and min_depth_std <= item["depth_std"] <= max_depth_std
                )
            )
            if (
                qualifying >= persistence
                and self.semantic_min_depth_seen <= max_min_depth
            ):
                # This signature is deliberately bounded to route startup and
                # describes background geometry, not a transient actor.  Once
                # confirmed it remains the strongest route-local evidence;
                # a vehicle waiting at the signal must not later be relabelled
                # as generic static-close merely because it stayed stopped.
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return (
                    class_id, confidence,
                    "semantic_static_background_start", guarded,
                )

        # ConstructionObstacleTwoWays presents a stationary, close forward
        # radar wall on a nearly straight online route before its opposing
        # traffic arrives.  Preserve that temporal precursor so the later
        # traffic cannot be mistaken for class 44's turning-flow event.  The
        # route term is only an online heading error; no map coordinate, XML,
        # route id, filename or evaluator metadata is consumed.
        for profile, class_id, confidence in static_straight_obstacle_start:
            deadline = max(1, int(_finite(
                profile.get("startup_deadline_frames"), 60
            )))
            if self.semantic_observation_count > deadline:
                continue
            window = max(1, int(_finite(
                profile.get("persistence_window_frames"), 10
            )))
            persistence = max(1, int(_finite(
                profile.get("min_persistence_frames"), window
            )))
            recent = list(self.semantic_static_history)[-window:]
            qualifying = sum(
                1 for item in recent
                if (
                    item["ego_speed"] <= _finite(profile.get("max_ego_speed"), 0.5)
                    and _finite(profile.get("min_front_min"), 5.0)
                    <= item["front_min"]
                    <= _finite(profile.get("max_front_min"), 12.0)
                    and item["close"] >= _finite(profile.get("min_front_close"), 1.0)
                    and item["dynamic"] <= _finite(profile.get("max_dynamic"), 1.0)
                    and item["route_heading_error"] <= _finite(
                        profile.get("max_route_heading_error_rad"), 0.2
                    )
                )
            )
            if len(recent) >= window and qualifying >= persistence:
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return (
                    class_id, confidence,
                    "semantic_static_straight_obstacle_start", guarded,
                )

        # A signalized opposing flow is not the generic high-dynamic cut-in
        # family: it repeatedly fills the forward radar while the ego waits
        # near the junction.  Require a dense dynamic+closing pattern in a
        # bounded startup window.  The deadline is part of the online event
        # lifecycle and prevents later construction traffic from acquiring
        # this identity; no route metadata or test order is consulted.
        opposing_class_ids = set()
        for profile, class_id, confidence in opposing_flow_start:
            opposing_class_ids.add(class_id)
            deadline = max(1, int(_finite(profile.get("startup_deadline_frames"), 150)))
            if self.semantic_observation_count > deadline:
                continue
            window = max(1, int(_finite(profile.get("persistence_window_frames"), 8)))
            persistence = max(1, int(_finite(profile.get("min_persistence_frames"), 6)))
            min_dynamic = _finite(profile.get("min_dynamic"), 8.0)
            min_closing = _finite(profile.get("min_closing"), 8.0)
            min_front = _finite(profile.get("min_front_min"), 0.0)
            max_front = _finite(profile.get("max_front_min"), 10.0)
            max_ego_speed = _finite(profile.get("max_ego_speed"), 0.75)
            recent = list(self.semantic_static_history)[-window:]
            qualifying = sum(
                1 for item in recent
                if (
                    item["dynamic"] >= min_dynamic
                    and item["closing"] >= min_closing
                    and min_front <= item["front_min"] <= max_front
                    and item["ego_speed"] <= max_ego_speed
                )
            )
            self.semantic_opposing_flow_max_qualifying = max(
                self.semantic_opposing_flow_max_qualifying, qualifying
            )
            if qualifying >= persistence:
                if turning_flow_start:
                    self.semantic_flow_pending_class = class_id
                    # The early envelope can become conclusive on the same
                    # frame that the first actor reaches the merge point. Do
                    # not discard that frame and classify from the following
                    # sparse tail; use its joint fast-close/slow-near shape.
                    return self._resolve_pending_flow(
                        latest_flow_fast_close, latest_flow_slow_near,
                        _finite(features.get("ego_speed"), 0.0),
                        turning_flow_start, guarded,
                    )
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return class_id, confidence, "semantic_opposing_flow_start", guarded
            prehold_min = max(
                1, int(_finite(profile.get("prehold_min_persistence_frames"), 0))
            )
            prehold_max_frames = max(
                1, int(_finite(profile.get("prehold_max_frames"), 60))
            )
            if (
                profile.get("prehold_min_persistence_frames") is not None
                and qualifying >= prehold_min
                and self.semantic_specific_lock_class == UNKNOWN_CLASS
                and _finite(features.get("ego_speed"), 0.0) <= max_ego_speed
            ):
                if self.semantic_flow_prehold_started_at is None:
                    self.semantic_flow_prehold_started_at = self.semantic_observation_count
                prehold_age = (
                    self.semantic_observation_count
                    - self.semantic_flow_prehold_started_at
                )
                if prehold_age <= prehold_max_frames:
                    return (
                        UNKNOWN_CLASS, 0.0,
                        "semantic_flow_prehold", guarded,
                    )

        if self.semantic_specific_lock_class in opposing_class_ids:
            return (
                self.semantic_specific_lock_class,
                self.semantic_specific_lock_confidence,
                "semantic_specific_lock_persisted",
                guarded,
            )

        # Track an approaching road-speed cluster across time.  This gate is
        # delayed until class 33's bounded opposing-flow gate has completed,
        # and it cannot replace any earlier specific lock such as class 04.
        if self.semantic_specific_lock_class == UNKNOWN_CLASS:
            for profile, class_id, confidence in turning_flow_start:
                minimum_observations = max(
                    1, int(_finite(profile.get("minimum_observations"), 145))
                )
                deadline = max(
                    minimum_observations,
                    int(_finite(profile.get("startup_deadline_frames"), 240)),
                )
                if not minimum_observations <= self.semantic_observation_count <= deadline:
                    continue
                window = max(4, int(_finite(
                    profile.get("persistence_window_frames"), 40
                )))
                recent = list(self.semantic_fast_approach_history)[-window:]
                if len(recent) < window:
                    continue
                half = len(recent) // 2
                earlier = [value for value in recent[:half] if value is not None]
                later = [value for value in recent[half:] if value is not None]
                persistence = max(2, int(_finite(
                    profile.get("min_persistence_frames"), 8
                )))
                if not earlier or not later or len(earlier) + len(later) < persistence:
                    continue
                depth_drop = float(np.median(earlier) - np.median(later))
                if depth_drop >= _finite(profile.get("min_depth_drop"), 6.0):
                    self.semantic_specific_lock_class = class_id
                    self.semantic_specific_lock_confidence = confidence
                    return class_id, confidence, "semantic_turning_flow_start", guarded

        turning_class_ids = {item[1] for item in turning_flow_start}
        if self.semantic_specific_lock_class in turning_class_ids:
            return (
                self.semantic_specific_lock_class,
                self.semantic_specific_lock_confidence,
                "semantic_specific_lock_persisted",
                guarded,
            )

        # Once the same-direction merge-flow event has specifically identified
        # class 16, the actor naturally rotates through a modest approaching
        # radial signature while crossing the merge geometry.  That later
        # phase is part of the same event and must not relabel the route as the
        # class-04 construction-closing family.  Class16 may still replace only
        # a startup-static class04 above; this guard makes that promotion
        # one-way and route-local.
        if self.semantic_specific_lock_class == 16:
            return (
                self.semantic_specific_lock_class,
                self.semantic_specific_lock_confidence,
                "semantic_specific_lock_persisted",
                guarded,
            )

        # A persistent approaching dynamic cluster is more specific than a
        # generic high-dynamic signature.  Check it first so a construction
        # route with opposing traffic is not prematurely claimed by the
        # broad cut-in profile.  All fields are current radar-derived
        # quantities and the decision is bounded by this persistence window.
        for (
            minimum_dynamic,
            maximum_dynamic,
            minimum_closing,
            minimum_close,
            maximum_front_min,
            persistence,
            window,
            min_consecutive,
            approach_deferral,
            broad_lock_delay,
            class_id,
            confidence,
        ) in sorted(dynamic_closing, reverse=True):
            recent = list(self.semantic_static_history)[-window:]
            qualifying = sum(
                1
                for item in recent
                if (
                    minimum_dynamic <= item["dynamic"] <= maximum_dynamic
                    and item.get("closing", 0.0) >= minimum_closing
                    and item["close"] >= minimum_close
                    and 0.0 < item.get("front_min", 0.0) <= maximum_front_min
                )
            )
            consecutive = 0
            max_consecutive = 0
            for item in recent:
                if (
                    minimum_dynamic <= item["dynamic"] <= maximum_dynamic
                    and item.get("closing", 0.0) >= minimum_closing
                    and item["close"] >= minimum_close
                    and 0.0 < item.get("front_min", 0.0) <= maximum_front_min
                ):
                    consecutive += 1
                    max_consecutive = max(max_consecutive, consecutive)
                else:
                    consecutive = 0
            required_consecutive = min_consecutive or persistence
            self.semantic_closing_max_consecutive = max(
                self.semantic_closing_max_consecutive, max_consecutive
            )
            # A close, approaching-looking burst can be shared by several
            # ordinary families before their distinguishing event is visible.
            # Defer only the broad static/low-dynamic fallbacks for a bounded
            # number of sensor frames. Specific dynamic signatures still have
            # priority and may lock once their persistence gates are met.
            if (
                approach_deferral > 0
                and latest_close >= minimum_close
                and _finite(features.get("front_closing_count", 0.0)) >= minimum_closing
            ):
                self.semantic_approach_seen = True
                self.semantic_approach_defer_remaining = max(
                    self.semantic_approach_defer_remaining, approach_deferral
                )
            if (
                (
                    maximum_front_min >= 1e8
                    or not self.semantic_high_dynamic_seen
                )
                and qualifying >= persistence
                and max_consecutive >= required_consecutive
            ):
                self.semantic_high_dynamic_seen = True
                self.semantic_broad_lock_remaining = 0
                self.semantic_broad_lock_completed = True
                self.semantic_specific_lock_class = class_id
                self.semantic_specific_lock_confidence = confidence
                return class_id, confidence, "semantic_dynamic_closing", guarded

        # Once a specific closing signature has been confirmed, preserve that
        # category for the rest of this route.  A construction route can later
        # expose a dense opposing-flow burst that also satisfies the generic
        # high-dynamic signature; replacing the already confirmed specific
        # event would route the same scene into a less appropriate rule set.
        # This uses only the route-local online radar state and is reset above.
        if self.semantic_specific_lock_class != UNKNOWN_CLASS:
            return (
                self.semantic_specific_lock_class,
                self.semantic_specific_lock_confidence,
                "semantic_specific_lock_persisted",
                guarded,
            )

        # A strong dynamic event dominates a low-dynamic or static
        # interpretation, but only after it persists in a bounded temporal
        # window.  A single startup/background spike must not turn a side-lane
        # hazard into the high-dynamic class.  Once observed, the high event
        # remains the route's evidence state so that a later disappearance of
        # the actor cannot make the rule router oscillate back to a static or
        # low-dynamic class.
        for minimum, persistence, window, class_id, confidence in sorted(high, reverse=True):
            recent = list(self.semantic_dynamic_history)[-window:]
            persistence_count = sum(1 for value in recent if value >= minimum)
            self.semantic_high_persistence_count = max(
                self.semantic_high_persistence_count, persistence_count
            )
            if persistence_count >= persistence:
                self.semantic_high_dynamic_seen = True
                self.semantic_broad_lock_remaining = 0
                self.semantic_broad_lock_completed = True
                return class_id, confidence, "semantic_dynamic_high", guarded

        if not self.semantic_high_dynamic_seen:
            broad_delay = max(
                (item[9] for item in dynamic_closing), default=0
            )
            broad_candidate = None
            if self.semantic_approach_defer_remaining > 0:
                self.semantic_approach_defer_remaining -= 1
            for minimum, maximum, class_id, confidence in sorted(low, reverse=True):
                # Ignore isolated values above the low-dynamic band unless
                # they also satisfy a persistent high signature.  This is a
                # robust temporal peak, not a route-specific exception.
                low_values = [
                    value for value in self.semantic_dynamic_history
                    if value <= maximum
                ]
                low_peak = max(low_values) if low_values else 0.0
                if minimum <= low_peak <= maximum:
                    self.semantic_low_dynamic_seen = True
                    broad_candidate = (class_id, confidence, "semantic_dynamic_low")
                    break

            if broad_candidate is None:
                for minimum, side_minimum, frames, class_id, confidence in sorted(static, reverse=True):
                    static_frames = sum(
                        1
                        for item in list(self.semantic_static_history)[-4:]
                        if (
                            item["dynamic"] <= 0.0
                            and item["close"] >= minimum
                            and min(item["left"], item["right"]) >= side_minimum
                        )
                    )
                    if (
                        latest_dynamic <= 0.0
                        and latest_close >= minimum
                        and min(latest_left, latest_right) >= side_minimum
                        and static_frames >= frames
                    ):
                        self.semantic_static_seen = True
                        broad_candidate = (class_id, confidence, "semantic_static_close")
                        break

            # A low-dynamic glimpse is only a broad category cue.  If this
            # route has already shown an approaching event and the same close
            # obstruction is still present, do not hand control to that broad
            # class merely because closing points flickered out for a few
            # radar frames.  Keep the safe UNKNOWN path until either the
            # specific closing signature matures above or the close obstacle
            # actually clears.  True side-lane low-dynamic events are not
            # delayed: their front obstacle is outside this close corridor.
            approach_obstacle_pending = any(
                self.semantic_approach_seen
                and latest_close >= item[3]
                and 0.0 < _finite(features.get("front_min", 0.0)) <= item[4]
                for item in dynamic_closing
            )
            if (
                broad_candidate is not None
                and broad_candidate[2] == "semantic_dynamic_low"
                and approach_obstacle_pending
            ):
                return (
                    UNKNOWN_CLASS,
                    0.0,
                    "semantic_approach_obstacle_pending",
                    guarded,
                )

            if broad_candidate is not None and broad_delay > 0:
                # Class-02/static and class-03/low signatures are useful
                # fallbacks, but they are not specific enough to commit while
                # a learned class-04 closing event may still be forming. Start
                # one bounded per-route delay on the first broad observation;
                # later approach frames refresh it, while a qualifying
                # specific signature above cancels it immediately.
                if (
                    broad_candidate[2] == "semantic_static_close"
                    and not self.semantic_broad_lock_started
                    and not self.semantic_broad_lock_completed
                ):
                    self.semantic_broad_lock_started = True
                    self.semantic_broad_lock_remaining = broad_delay
                if self.semantic_approach_defer_remaining > 0 and self.semantic_broad_lock_started:
                    self.semantic_broad_lock_remaining = max(
                        self.semantic_broad_lock_remaining, broad_delay
                    )
                if self.semantic_broad_lock_remaining > 0:
                    self.semantic_broad_lock_remaining -= 1
                    if self.semantic_broad_lock_remaining <= 0:
                        self.semantic_broad_lock_completed = True
                    return UNKNOWN_CLASS, 0.0, (
                        "semantic_approach_pending"
                        if self.semantic_approach_defer_remaining > 0
                        else "semantic_broad_candidate_pending"
                    ), guarded
                self.semantic_broad_lock_completed = True

            if self.semantic_approach_defer_remaining > 0:
                return UNKNOWN_CLASS, 0.0, "semantic_approach_pending", guarded

            if broad_candidate is not None:
                return broad_candidate[0], broad_candidate[1], broad_candidate[2], guarded

            if self.semantic_broad_lock_remaining > 0:
                self.semantic_broad_lock_remaining -= 1
                if self.semantic_broad_lock_remaining <= 0:
                    self.semantic_broad_lock_completed = True
                return UNKNOWN_CLASS, 0.0, "semantic_broad_candidate_pending", guarded

            for minimum, side_minimum, frames, class_id, confidence in sorted(static, reverse=True):
                static_frames = sum(
                    1
                    for item in list(self.semantic_static_history)[-4:]
                    if (
                        item["dynamic"] <= 0.0
                        and item["close"] >= minimum
                        and min(item["left"], item["right"]) >= side_minimum
                    )
                )
                if (
                    latest_dynamic <= 0.0
                    and latest_close >= minimum
                    and min(latest_left, latest_right) >= side_minimum
                    and static_frames >= frames
                ):
                    self.semantic_static_seen = True
                    return class_id, confidence, "semantic_static_close", guarded

            if self.semantic_low_dynamic_seen:
                for minimum, maximum, class_id, confidence in sorted(low, reverse=True):
                    low_values = [
                        value for value in self.semantic_dynamic_history
                        if value <= maximum
                    ]
                    low_peak = max(low_values) if low_values else 0.0
                    if minimum <= low_peak <= maximum:
                        return class_id, confidence, "semantic_dynamic_low_persisted", guarded
            if self.semantic_static_seen and self.semantic_max_dynamic <= 0.0:
                for minimum, side_minimum, frames, class_id, confidence in sorted(static, reverse=True):
                    return class_id, confidence, "semantic_static_persisted", guarded

        # At this point a centroid may point at a guarded class solely because
        # the route has not exposed its event yet.  Hold UNKNOWN for those
        # guarded predictions; otherwise let unguarded incremental classes use
        # the normal centroid path.
        return UNKNOWN_CLASS, 0.0, "semantic_evidence_pending", guarded

    def _predict(self, vector):
        if self.model is None:
            return UNKNOWN_CLASS, 0.0, UNKNOWN_CLASS, 0.0, [], "model_not_available"
        mean = self.model["mean"]
        scale = self.model["scale"]
        if len(mean) != len(vector) or len(scale) != len(vector):
            return UNKNOWN_CLASS, 0.0, UNKNOWN_CLASS, 0.0, [], "model_dimension_mismatch"
        scale = np.where(np.abs(scale) < 1e-6, 1.0, scale)
        vector = (np.asarray(vector, dtype=np.float64) - mean) / scale
        distances = []
        for class_id, centroid in self.model["centroids"].items():
            if len(centroid) != len(vector):
                continue
            distance = float(np.mean((vector - centroid) ** 2))
            distances.append((distance, int(class_id)))
        if not distances:
            return UNKNOWN_CLASS, 0.0, UNKNOWN_CLASS, 0.0, [], "no_compatible_centroid"
        # A single labelled class cannot distinguish that class from any
        # unseen ordinary class.  Treating the absent second centroid as a
        # synthetic distance would make every route lock to the only trained
        # label and contaminate the category router.  Offline development may
        # train incrementally, but runtime classification requires at least
        # two observed class centroids before it can claim a class.
        if len(distances) < 2:
            nearest = [(cid, round(dist, 6)) for dist, cid in distances[:5]]
            return UNKNOWN_CLASS, 0.0, UNKNOWN_CLASS, 0.0, nearest, "insufficient_class_centroids"
        distances.sort()
        best_distance, candidate = distances[0]
        second_distance = distances[1][0] if len(distances) > 1 else best_distance + 1.0
        # Convert the margin to a bounded confidence; a small margin remains
        # UNKNOWN rather than making a random category choice.
        margin = max(0.0, second_distance - best_distance)
        confidence = float(1.0 - math.exp(-margin))
        nearest = [(cid, round(dist, 6)) for dist, cid in distances[:5]]
        return candidate, confidence, candidate, margin, nearest, "centroid_distance"

    def observe(
        self, packet, ego_speed=0.0, steer=0.0, brake=0.0,
        timestamp=None, route_heading_error=None,
    ):
        features = self.extractor.extract(packet, ego_speed, steer, brake, timestamp)
        features["route_heading_error"] = (
            _finite(route_heading_error, 1e9)
            if route_heading_error is not None else 1e9
        )
        candidate, confidence, _, margin, nearest, reason = self._predict(features["vector"])
        semantic = self._semantic_candidate(features)
        if semantic is not None:
            semantic_candidate, semantic_confidence, semantic_reason, guarded = semantic
            semantic_profile = (
                ((self.model or {}).get("semantic_signatures") or {}).get(
                    str(semantic_candidate), {}
                )
                if semantic_candidate != UNKNOWN_CLASS else {}
            ) or {}
            minimum_candidate_observations = max(1, int(_finite(
                semantic_profile.get("minimum_candidate_observations"), 1
            )))
            if (
                semantic_candidate != UNKNOWN_CLASS
                and self.locked_class_id == UNKNOWN_CLASS
                and self.semantic_observation_count < minimum_candidate_observations
            ):
                # Evidence may be conclusive before it is safe to hand over
                # vehicle control. Preserve the route-specific evidence, but
                # keep the runtime class UNKNOWN until the profile's proven
                # control-ownership maturity point. This is sensor-time based
                # and contains no route or evaluator identity.
                semantic_candidate = UNKNOWN_CLASS
                semantic_confidence = 0.0
                semantic_reason = semantic_reason + "_control_maturing"
            if semantic_candidate != UNKNOWN_CLASS:
                candidate = semantic_candidate
                confidence = semantic_confidence
                margin = semantic_confidence
                reason = semantic_reason
            elif candidate in guarded:
                candidate = UNKNOWN_CLASS
                confidence = 0.0
                margin = 0.0
                reason = semantic_reason
        enough_history = int(features["history_frames"]) >= self.min_history
        if not enough_history:
            candidate = UNKNOWN_CLASS
            confidence = 0.0
            reason = "warming_temporal_window"
        if candidate != UNKNOWN_CLASS and confidence >= self.confidence_threshold:
            self.candidate_history.append(candidate)
        else:
            self.candidate_history.append(UNKNOWN_CLASS)

        locked = self.locked_class_id
        if locked == UNKNOWN_CLASS:
            if candidate != UNKNOWN_CLASS:
                count = sum(1 for item in list(self.candidate_history)[-4:] if item == candidate)
                if count >= 3:
                    locked = candidate
                    self.locked_class_id = candidate
                    self.locked_confidence = confidence
        elif candidate == locked:
            self.locked_confidence = max(self.locked_confidence * 0.95, confidence)
        elif (
            locked == 4
            and candidate == 16
            and reason == "semantic_merge_flow_start"
        ):
            # Class04 may lock on the merge ramp's static guardrail before
            # traffic arrives.  A temporally confirmed same-direction flow is
            # more specific online evidence and must take ownership before a
            # stop in the actor-flow conflict corridor.  This transition uses
            # only the radar window; no route identity or evaluator metadata.
            locked = candidate
            self.locked_class_id = candidate
            self.locked_confidence = confidence
        else:
            # Hysteresis: a replacement must be consistently observed and have
            # a strong margin.  A single noisy radar frame cannot switch rules.
            recent = list(self.candidate_history)[-8:]
            if (
                candidate != UNKNOWN_CLASS
                and confidence >= min(0.90, self.confidence_threshold + 0.12)
                and recent.count(candidate) >= 6
            ):
                locked = candidate
                self.locked_class_id = candidate
                self.locked_confidence = confidence

        evidence = {
            "history_frames": features["history_frames"],
            "point_count": features["point_count"],
            "front_close_count": features["front_close_count"],
            "front_ttc": features["front_ttc"],
            "depth_trend": features["depth_trend"],
            "nearest_candidates": nearest,
            "semantic_max_dynamic": round(float(self.semantic_max_dynamic), 6),
            "semantic_high_persistence_count": int(self.semantic_high_persistence_count),
            "semantic_static_frames": int(sum(
                1 for item in self.semantic_static_history
                if item["dynamic"] <= 0.0 and item["close"] >= 1.0
            )),
            "semantic_high_dynamic_seen": bool(self.semantic_high_dynamic_seen),
            "semantic_low_dynamic_seen": bool(self.semantic_low_dynamic_seen),
            "semantic_static_seen": bool(self.semantic_static_seen),
            "semantic_approach_defer_remaining": int(self.semantic_approach_defer_remaining),
            "semantic_approach_seen": bool(self.semantic_approach_seen),
            "semantic_broad_lock_remaining": int(self.semantic_broad_lock_remaining),
            "semantic_broad_lock_started": bool(self.semantic_broad_lock_started),
            "semantic_broad_lock_completed": bool(self.semantic_broad_lock_completed),
            "semantic_specific_lock_class": int(self.semantic_specific_lock_class),
            "semantic_specific_lock_confidence": float(
                self.semantic_specific_lock_confidence
            ),
            "semantic_closing_max_consecutive": int(
                self.semantic_closing_max_consecutive
            ),
            "semantic_geometry4_evaluated": bool(self.semantic_geometry4_evaluated),
            "semantic_geometry4_near_level_std": round(
                float(self.semantic_geometry4_near_level_std), 6
            ),
            "semantic_geometry4_left_wide_median": round(
                float(self.semantic_geometry4_left_wide_median), 6
            ),
            "semantic_geometry4_matched": bool(self.semantic_geometry4_matched),
            "semantic_opposing_flow_max_qualifying": int(
                self.semantic_opposing_flow_max_qualifying
            ),
            "semantic_merge_flow_max_qualifying": int(
                self.semantic_merge_flow_max_qualifying
            ),
            "semantic_flow_pending_class": int(self.semantic_flow_pending_class),
            "semantic_flow_pass_seen": bool(self.semantic_flow_pass_seen),
            "semantic_flow_post_frames": int(self.semantic_flow_post_frames),
            "semantic_flow_max_slow_near": float(self.semantic_flow_max_slow_near),
            "semantic_flow_episode_active": bool(self.semantic_flow_episode_active),
            "semantic_flow_completed_episodes": int(self.semantic_flow_completed_episodes),
            "semantic_flow_clear_frames": int(self.semantic_flow_clear_frames),
            "semantic_flow_dense_clear_frames": int(self.semantic_flow_dense_clear_frames),
            "semantic_flow_approach_speed_samples": int(
                len(self.semantic_flow_approach_speeds)
            ),
            "semantic_flow_approach_speed_q90": float(
                np.quantile(list(self.semantic_flow_approach_speeds), 0.90)
                if self.semantic_flow_approach_speeds else 0.0
            ),
            "semantic_minimum_candidate_observations": int(
                max(1, int(_finite(
                    (((self.model or {}).get("semantic_signatures") or {}).get(
                        str(self.semantic_specific_lock_class), {}
                    ) or {}).get("minimum_candidate_observations"), 1
                )))
            ),
        }
        decision = {
            "class_id": int(locked),
            "candidate_class_id": int(candidate),
            "confidence": round(float(self.locked_confidence if locked else confidence), 6),
            "candidate_confidence": round(float(confidence), 6),
            "margin": round(float(margin), 6),
            "reason": reason,
            "evidence": evidence,
            "feature_vector": features["vector"],
            "feature_names": vector_feature_names(),
            "history_frames": int(features["history_frames"]),
            "locked": bool(locked != UNKNOWN_CLASS),
        }
        if self.last_decision is None or decision["class_id"] != self.last_decision["class_id"]:
            decision["transition"] = {
                "from": int(self.last_decision["class_id"] if self.last_decision else UNKNOWN_CLASS),
                "to": int(decision["class_id"]),
            }
        else:
            decision["transition"] = None
        self.last_decision = decision
        self.last_timestamp = timestamp
        decision["features"] = features
        return decision
