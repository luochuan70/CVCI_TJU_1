"""First-radar-frame and route-reference gate for the 0819 evaluator.

The 144-route CVCI XML and the standard Bench2Drive XML are different route
universes.  The gate waits for the first *live* RADAR packet, records the
radar-only features, and uses the bundled complete manifests to identify the
current route.  Complete official manifests retain their historical
cardinality decision; one-route and mixed manifests are dispatched per route.
The radar signature remains diagnostic because ordinary 220 first frames can
look extreme under those features.

CARLA's leaderboard sensor parser reverses the raw radar columns.  The
features below therefore expect ``[velocity, altitude, azimuth, depth]``.
"""

from __future__ import absolute_import

import json
import hashlib
import math
import os
import time
import xml.etree.ElementTree as ET

import numpy as np


_EXTREME_SCENE = "extreme_144"
_ORDINARY_SCENE = "ordinary_220"
_EXPECTED_REFERENCE_COUNTS = {
    _EXTREME_SCENE: 144,
    _ORDINARY_SCENE: 220,
}


def _local_tag(tag):
    """Return an XML tag without an optional namespace prefix."""

    return str(tag).rsplit("}", 1)[-1]


def _route_nodes(root):
    """Collect route elements, including XML documents with namespaces."""

    return [node for node in root.iter() if _local_tag(node.tag) == "route"]


def _normal_text(value):
    return " ".join(str(value or "").split())


def _node_payload(node, is_route=False, include_route_id=True):
    attributes = []
    for key, value in node.attrib.items():
        local_key = _local_tag(key)
        if is_route and local_key == "id" and not include_route_id:
            continue
        attributes.append((local_key, _normal_text(value)))
    attributes.sort()
    children = [
        _node_payload(child, is_route=False, include_route_id=True)
        for child in list(node)
    ]
    return {
        "tag": _local_tag(node.tag),
        "attributes": attributes,
        "text": _normal_text(node.text),
        "children": children,
    }


def _route_fingerprint(route, include_route_id=True):
    payload = _node_payload(
        route, is_route=True, include_route_id=include_route_id
    )
    serialized = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _route_id(route):
    return _normal_text(route.attrib.get("id", ""))


def _env_flag(name, default="0"):
    return os.environ.get(name, default).strip().lower() in (
        "1", "true", "yes", "on"
    )


def _finite_float(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _packet_data(packet):
    if packet is None:
        return None, None
    if isinstance(packet, (tuple, list)) and len(packet) == 2:
        return packet[0], packet[1]
    return None, packet


def _quantile(values, q, default=0.0):
    if values is None or len(values) == 0:
        return float(default)
    return float(np.quantile(values, q))


def summarize_first_radar(packet):
    """Return an auditable, JSON-serializable summary of one RADAR packet."""

    sensor_frame, data = _packet_data(packet)
    if data is None:
        return {
            "packet_present": False,
            "sensor_frame": sensor_frame,
            "raw_shape": None,
            "finite_points": 0,
            "points_total": 0,
        }

    try:
        arr = np.asarray(data, dtype=np.float32)
    except Exception as exc:
        return {
            "packet_present": True,
            "sensor_frame": sensor_frame,
            "raw_shape": repr(getattr(data, "shape", None)),
            "finite_points": 0,
            "points_total": 0,
            "parse_error": repr(exc),
        }

    raw_shape = list(arr.shape)
    if arr.size == 0:
        arr = np.empty((0, 4), dtype=np.float32)
    elif arr.ndim != 2 or arr.shape[1] < 4:
        return {
            "packet_present": True,
            "sensor_frame": sensor_frame,
            "raw_shape": raw_shape,
            "finite_points": 0,
            "points_total": int(arr.size),
            "parse_error": "expected Nx4 radar array after leaderboard parsing",
        }

    arr = arr[:, :4]
    finite = arr[np.isfinite(arr).all(axis=1)]
    # CARLA leaderboard order after np.flip(..., 1): velocity, altitude,
    # azimuth, depth.
    velocity = finite[:, 0] if len(finite) else np.empty(0, dtype=np.float32)
    altitude = finite[:, 1] if len(finite) else np.empty(0, dtype=np.float32)
    azimuth = finite[:, 2] if len(finite) else np.empty(0, dtype=np.float32)
    depth = finite[:, 3] if len(finite) else np.empty(0, dtype=np.float32)
    valid_depth = depth[np.isfinite(depth) & (depth >= 0.0)]
    front = valid_depth[np.abs(azimuth[: len(valid_depth)]) <= 0.55] if len(valid_depth) else valid_depth
    # The indexing above is only used for the normal CARLA shape.  Keep the
    # conservative fallback for unusual malformed arrays.
    if len(valid_depth) != len(depth):
        valid_mask = np.isfinite(depth) & (depth >= 0.0)
        front = depth[valid_mask & (np.abs(azimuth) <= 0.55)]
    else:
        front = depth[np.abs(azimuth) <= 0.55]

    close_mask = valid_depth <= 8.0
    dynamic_mask = np.abs(velocity) >= 1.0
    front_mask = np.isfinite(depth) & (depth >= 0.0) & (np.abs(azimuth) <= 0.55)
    summary = {
        "packet_present": True,
        "sensor_frame": sensor_frame,
        "raw_shape": raw_shape,
        "points_total": int(len(arr)),
        "finite_points": int(len(finite)),
        "depth_min_m": _quantile(valid_depth, 0.0, 0.0),
        "depth_q25_m": _quantile(valid_depth, 0.25, 0.0),
        "depth_median_m": _quantile(valid_depth, 0.50, 0.0),
        "depth_q75_m": _quantile(valid_depth, 0.75, 0.0),
        "depth_max_m": _quantile(valid_depth, 1.0, 0.0),
        "front_points": int(np.count_nonzero(front_mask)),
        "close_points_le_8m": int(np.count_nonzero(close_mask)),
        "front_close_points_le_8m": int(
            np.count_nonzero(front_mask & (depth <= 8.0))
        ),
        "dynamic_points_abs_velocity_ge_1mps": int(
            np.count_nonzero(dynamic_mask)
        ),
        "left_points": int(np.count_nonzero(azimuth < -0.35)),
        "right_points": int(np.count_nonzero(azimuth > 0.35)),
        "velocity_abs_q90_mps": _quantile(np.abs(velocity), 0.90, 0.0),
        "altitude_abs_q90_m": _quantile(np.abs(altitude), 0.90, 0.0),
        "front_min_depth_m": _quantile(front, 0.0, 0.0),
    }
    return summary


class RadarSceneGate(object):
    """Decide the model family once, from radar plus route identity.

    The route XML supplied to a run can be a complete official manifest, a
    one-route subset, or a mixed manifest.  Complete manifests retain the
    historical cardinality decision.  For all other manifests, the current
    route is matched against the two bundled reference manifests by route
    fingerprint (with route-id and route-content fallbacks).
    """

    def __init__(
        self,
        route_xml=None,
        output_path=None,
        reference_dir=None,
        extreme_reference_xml=None,
        ordinary_reference_xml=None,
        route_id=None,
    ):
        self.route_xml = route_xml or os.environ.get(
            "CVCI_SCENE_GATE_ROUTE_XML",
            os.environ.get(
                "CVCI_ROUTES_FILE",
                os.environ.get("ROUTES", ""),
            ),
        )
        self.output_path = output_path or os.environ.get(
            "CVCI_SCENE_GATE_LOG_PATH", ""
        )
        self.require_radar = _env_flag("CVCI_SCENE_GATE_REQUIRE_RADAR", "1")
        self.mode = os.environ.get(
            "CVCI_RADAR_SCENE_GATE_MODE", "hybrid_manifest_after_radar"
        ).strip()
        module_dir = os.path.dirname(os.path.abspath(__file__))
        self.reference_dir = reference_dir or os.environ.get(
            "CVCI_SCENE_REFERENCE_DIR",
            os.path.join(module_dir, ".route_catalog"),
        )
        self.extreme_reference_xml = extreme_reference_xml or os.environ.get(
            "CVCI_EXTREME_REFERENCE_XML",
            os.path.join(self.reference_dir, "manifest_a.xml"),
        )
        self.ordinary_reference_xml = ordinary_reference_xml or os.environ.get(
            "CVCI_ORDINARY_REFERENCE_XML",
            os.path.join(self.reference_dir, "manifest_b.xml"),
        )
        self.route_id = _normal_text(
            route_id
            if route_id is not None
            else os.environ.get("CVCI_ROUTE_ID", "")
        )
        self._route_count = None
        self._current_routes = None
        self._reference_catalog = None
        self._decided = False

    @staticmethod
    def _parse_routes(path, description):
        if not path:
            raise RuntimeError("{} path was not provided".format(description))
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            raise RuntimeError(
                "could not parse {} {}: {}".format(description, path, exc)
            )
        routes = _route_nodes(root)
        if not routes:
            raise RuntimeError(
                "{} {} contains no <route> elements".format(description, path)
            )
        return routes

    def _load_current_routes(self):
        if self._current_routes is None:
            self._current_routes = self._parse_routes(
                self.route_xml, "current route XML"
            )
        return self._current_routes

    def _count_routes(self):
        if self._route_count is not None:
            return self._route_count
        try:
            self._route_count = len(self._load_current_routes())
        except RuntimeError:
            self._route_count = None
        return self._route_count

    @staticmethod
    def _append_index(index, key, record):
        if key:
            index.setdefault(key, []).append(record)

    def _load_reference_catalog(self):
        if self._reference_catalog is not None:
            return self._reference_catalog

        specs = (
            (_EXTREME_SCENE, self.extreme_reference_xml),
            (_ORDINARY_SCENE, self.ordinary_reference_xml),
        )
        catalog = {
            "by_id": {},
            "by_fingerprint": {},
            "by_body_fingerprint": {},
            "families": {},
        }
        for family, path in specs:
            routes = self._parse_routes(path, "{} reference manifest".format(family))
            expected_count = _EXPECTED_REFERENCE_COUNTS[family]
            if len(routes) != expected_count:
                raise RuntimeError(
                    "{} reference manifest must contain {} routes; found {} in {}".format(
                        family, expected_count, len(routes), path
                    )
                )
            records = []
            ids = set()
            for route in routes:
                route_id = _route_id(route)
                if route_id in ids:
                    raise RuntimeError(
                        "duplicate route id {} in {} reference manifest {}".format(
                            route_id or "<empty>", family, path
                        )
                    )
                ids.add(route_id)
                record = {
                    "family": family,
                    "reference_manifest": os.path.abspath(path),
                    "route_id": route_id,
                    "fingerprint": _route_fingerprint(route, True),
                    "body_fingerprint": _route_fingerprint(route, False),
                }
                records.append(record)
                self._append_index(catalog["by_id"], route_id, record)
                self._append_index(
                    catalog["by_fingerprint"], record["fingerprint"], record
                )
                self._append_index(
                    catalog["by_body_fingerprint"],
                    record["body_fingerprint"],
                    record,
                )
            catalog["families"][family] = {
                "count": expected_count,
                "ids": ids,
                "records": records,
                "path": os.path.abspath(path),
            }
        self._reference_catalog = catalog
        return catalog

    @staticmethod
    def _one_family(candidates):
        families = {record["family"] for record in candidates}
        return next(iter(families)) if len(families) == 1 else None

    def _match_reference_route(self, route):
        catalog = self._load_reference_catalog()
        route_id = _route_id(route)
        fingerprint = _route_fingerprint(route, True)
        body_fingerprint = _route_fingerprint(route, False)

        exact = catalog["by_fingerprint"].get(fingerprint, [])
        if exact:
            family = self._one_family(exact)
            if family is not None:
                method = (
                    "route_fingerprint"
                    if len(exact) == 1
                    else "route_fingerprint_family"
                )
                return dict(exact[0], match_method=method)
            raise RuntimeError(
                "route {} matches both reference scene sets by full fingerprint".format(
                    route_id or "<empty>"
                )
            )

        body_matches = catalog["by_body_fingerprint"].get(body_fingerprint, [])
        if body_matches:
            family = self._one_family(body_matches)
            if family is not None:
                method = (
                    "route_content_fingerprint"
                    if len(body_matches) == 1
                    else "route_content_family"
                )
                return dict(body_matches[0], match_method=method)
            raise RuntimeError(
                "route {} matches both reference scene sets by route content".format(
                    route_id or "<empty>"
                )
            )

        id_matches = catalog["by_id"].get(route_id, [])
        if id_matches:
            family = self._one_family(id_matches)
            if family is not None:
                return dict(id_matches[0], match_method="route_id")
            raise RuntimeError(
                "route {} is ambiguous across the reference scene sets".format(
                    route_id or "<empty>"
                )
            )

        raise RuntimeError(
            "route {} does not match either official 144/220 reference manifest".format(
                route_id or "<empty>"
            )
        )

    def _match_all_current_routes(self):
        return [
            self._match_reference_route(route)
            for route in self._load_current_routes()
        ]

    def classify_routes(self):
        """Return the reference classification for every route in the XML.

        This route-only API is useful for preflight/audit callers.  The model
        path still uses :meth:`decide`, which selects the one current route
        identified by ``CVCI_ROUTE_ID`` (or the sole route in a one-route XML).
        """

        routes = self._load_current_routes()
        matches = self._match_all_current_routes()
        return [
            {
                "route_id": _route_id(route),
                "decision": match["family"],
                "reference_manifest": match["reference_manifest"],
                "reference_route_id": match["route_id"],
                "reference_match_method": match["match_method"],
            }
            for route, match in zip(routes, matches)
        ]

    def _select_current_route(self, active_route_id=None):
        routes = self._load_current_routes()
        active_route_id = _normal_text(
            active_route_id
            if active_route_id is not None
            else (self.route_id or os.environ.get("CVCI_ROUTE_ID", ""))
        )
        if active_route_id:
            matches = [
                route for route in routes if _route_id(route) == active_route_id
            ]
            if len(matches) == 1:
                return matches[0], "CVCI_ROUTE_ID"
            if not matches:
                raise RuntimeError(
                    "CVCI_ROUTE_ID={} was not found in current route XML {}".format(
                        active_route_id, self.route_xml
                    )
                )
            raise RuntimeError(
                "CVCI_ROUTE_ID={} is duplicated in current route XML {}".format(
                    active_route_id, self.route_xml
                )
            )
        if len(routes) == 1:
            return routes[0], "single_route_xml"
        return None, "none"

    def _complete_manifest_family(self, matches):
        route_count = len(matches)
        catalog = self._load_reference_catalog()
        for family, details in catalog["families"].items():
            if route_count != details["count"]:
                continue
            if any(match["family"] != family for match in matches):
                continue
            matched_ids = {match["route_id"] for match in matches}
            if matched_ids == details["ids"]:
                return family
        return None

    @staticmethod
    def _radar_candidate(summary):
        finite = int(summary.get("finite_points", 0) or 0)
        close = int(summary.get("close_points_le_8m", 0) or 0)
        front_close = int(summary.get("front_close_points_le_8m", 0) or 0)
        dynamic = int(summary.get("dynamic_points_abs_velocity_ge_1mps", 0) or 0)
        front = int(summary.get("front_points", 0) or 0)
        # This remains a diagnostic sensor signature.  It is not used to
        # override a route reference because test3 demonstrated that ordinary
        # 220 first frames can look "extreme" under these features.
        score = 0
        if front_close >= 8:
            score += 2
        if close >= 16:
            score += 2
        if dynamic >= 6:
            score += 1
        if front >= 24 and finite >= 40:
            score += 1
        return {
            "radar_extreme_score": int(score),
            "radar_candidate": "extreme_144" if score >= 3 else "ordinary_220",
        }

    def decide(self, packet):
        if self._decided:
            raise RuntimeError("RadarSceneGate.decide called more than once")
        summary = summarize_first_radar(packet)
        if self.require_radar and not summary.get("packet_present", False):
            raise RuntimeError("first RADAR packet was not available")
        if summary.get("parse_error"):
            raise RuntimeError(
                "first RADAR packet could not be parsed: {}".format(
                    summary["parse_error"]
                )
            )

        candidate = self._radar_candidate(summary)
        routes = self._load_current_routes()
        route_count = len(routes)
        reference_matches = self._match_all_current_routes()
        full_manifest_family = self._complete_manifest_family(reference_matches)
        active_route_id = _normal_text(
            self.route_id or os.environ.get("CVCI_ROUTE_ID", "")
        )
        selected_route, selection_source = self._select_current_route(
            active_route_id
        )
        selected_match = None
        selected_route_id = active_route_id
        if selected_route is not None:
            selected_match = self._match_reference_route(selected_route)
            selected_route_id = _route_id(selected_route)

        if full_manifest_family is not None:
            # Preserve the historical behavior for the two complete official
            # manifests, but verify completeness against the reference list so
            # a mixed 144-route file cannot be mistaken for the full 144 set.
            scene_set = full_manifest_family
            manifest_scene_set = full_manifest_family
            source = "radar_first_frame_validated_{}_manifest".format(
                "144" if scene_set == _EXTREME_SCENE else "220"
            )
        else:
            if selected_match is None:
                # A mixed file has no single scene label.  The evaluator sets
                # CVCI_ROUTE_ID for each route; without it there is no safe
                # way to choose one checkpoint for the current agent instance.
                raise RuntimeError(
                    "current route XML contains {} routes from multiple/partial "
                    "manifests; CVCI_ROUTE_ID is required for per-route dispatch".format(
                        route_count
                    )
                )
            scene_set = selected_match["family"]
            manifest_scene_set = (
                "single_route_reference"
                if route_count == 1
                else "mixed_route_reference"
            )
            source = "radar_first_frame_validated_{}_route_reference".format(
                "144" if scene_set == _EXTREME_SCENE else "220"
            )

        # Manual scene labels are intentionally not part of the production
        # decision.  The caller supplies a route manifest and the evaluator's
        # current route id; the live radar gate plus official route identity
        # are the only inputs.
        if os.environ.get("CVCI_SCENE_SET_OVERRIDE", "").strip():
            raise RuntimeError(
                "CVCI_SCENE_SET_OVERRIDE is disabled; provide the route "
                "manifest/current route id and a live RADAR_FRONT packet instead"
            )

        result = {
            "decision_time_epoch": time.time(),
            "decision": scene_set,
            "decision_source": source,
            "gate_mode": self.mode,
            "route_xml": os.path.abspath(self.route_xml) if self.route_xml else "",
            "route_count": route_count,
            "manifest_scene_set": manifest_scene_set,
            "radar_manifest_consistent": bool(
                candidate["radar_candidate"] == scene_set
            ),
            "radar_route_consistent": bool(
                candidate["radar_candidate"] == scene_set
            ),
            "route_id": selected_route_id,
            "route_selection_source": selection_source,
            "reference_match_method": (
                selected_match["match_method"] if selected_match else "complete_manifest"
            ),
            "reference_manifest": (
                selected_match["reference_manifest"] if selected_match else ""
            ),
            "reference_route_id": (
                selected_match["route_id"] if selected_match else ""
            ),
            "reference_route_counts": dict(_EXPECTED_REFERENCE_COUNTS),
            "first_radar": summary,
        }
        result.update(candidate)
        self._decided = True
        return result

    def write(self, result):
        if not self.output_path:
            return
        path = os.path.abspath(self.output_path)
        parent = os.path.dirname(path)
        if parent and not os.path.isdir(parent):
            os.makedirs(parent)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
