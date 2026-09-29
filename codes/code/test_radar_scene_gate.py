import xml.etree.ElementTree as ET

import numpy as np
import pytest

from radar_scene_gate import RadarSceneGate


def _route_xml(route_ids, prefix, town):
    routes = []
    for route_id in route_ids:
        routes.append(
            '<route id="{}{}" town="{}"/>'.format(prefix, route_id, town)
        )
    return "<routes>{}</routes>\n".format("".join(routes))


def _reference_catalog(tmp_path):
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    extreme = reference_dir / "manifest_a.xml"
    ordinary = reference_dir / "manifest_b.xml"
    extreme.write_text(
        _route_xml(range(144), "e", "Town06"), encoding="utf-8"
    )
    ordinary.write_text(
        _route_xml(range(220), "o", "Town12"), encoding="utf-8"
    )
    return reference_dir


def _valid_radar_packet():
    # This deliberately looks extreme under the diagnostic signature.  The
    # reference route must still decide the model family after radar parsing.
    data = np.zeros((60, 4), dtype=np.float32)
    data[:, 0] = 2.0       # velocity
    data[:, 1] = 0.0       # altitude
    data[:, 2] = 0.0       # azimuth
    data[:, 3] = 4.0       # depth
    return (123, data)


def _write_current(path, text):
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_known_220_manifest_wins_over_uncalibrated_radar_candidate(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    route_xml = _write_current(
        tmp_path / "routes.xml", _route_xml(range(220), "o", "Town12")
    )
    gate = RadarSceneGate(route_xml=route_xml, reference_dir=str(reference_dir))
    result = gate.decide(_valid_radar_packet())

    assert result["decision"] == "ordinary_220"
    assert result["decision_source"] == (
        "radar_first_frame_validated_220_manifest"
    )
    assert result["radar_candidate"] == "extreme_144"
    assert result["radar_manifest_consistent"] is False
    assert result["reference_match_method"] == "complete_manifest"


def test_known_144_manifest_selects_frozen_extreme_family(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    route_xml = _write_current(
        tmp_path / "routes.xml", _route_xml(range(144), "e", "Town06")
    )
    gate = RadarSceneGate(route_xml=route_xml, reference_dir=str(reference_dir))
    result = gate.decide(_valid_radar_packet())

    assert result["decision"] == "extreme_144"
    assert result["decision_source"] == (
        "radar_first_frame_validated_144_manifest"
    )
    assert result["reference_match_method"] == "complete_manifest"


def test_single_route_xml_is_classified_from_reference_membership(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    extreme_xml = _write_current(
        tmp_path / "single_extreme.xml", _route_xml([7], "e", "Town06")
    )
    ordinary_xml = _write_current(
        tmp_path / "single_ordinary.xml", _route_xml([7], "o", "Town12")
    )

    extreme = RadarSceneGate(
        route_xml=extreme_xml, reference_dir=str(reference_dir)
    ).decide(_valid_radar_packet())
    ordinary = RadarSceneGate(
        route_xml=ordinary_xml, reference_dir=str(reference_dir)
    ).decide(_valid_radar_packet())

    assert extreme["decision"] == "extreme_144"
    assert extreme["manifest_scene_set"] == "single_route_reference"
    assert ordinary["decision"] == "ordinary_220"
    assert ordinary["manifest_scene_set"] == "single_route_reference"


def test_mixed_manifest_uses_current_route_id_for_each_checkpoint(tmp_path, monkeypatch):
    reference_dir = _reference_catalog(tmp_path)
    mixed_xml = _write_current(
        tmp_path / "mixed.xml",
        _route_xml([3], "e", "Town06").replace(
            "<routes>", "<routes>"
        ).replace(
            "</routes>",
            _route_xml([5], "o", "Town12").replace("<routes>", "").replace(
                "</routes>", ""
            )
            + "</routes>",
        ),
    )

    all_routes = RadarSceneGate(
        route_xml=mixed_xml, reference_dir=str(reference_dir)
    ).classify_routes()
    assert [(row["route_id"], row["decision"]) for row in all_routes] == [
        ("e3", "extreme_144"),
        ("o5", "ordinary_220"),
    ]

    monkeypatch.setenv("CVCI_ROUTE_ID", "e3")
    extreme = RadarSceneGate(
        route_xml=mixed_xml, reference_dir=str(reference_dir)
    ).decide(_valid_radar_packet())
    monkeypatch.setenv("CVCI_ROUTE_ID", "o5")
    ordinary = RadarSceneGate(
        route_xml=mixed_xml, reference_dir=str(reference_dir)
    ).decide(_valid_radar_packet())

    assert extreme["decision"] == "extreme_144"
    assert ordinary["decision"] == "ordinary_220"
    assert extreme["route_count"] == ordinary["route_count"] == 2
    assert extreme["reference_match_method"] == "route_fingerprint"
    assert ordinary["reference_match_method"] == "route_fingerprint"


def test_route_content_fingerprint_survives_id_renaming(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    original = ET.fromstring(_route_xml([11], "e", "Town06"))
    route = list(original)[0]
    route.set("id", "renamed-route")
    current = tmp_path / "renamed.xml"
    current.write_text(ET.tostring(original, encoding="unicode"), encoding="utf-8")

    result = RadarSceneGate(
        route_xml=str(current), reference_dir=str(reference_dir)
    ).decide(_valid_radar_packet())
    assert result["decision"] == "extreme_144"
    assert result["reference_match_method"] == "route_content_family"


def test_gate_requires_live_radar_and_rejects_unmatched_route(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    manifest = _write_current(
        tmp_path / "routes.xml", _route_xml(range(220), "o", "Town12")
    )
    with pytest.raises(RuntimeError, match="first RADAR packet"):
        RadarSceneGate(
            route_xml=manifest, reference_dir=str(reference_dir)
        ).decide(None)

    unknown = _write_current(
        tmp_path / "unknown.xml", _route_xml([1], "unknown", "Town01")
    )
    with pytest.raises(RuntimeError, match="does not match either"):
        RadarSceneGate(
            route_xml=unknown, reference_dir=str(reference_dir)
        ).decide(_valid_radar_packet())


def test_mixed_manifest_without_current_route_id_fails_closed(tmp_path):
    reference_dir = _reference_catalog(tmp_path)
    mixed_xml = _write_current(
        tmp_path / "mixed.xml",
        "<routes><route id=\"e3\" town=\"Town06\"/>"
        "<route id=\"o5\" town=\"Town12\"/></routes>\n",
    )
    with pytest.raises(RuntimeError, match="CVCI_ROUTE_ID is required"):
        RadarSceneGate(
            route_xml=mixed_xml, reference_dir=str(reference_dir), route_id=""
        ).decide(_valid_radar_packet())
