#!/usr/bin/env python3
"""Verify that the delivery is complete, relocatable and internally consistent."""

from __future__ import print_function

import argparse
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED = {
    "weights/drivetransformer_large.pth": (
        6237130912,
        "982f2024cb17f59541d0333a1b48ad74267ee0ba91eca3abe880afc60ed8c2f5",
    ),
    "weights/iter_25000.pth": (
        6197923070,
        "49a556d650702da513cafe9d6dac7e28f01698eccb48117dde0aeb50b1bceed5",
    ),
    "codes/code/pointpillar_kitti.pth": (
        19383576,
        "4c83fc0fa02575b9b3e9dec676f698e7a70bb5a795e89f91df8a96b916fa19e2",
    ),
}


def digest(path):
    value = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def route_count(relative):
    root = ET.parse(os.path.join(ROOT, relative)).getroot()
    routes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "route"]
    identifiers = [str(node.attrib.get("id", "")).strip() for node in routes]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("missing or duplicate route id in " + relative)
    return len(routes)


def scan_host_paths():
    # Construct these markers so this verifier does not itself embed a
    # machine-specific absolute path literal.
    sep = os.sep
    markers = (
        sep + "home" + sep + "carla" + sep,
        sep + "opt" + sep + "CARLA",
        sep + "workspace" + sep,
    )
    suffixes = (".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".ini")
    findings = []
    for parent, dirs, files in os.walk(ROOT):
        dirs[:] = [name for name in dirs if name not in (".git", "results")]
        for name in files:
            if not name.endswith(suffixes):
                continue
            path = os.path.join(parent, name)
            try:
                text = open(path, "r", encoding="utf-8").read()
            except (OSError, UnicodeError):
                continue
            if any(marker in text for marker in markers):
                findings.append(os.path.relpath(path, ROOT))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="skip checkpoint content hashes")
    args = parser.parse_args()
    failures = []

    required = (
        "codes/run_all_routes.sh",
        "codes/code/ordinary_online_model_agent.py",
        "codes/code/radar_model_switch_agent.py",
        "codes/code/scene_runtime_dispatch.py",
        "codes/scene_variants/extreme_144/cvci_auxiliary_system.py",
        "ordinary_rules/router.py",
        "ordinary_rules/class_01.py",
        "ordinary_rules/class_44.py",
        "harness/ordinary_classifier.py",
        "harness/radar_features.py",
        "data/classifier_model.json",
    )
    for relative in required:
        if not os.path.isfile(os.path.join(ROOT, relative)):
            failures.append("missing " + relative)

    try:
        if route_count("route/bench2drive220.xml") != 220:
            failures.append("ordinary route count is not 220")
        if route_count("route/CVCI_BenchMark.xml") != 144:
            failures.append("extreme route count is not 144")
    except Exception as exc:
        failures.append("route audit failed: " + repr(exc))

    for relative, (size, expected_hash) in EXPECTED.items():
        path = os.path.join(ROOT, relative)
        if not os.path.isfile(path):
            failures.append("missing " + relative)
            continue
        if os.path.islink(path):
            failures.append("checkpoint must not be a symlink: " + relative)
        if os.path.getsize(path) != size:
            failures.append("wrong size for " + relative)
        if not args.quick:
            if digest(path) != expected_hash:
                failures.append("wrong SHA-256 for " + relative)

    try:
        model = json.load(open(os.path.join(ROOT, "data/classifier_model.json"), encoding="utf-8"))
        if not model.get("feature_names") or not model.get("semantic_signatures"):
            failures.append("classifier model is incomplete")
    except Exception as exc:
        failures.append("classifier model unreadable: " + repr(exc))

    findings = scan_host_paths()
    if findings:
        failures.append("host-specific paths found in: " + ", ".join(findings))

    if failures:
        for failure in failures:
            print("FAIL:", failure, file=sys.stderr)
        return 1
    print("package verification passed")
    print("root=" + ROOT)
    print("ordinary_routes=220 extreme_routes=144 checkpoints=3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
