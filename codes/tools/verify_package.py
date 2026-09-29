#!/usr/bin/env python3
"""Verify that the delivery is complete, relocatable and internally consistent."""

from __future__ import print_function

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET


PACKAGE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def route_count(relative):
    root = ET.parse(os.path.join(PACKAGE_ROOT, relative)).getroot()
    routes = [node for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "route"]
    identifiers = [str(node.attrib.get("id", "")).strip() for node in routes]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise RuntimeError("missing or duplicate route id in " + relative)
    return len(routes)


def result_count(route_set):
    root = os.path.join(PACKAGE_ROOT, "results", route_set, "routes")
    paths = []
    if os.path.isdir(root):
        for name in os.listdir(root):
            path = os.path.join(root, name, "result.json")
            if os.path.isfile(path):
                paths.append(path)
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            payload = json.load(handle)
        records = (payload.get("_checkpoint") or {}).get("records") or []
        if not records:
            raise RuntimeError("result has no evaluator record: " + path)
    return len(paths)


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
    for parent, dirs, files in os.walk(PACKAGE_ROOT):
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
                findings.append(os.path.relpath(path, PACKAGE_ROOT))
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    failures = []

    required = (
        "codes/run_all_routes.sh",
        "codes/code/ordinary_online_model_agent.py",
        "codes/code/radar_model_switch_agent.py",
        "codes/code/scene_runtime_dispatch.py",
        "codes/scene_variants/extreme_144/cvci_auxiliary_system.py",
        "codes/ordinary_rules/router.py",
        "codes/ordinary_rules/class_01.py",
        "codes/ordinary_rules/class_44.py",
        "codes/harness/ordinary_classifier.py",
        "codes/harness/radar_features.py",
        "codes/data/classifier_model.json",
        "weights/drivetransformer_large.pth",
        "weights/iter_25000.pth",
        "codes/code/pointpillar_kitti.pth",
    )
    for relative in required:
        path = os.path.join(PACKAGE_ROOT, relative)
        if not os.path.isfile(path):
            failures.append("missing " + relative)
        elif relative.endswith(".pth") and os.path.islink(path):
            failures.append("checkpoint must not be a symlink: " + relative)

    try:
        if route_count("route/bench2drive220.xml") != 220:
            failures.append("ordinary route count is not 220")
        if route_count("route/CVCI_BenchMark.xml") != 144:
            failures.append("extreme route count is not 144")
    except Exception as exc:
        failures.append("route audit failed: " + repr(exc))

    try:
        if result_count("220") != 220:
            failures.append("ordinary result count is not 220")
        if result_count("144") != 144:
            failures.append("extreme result count is not 144")
    except Exception as exc:
        failures.append("result audit failed: " + repr(exc))

    try:
        model_path = os.path.join(PACKAGE_ROOT, "codes/data/classifier_model.json")
        with open(model_path, encoding="utf-8") as handle:
            model = json.load(handle)
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
    print("root=" + PACKAGE_ROOT)
    print("ordinary_routes=220 extreme_routes=144 ordinary_results=220 extreme_results=144 checkpoints=3")
    return 0


if __name__ == "__main__":
    sys.exit(main())
