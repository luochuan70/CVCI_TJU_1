import os

from scene_runtime_dispatch import (
    EXTREME_144_PROFILE,
    apply_extreme_144_profile,
    runtime_variant_manifest,
)


def test_extreme_144_uses_copied_0725_rule_stack():
    manifest = runtime_variant_manifest("extreme_144")
    assert manifest["name"] == "0725_extreme_144"
    assert manifest["source_root"] == "copied_0725_submission_DriveTransformer"
    assert [item["path"].split("/scene_variants/", 1)[-1] for item in manifest["files"][:2]] == [
        "extreme_144/cvci_auxiliary_system.py",
        "extreme_144/cvci_historical_dedicated_dispatch.py",
    ]
    assert manifest["files"][2]["path"].endswith(
        "historical_controllers/state/historical_source_reconstruction_verification.json"
    )
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_ordinary_220_uses_0810_auxiliary_baseline():
    manifest = runtime_variant_manifest("ordinary_220")
    assert manifest["name"] == "0810_ordinary_220"
    assert manifest["source_root"] == "0810_0629_DriveTransformer"
    assert [item["path"].split("/codes/", 1)[-1] for item in manifest["files"]] == [
        "DriveTransformer/team_code/cvci_auxiliary_system.py",
        "DriveTransformer/team_code/ordinary_scene_rules.py",
        "DriveTransformer/team_code/cvci_historical_dedicated_dispatch.py",
    ]
    assert all(len(item["sha256"]) == 64 for item in manifest["files"])


def test_extreme_profile_replaces_ordinary_runtime_flags(monkeypatch):
    ordinary_only = (
        "CVCI_ORDINARY_SCENE_RULES_ENABLED",
        "CVCI_STANDARD_FALLBACK_ENABLED",
        "CVCI_STANDARD_ONLY_MODE",
        "CVCI_STANDARD_SPECIAL_CONFIDENCE_THRESHOLD",
    )
    for key in ordinary_only:
        monkeypatch.setenv(key, "1")

    class Agent(object):
        model_inference_interval = 2

    agent = Agent()
    apply_extreme_144_profile(agent)

    assert agent.model_inference_interval == 16
    assert all(
        os.environ.get(key) == value
        for key, value in EXTREME_144_PROFILE.items()
    )
    assert all(key not in os.environ for key in ordinary_only)
