"""Select the rule-stack implementation after route-scene classification.

The model is shared by both scene families, but the post-model control stack
is intentionally versioned.  Extreme routes use the frozen 0725 modules;
ordinary routes use the 0810/0629 auxiliary baseline without 0824 additions.
"""

from __future__ import absolute_import

import hashlib
import importlib
import os
import sys


_EXTREME = "extreme_144"
_ORDINARY = "ordinary_220"


# This is the recorded 0725 final-144 runtime profile.  Keep it in the
# dispatch module as a second line of defence: the route launcher applies the
# same profile before CARLA/evaluator setup, while this function reapplies the
# values at the exact point where the frozen auxiliary stack is constructed.
# The latter matters for settings which are read by dataclass defaults during
# CVCIAuxiliarySystem() construction.
EXTREME_144_PROFILE = {
    "CVCI_AGENT_DEVICE": "cuda",
    "CVCI_ALLOW_ROUTE_PRIOR": "0",
    "CVCI_AUXILIARY_SYSTEM_ENABLED": "1",
    "CVCI_AUXILIARY_PERCEPTION_ENABLED": "1",
    "CVCI_AUX_LOG_PERIOD": "10",
    "CVCI_AUX_MAX_LATENCY_MS": "500",
    "CVCI_CARLA_EXTRA_ARGS": "-quality-level=Low -ResX=640 -ResY=360",
    "CVCI_CARLA_PID_REQUIRED": "0",
    "CVCI_CARLA_PROACTIVE_RESTART_EVERY_WORLD_LOADS": "4",
    "CVCI_CARLA_ROUTE_LOAD_RETRIES": "2",
    "CVCI_CARLA_ROUTE_LOAD_TIMEOUT": "90",
    "CVCI_CARLA_ROUTE_RUN_RETRIES": "0",
    "CVCI_CARLA_STANDARD_MAP_EXTRA_ARGS": "-quality-level=Epic -ResX=640 -ResY=360",
    "CVCI_CARLA_BOOT_WAIT": "30",
    "CVCI_EXACT_HISTORICAL_DEDICATED_CONTROLLERS": "1",
    "CVCI_FORCE_MACRO_SCENARIO": "",
    "CVCI_FORMAL_SCORE_FLOOR": "75",
    "CVCI_GEOMETRY_LIDAR_SAFETY_ONLY": "1",
    "CVCI_HARD_EXIT_ON_SIM_CRASH": "1",
    "CVCI_HISTORICAL_RESTORATION_DIRECT_FULL": "1",
    "CVCI_LEGACY_DETECTION_RULES_ENABLED": "0",
    "CVCI_LIDAR_DETECTOR_ASYNC": "0",
    "CVCI_LIDAR_DETECTOR_ENABLED": "1",
    "CVCI_LIDAR_DETECTOR_INTERVAL_SEC": "0.5",
    "CVCI_LIDAR_DETECTOR_REQUIRED": "1",
    "CVCI_LIDAR_ENABLED": "1",
    "CVCI_MACRO_LOCK_CONFIRM_FRAMES": "1",
    "CVCI_MACRO_LOCK_WARMUP_FRAMES": "1",
    "CVCI_MACRO_STEERING_MODES_JSON": (
        '{"avoid_a_disabled_vehicle":"carla_public_pid",'
        '"blind_spot_hidden_car":"carla_public_pid",'
        '"crazy_motor":"carla_public_pid",'
        '"four_students_crossing_the_road":"carla_public_pid",'
        '"high_speed_reckless_lane_cutting":"direct_p_fallback",'
        '"high_speed_temporary_construction":"direct_p_fallback",'
        '"highway_accident_vehicle":"direct_p_fallback",'
        '"missing_car":"direct_p_fallback",'
        '"reverse_vehicle":"carla_public_pid",'
        '"roundabout":"direct_p_fallback",'
        '"slanted_motor_and_children":"carla_public_pid",'
        '"trucks_encountered_during_construction":"direct_p_fallback"}'
    ),
    "CVCI_MODEL_INFERENCE_INTERVAL": "16",
    "CVCI_RECORD_FAILURE_MP4": "0",
    "CVCI_STAGE": "8",
    "CVCI_STEERING_MODE": "carla_public_pid",
}

# These legacy ordinary-only flags must not leak into a direct 0725 auxiliary
# construction when the runtime is invoked outside the canonical shell
# launcher.  The restored 0810 ordinary auxiliary no longer reads them.
_ORDINARY_ONLY_ENV = (
    "CVCI_ORDINARY_SCENE_RULES_ENABLED",
    "CVCI_STANDARD_FALLBACK_ENABLED",
    "CVCI_STANDARD_ONLY_MODE",
    "CVCI_STANDARD_SPECIAL_CONFIDENCE_THRESHOLD",
)


def apply_extreme_144_profile(agent=None):
    """Apply the proven 0725 environment before constructing the 144 stack."""

    for key in _ORDINARY_ONLY_ENV:
        os.environ.pop(key, None)
    os.environ.update(EXTREME_144_PROFILE)
    package_root = _package_code_root()
    package_code_root = os.path.join(package_root, "code")
    # Keep the detector runtime self-contained after the package is renamed or
    # moved.  Explicit audited overrides still win, but an empty inherited
    # value must not mask the package-local dependency.
    if not os.environ.get("CVCI_OPENPCDET_ROOT", "").strip():
        os.environ["CVCI_OPENPCDET_ROOT"] = os.path.join(
            package_code_root, "OpenPCDet"
        )
    if not os.environ.get("CVCI_LIDAR_DETECTOR_CONFIG", "").strip():
        os.environ["CVCI_LIDAR_DETECTOR_CONFIG"] = os.path.join(
            package_code_root, "pointpillar.yaml"
        )
    # The frozen extreme stack requires PointPillar.  Keep the default
    # checkpoint relocatable with the package while allowing an explicit
    # externally audited override.
    if not os.environ.get("CVCI_LIDAR_DETECTOR_MODEL", "").strip():
        os.environ["CVCI_LIDAR_DETECTOR_MODEL"] = os.path.join(
            package_code_root, "pointpillar_kitti.pth"
        )
    # DriveTransformerAgent reads this once during setup, before the first
    # radar frame.  The launcher already sets it early; patch it as well so a
    # direct evaluator invocation cannot retain the ordinary value of 2.
    if agent is not None and hasattr(agent, "model_inference_interval"):
        agent.model_inference_interval = int(
            EXTREME_144_PROFILE["CVCI_MODEL_INFERENCE_INTERVAL"]
        )


def _package_code_root():
    # This helper lives in ``code/code`` beside the evaluator-facing agent;
    # the packaged DriveTransformer and scene_variants directories live one
    # level above it.
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_record(relative_path):
    path = os.path.join(_package_code_root(), relative_path)
    if not os.path.isfile(path):
        raise RuntimeError("runtime code file is missing: {}".format(path))
    return {
        "path": os.path.abspath(path),
        "sha256": _sha256(path),
    }


def runtime_variant_manifest(decision):
    """Return the selected rule-stack name and auditable source hashes."""

    if decision == _EXTREME:
        relative_files = [
            "scene_variants/extreme_144/cvci_auxiliary_system.py",
            "scene_variants/extreme_144/cvci_historical_dedicated_dispatch.py",
            "historical_controllers/state/historical_source_reconstruction_verification.json",
        ]
        return {
            "name": "0725_extreme_144",
            "source_root": "copied_0725_submission_DriveTransformer",
            "environment_profile": dict(EXTREME_144_PROFILE),
            "files": [_source_record(path) for path in relative_files],
        }
    if decision == _ORDINARY:
        relative_files = [
            "DriveTransformer/team_code/cvci_auxiliary_system.py",
            "DriveTransformer/team_code/ordinary_scene_rules.py",
            "DriveTransformer/team_code/cvci_historical_dedicated_dispatch.py",
        ]
        return {
            "name": "0810_ordinary_220",
            "source_root": "0810_0629_DriveTransformer",
            "files": [_source_record(path) for path in relative_files],
        }
    raise RuntimeError("unknown route-scene decision: {}".format(decision))


def activate(agent, decision):
    """Bind the selected post-model auxiliary implementation to ``agent``."""

    manifest = runtime_variant_manifest(decision)
    if decision == _EXTREME:
        # Apply the profile before importing/constructing any frozen module.
        # The route launcher does this earlier for CARLA renderer and evaluator
        # settings; this call covers direct evaluator use and dataclass fields
        # that are sampled at object construction time.
        apply_extreme_144_profile(agent=agent)
        package_root = _package_code_root()
        if package_root not in sys.path:
            sys.path.insert(0, package_root)
        os.environ.setdefault(
            "CVCI_HISTORICAL_ASSETS_ROOT",
            os.path.join(package_root, "historical_controllers"),
        )
        # The copied 0725 auxiliary source imports the historical dispatcher
        # through the legacy DriveTransformer module name.  Bind that name to
        # the colocated 0725 dispatcher before the auxiliary object is made;
        # otherwise the ordinary 0819 dispatcher would be selected simply
        # because both branches share the same package namespace.
        historical_dispatch = importlib.import_module(
            "scene_variants.extreme_144.cvci_historical_dedicated_dispatch"
        )
        sys.modules[
            "DriveTransformer.team_code.cvci_historical_dedicated_dispatch"
        ] = historical_dispatch
        module = importlib.import_module(
            "scene_variants.extreme_144.cvci_auxiliary_system"
        )
        agent.cvci_auxiliary_system = module.CVCIAuxiliarySystem()
        # Keep the selected profile visible in the route gate record.  This is
        # deliberately a value copy so later environment changes cannot alter
        # the evidence for the already-created controller.
        agent._scene_gate_profile = dict(EXTREME_144_PROFILE)
    elif decision == _ORDINARY:
        # Construct the 44-class adapter only after the existing family gate
        # has selected ordinary_220.  It is absent from the frozen 144 object
        # graph and receives no route metadata.
        if os.environ.get("CVCI_ORDINARY_44_RULES_ENABLED", "1").strip().lower() in (
            "1", "true", "yes", "on"
        ):
            from ordinary_rules.router import OrdinaryRuleRouter

            router = OrdinaryRuleRouter(
                model_path=os.environ.get("CVCI_ORDINARY_CLASSIFIER_MODEL", ""),
                record_path=os.environ.get("CVCI_RADAR_RECORD_PATH", ""),
                route_pid_steer=agent.cvci_auxiliary_system._carla_route_pid_steer,
            )
            agent._ordinary_44_router = router
            agent.cvci_auxiliary_system._ordinary_44_router = router
    # The ordinary branch deliberately retains the object constructed by the
    # packaged DriveTransformer agent during setup; its auxiliary modules are
    # the 0810-compatible files recorded in the manifest above.
    agent._runtime_code_variant = manifest["name"]
    return manifest
