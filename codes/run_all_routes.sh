#!/usr/bin/env bash
set -euo pipefail

# Portable hybrid run.  The route evaluator is started once per route so a native
# CARLA failure or an agent OOM cannot silently prevent later XML routes from
# being attempted.  The historical name is retained for compatibility; the
# manifest may contain a complete, partial, single-route, or mixed route set.
# The agent selects the checkpoint and the corresponding rule-stack version
# separately for each route: copied 0725 code for extreme_144, and the
# 0810/0629 ordinary auxiliary baseline for ordinary_220.

CODES_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${CODES_ROOT}/.." && pwd)"
# Full-set evidence is kept beside the package's results directory.  A fully
# explicit CVCI_OUTPUT_ROOT remains available for resumptions or experiments.
RESULTS_ROOT="${PACKAGE_ROOT}/results"
OUT=""
DT_ROOT="${CODES_ROOT}/DriveTransformer"
CVCI_ROOT="${CODES_ROOT}/CVCI_BenchMark"
ROUTE_DIR="${PACKAGE_ROOT}/route"

make_absolute() {
    case "$1" in
        /*) printf '%s\n' "$1" ;;
        *) printf '%s/%s\n' "$(pwd)" "$1" ;;
    esac
}

ROUTE_SET="${CVCI_ROUTE_SET:-220}"
case "${ROUTE_SET}" in
    144|extreme_144)
        DEFAULT_ROUTES_FILE="${ROUTE_DIR}/CVCI_BenchMark.xml"
        ;;
    220|ordinary_220)
        DEFAULT_ROUTES_FILE="${ROUTE_DIR}/bench2drive220.xml"
        ;;
    *)
        echo "CVCI_ROUTE_SET must be 144/extreme_144 or 220/ordinary_220: ${ROUTE_SET}" >&2
        exit 2
        ;;
esac
if [[ -n "${CVCI_ROUTES_FILE:-}" ]]; then
    ROUTES_FILE="$(make_absolute "${CVCI_ROUTES_FILE}")"
elif [[ -n "${CVCI_220_ROUTES:-}" && -z "${CVCI_ROUTE_SET:-}" ]]; then
    # Preserve the historical 220 override for existing callers.
    ROUTES_FILE="$(make_absolute "${CVCI_220_ROUTES}")"
else
    ROUTES_FILE="${DEFAULT_ROUTES_FILE}"
fi
SCENE_REFERENCE_DIR="${CVCI_SCENE_REFERENCE_DIR:-${CODES_ROOT}/code/.route_catalog}"
SCENE_REFERENCE_DIR="$(make_absolute "${SCENE_REFERENCE_DIR}")"
EXTREME_REFERENCE_XML="${CVCI_EXTREME_REFERENCE_XML:-${SCENE_REFERENCE_DIR}/manifest_a.xml}"
EXTREME_REFERENCE_XML="$(make_absolute "${EXTREME_REFERENCE_XML}")"
ORDINARY_REFERENCE_XML="${CVCI_ORDINARY_REFERENCE_XML:-${SCENE_REFERENCE_DIR}/manifest_b.xml}"
ORDINARY_REFERENCE_XML="$(make_absolute "${ORDINARY_REFERENCE_XML}")"

# Python and CARLA remain host/environment dependencies.  The model stack,
# including the required MMCV source and compiled extensions, is packaged
# under this directory.  A PATH-visible Python is accepted and resolved once
# so later pushd calls cannot change its meaning.
PYTHON="${PYTHON:-python3}"
if [[ "${PYTHON}" != /* ]]; then
    if [[ "${PYTHON}" == */* ]]; then
        PYTHON="$(make_absolute "${PYTHON}")"
    else
        PYTHON="$(command -v "${PYTHON}" || true)"
    fi
fi
CARLA_ROOT="${CARLA_ROOT:-}"
PACKAGE_OPENPCDET_ROOT="${CODES_ROOT}/code/OpenPCDet"
PACKAGE_LIDAR_DETECTOR_CONFIG="${CODES_ROOT}/code/pointpillar.yaml"
OPENPCDET_ROOT="${CVCI_OPENPCDET_ROOT:-${OPENPCDET_ROOT:-${PACKAGE_OPENPCDET_ROOT}}}"
LIDAR_DETECTOR_CONFIG="${CVCI_LIDAR_DETECTOR_CONFIG:-${PACKAGE_LIDAR_DETECTOR_CONFIG}}"
# The 0725 extreme profile requires PointPillar.  Keep the checkpoint inside
# the relocatable package by default; an explicitly audited external override
# remains supported through CVCI_LIDAR_DETECTOR_MODEL.
LIDAR_DETECTOR_MODEL="${CVCI_LIDAR_DETECTOR_MODEL:-${CODES_ROOT}/code/pointpillar_kitti.pth}"
ROUTE_MACRO_TABLE="${CVCI_ROUTE_MACRO_TABLE:-}"
if [[ -n "${CARLA_ROOT}" && "${CARLA_ROOT}" != /* ]]; then
    CARLA_ROOT="$(make_absolute "${CARLA_ROOT}")"
fi
if [[ -n "${OPENPCDET_ROOT}" && "${OPENPCDET_ROOT}" != /* ]]; then
    OPENPCDET_ROOT="$(make_absolute "${OPENPCDET_ROOT}")"
fi
if [[ -n "${LIDAR_DETECTOR_CONFIG}" && "${LIDAR_DETECTOR_CONFIG}" != /* ]]; then
    LIDAR_DETECTOR_CONFIG="$(make_absolute "${LIDAR_DETECTOR_CONFIG}")"
fi
if [[ -n "${LIDAR_DETECTOR_MODEL}" && "${LIDAR_DETECTOR_MODEL}" != /* ]]; then
    LIDAR_DETECTOR_MODEL="$(make_absolute "${LIDAR_DETECTOR_MODEL}")"
fi
if [[ -n "${ROUTE_MACRO_TABLE}" && "${ROUTE_MACRO_TABLE}" != /* ]]; then
    ROUTE_MACRO_TABLE="$(make_absolute "${ROUTE_MACRO_TABLE}")"
fi

if [[ "${ROUTE_SET}" == "220" || "${ROUTE_SET}" == "ordinary_220" ]]; then
    AGENT="${CODES_ROOT}/code/ordinary_online_model_agent.py"
else
    AGENT="${CODES_ROOT}/code/radar_model_switch_agent.py"
fi
CONFIG="${DT_ROOT}/adzoo/drivetransformer/configs/drivetransformer/drivetransformer_large.py"
EXTREME_CHECKPOINT="${CVCI_EXTREME_CHECKPOINT:-${PACKAGE_ROOT}/weights/iter_25000.pth}"
ORDINARY_CHECKPOINT="${CVCI_ORDINARY_CHECKPOINT:-${PACKAGE_ROOT}/weights/drivetransformer_large.pth}"
ORDINARY_CLASSIFIER_MODEL="${CVCI_ORDINARY_CLASSIFIER_MODEL:-${PACKAGE_ROOT}/data/classifier_model.json}"
if [[ "${EXTREME_CHECKPOINT}" != /* ]]; then
    EXTREME_CHECKPOINT="$(make_absolute "${EXTREME_CHECKPOINT}")"
fi
if [[ "${ORDINARY_CHECKPOINT}" != /* ]]; then
    ORDINARY_CHECKPOINT="$(make_absolute "${ORDINARY_CHECKPOINT}")"
fi
if [[ "${ORDINARY_CLASSIFIER_MODEL}" != /* ]]; then
    ORDINARY_CLASSIFIER_MODEL="$(make_absolute "${ORDINARY_CLASSIFIER_MODEL}")"
fi
PORT="${1:-2100}"
TM_PORT="${2:-8100}"
ROUTE_TIMEOUT_SEC="${CVCI_FULL_ROUTE_TIMEOUT_SEC:-3600}"
CLIENT_TIMEOUT_SEC="${CVCI_CLIENT_TIMEOUT_SEC:-300}"
SCENARIO_TIMEOUT_SEC="${CVCI_SCENARIO_TIMEOUT_SEC:-300}"
AGENT_TIMEOUT_SEC="${CVCI_AGENT_TIMEOUT_SEC:-120}"
MAX_ATTEMPTS="${CVCI_MAX_ATTEMPTS:-2}"
MAX_NEW_ATTEMPTS="${CVCI_MAX_NEW_ATTEMPTS:-0}"
SKIP_ROUTES_CSV="${CVCI_SKIP_ROUTES:-}"

export CODES_ROOT PACKAGE_ROOT OUT ROUTE_DIR ROUTE_SET DT_ROOT CVCI_ROOT ROUTES_FILE PYTHON CARLA_ROOT
export AGENT CONFIG EXTREME_CHECKPOINT ORDINARY_CHECKPOINT PORT TM_PORT
export CLIENT_TIMEOUT_SEC SCENARIO_TIMEOUT_SEC AGENT_TIMEOUT_SEC MAX_ATTEMPTS MAX_NEW_ATTEMPTS
export SCENARIO_RUNNER_ROOT="${CVCI_ROOT}/scenario_runner"
export LEADERBOARD_ROOT="${CVCI_ROOT}/leaderboard"
export CVCI_HISTORICAL_ASSETS_ROOT="${CVCI_HISTORICAL_ASSETS_ROOT:-${CODES_ROOT}/historical_controllers}"
if [[ "${CVCI_HISTORICAL_ASSETS_ROOT}" != /* ]]; then
    export CVCI_HISTORICAL_ASSETS_ROOT="$(make_absolute "${CVCI_HISTORICAL_ASSETS_ROOT}")"
fi
export CVCI_EXTREME_CHECKPOINT="${EXTREME_CHECKPOINT}"
export CVCI_ORDINARY_CHECKPOINT="${ORDINARY_CHECKPOINT}"
export CVCI_ORDINARY_CLASSIFIER_MODEL="${ORDINARY_CLASSIFIER_MODEL}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CVCI_AGENT_DEVICE="${CVCI_AGENT_DEVICE:-cuda}"
export CVCI_SCENE_REFERENCE_DIR="${SCENE_REFERENCE_DIR}"
export CVCI_EXTREME_REFERENCE_XML="${EXTREME_REFERENCE_XML}"
export CVCI_ORDINARY_REFERENCE_XML="${ORDINARY_REFERENCE_XML}"
export IS_BENCH2DRIVE=True
export DISABLE_BEV_SENSOR=1
export CHALLENGE_TRACK_CODENAME=SENSORS
export CVCI_AUXILIARY_SYSTEM_ENABLED=1
export CVCI_AUXILIARY_PERCEPTION_ENABLED=1
export CVCI_LIDAR_ENABLED=1
export CVCI_LIDAR_DETECTOR_ENABLED="${CVCI_LIDAR_DETECTOR_ENABLED:-0}"
export CVCI_LIDAR_DETECTOR_REQUIRED="${CVCI_LIDAR_DETECTOR_REQUIRED:-0}"
# The ordinary profile matches the 0810 launcher.  The 0824 ordinary-scene
# additions are not loaded by the restored auxiliary baseline.
export CVCI_GEOMETRY_LIDAR_SAFETY_ONLY=1
export CVCI_STANDARD_FALLBACK_ENABLED=1
export CVCI_STANDARD_ONLY_MODE=1
export CVCI_STANDARD_SPECIAL_CONFIDENCE_THRESHOLD=0.85
export CVCI_EXACT_HISTORICAL_DEDICATED_CONTROLLERS=0
export CVCI_MODEL_INFERENCE_INTERVAL="${CVCI_MODEL_INFERENCE_INTERVAL:-2}"
export CVCI_ALLOW_ROUTE_PRIOR=0
export CVCI_LEGACY_DETECTION_RULES_ENABLED=0
export CVCI_STEERING_MODE=carla_public_pid
export CVCI_AUX_LOG_PERIOD="${CVCI_AUX_LOG_PERIOD:-100}"
export CVCI_RADAR_SCENE_GATE_MODE="${CVCI_RADAR_SCENE_GATE_MODE:-hybrid_manifest_after_radar}"
export CVCI_SCENE_GATE_REQUIRE_RADAR=1
# Ordinary 220 uses Epic rendering by default.  A controlled route-24759 A/B
# showed that the prior Low default repeatedly crashed CARLA before the first
# radar frame, while Epic completed the official route.  The frozen 144
# profile below still supplies its own explicit renderer values per route.
export CVCI_CARLA_EXTRA_ARGS="${CVCI_CARLA_EXTRA_ARGS:--quality-level=Epic -ResX=640 -ResY=360}"
export CVCI_CARLA_STANDARD_MAP_EXTRA_ARGS="${CVCI_CARLA_STANDARD_MAP_EXTRA_ARGS:--quality-level=Epic -ResX=640 -ResY=360}"
export CVCI_CARLA_ROUTE_LOAD_RETRIES="${CVCI_CARLA_ROUTE_LOAD_RETRIES:-1}"
export CVCI_CARLA_ROUTE_RUN_RETRIES=0
export CVCI_CARLA_BOOT_WAIT="${CVCI_CARLA_BOOT_WAIT:-20}"
export CVCI_OPENPCDET_ROOT="${OPENPCDET_ROOT}"
export CVCI_LIDAR_DETECTOR_CONFIG="${LIDAR_DETECTOR_CONFIG}"
export CVCI_LIDAR_DETECTOR_MODEL="${LIDAR_DETECTOR_MODEL}"
export CVCI_ROUTE_MACRO_TABLE="${ROUTE_MACRO_TABLE}"

# Per-route 0725 profile.  The shell launcher must apply this before the
# evaluator starts because CARLA renderer/recovery settings and the parent's
# model-inference cadence are consumed before the first RADAR packet.  The
# runtime dispatcher reapplies the same values when it constructs the frozen
# 144 auxiliary stack.
EXTREME_144_ENV=(
    "CVCI_AGENT_DEVICE=cuda"
    "CVCI_ALLOW_ROUTE_PRIOR=0"
    "CVCI_AUXILIARY_SYSTEM_ENABLED=1"
    "CVCI_AUXILIARY_PERCEPTION_ENABLED=1"
    "CVCI_AUX_LOG_PERIOD=10"
    "CVCI_AUX_MAX_LATENCY_MS=500"
    "CVCI_CARLA_EXTRA_ARGS=-quality-level=Low -ResX=640 -ResY=360"
    "CVCI_CARLA_PID_REQUIRED=0"
    "CVCI_CARLA_PROACTIVE_RESTART_EVERY_WORLD_LOADS=4"
    "CVCI_CARLA_ROUTE_LOAD_RETRIES=2"
    "CVCI_CARLA_ROUTE_LOAD_TIMEOUT=90"
    "CVCI_CARLA_ROUTE_RUN_RETRIES=0"
    "CVCI_CARLA_STANDARD_MAP_EXTRA_ARGS=-quality-level=Epic -ResX=640 -ResY=360"
    "CVCI_CARLA_BOOT_WAIT=30"
    "CVCI_EXACT_HISTORICAL_DEDICATED_CONTROLLERS=1"
    "CVCI_FORCE_MACRO_SCENARIO="
    "CVCI_FORMAL_SCORE_FLOOR=75"
    "CVCI_GEOMETRY_LIDAR_SAFETY_ONLY=1"
    "CVCI_HARD_EXIT_ON_SIM_CRASH=1"
    "CVCI_HISTORICAL_RESTORATION_DIRECT_FULL=1"
    "CVCI_LEGACY_DETECTION_RULES_ENABLED=0"
    "CVCI_LIDAR_DETECTOR_ASYNC=0"
    "CVCI_LIDAR_DETECTOR_CONFIG=${LIDAR_DETECTOR_CONFIG}"
    "CVCI_LIDAR_DETECTOR_ENABLED=1"
    "CVCI_LIDAR_DETECTOR_INTERVAL_SEC=0.5"
    "CVCI_LIDAR_DETECTOR_MODEL=${LIDAR_DETECTOR_MODEL}"
    "CVCI_LIDAR_DETECTOR_REQUIRED=1"
    "CVCI_LIDAR_ENABLED=1"
    "CVCI_MACRO_LOCK_CONFIRM_FRAMES=1"
    "CVCI_MACRO_LOCK_WARMUP_FRAMES=1"
    'CVCI_MACRO_STEERING_MODES_JSON={"avoid_a_disabled_vehicle":"carla_public_pid","blind_spot_hidden_car":"carla_public_pid","crazy_motor":"carla_public_pid","four_students_crossing_the_road":"carla_public_pid","high_speed_reckless_lane_cutting":"direct_p_fallback","high_speed_temporary_construction":"direct_p_fallback","highway_accident_vehicle":"direct_p_fallback","missing_car":"direct_p_fallback","reverse_vehicle":"carla_public_pid","roundabout":"direct_p_fallback","slanted_motor_and_children":"carla_public_pid","trucks_encountered_during_construction":"direct_p_fallback"}'
    "CVCI_MODEL_INFERENCE_INTERVAL=16"
    "CVCI_OPENPCDET_ROOT=${OPENPCDET_ROOT}"
    "CVCI_RECORD_FAILURE_MP4=0"
    "CVCI_ROUTE_MACRO_TABLE=${ROUTE_MACRO_TABLE}"
    "CVCI_STAGE=8"
    "CVCI_STEERING_MODE=carla_public_pid"
)
EXTREME_144_ENV_UNSET=(
    -u CVCI_ORDINARY_SCENE_RULES_ENABLED
    -u CVCI_STANDARD_FALLBACK_ENABLED
    -u CVCI_STANDARD_ONLY_MODE
    -u CVCI_STANDARD_SPECIAL_CONFIDENCE_THRESHOLD
)

# Known immutable hashes are injected into each route's gate evidence.  The
# Official checkpoint hashes are filled by the package manifest-preparation
# step before the run; an empty value remains explicit rather than being
# recomputed 220 times.
if [[ -f "${CODES_ROOT}/provenance/checkpoint_hashes.env" ]]; then
    # shellcheck disable=SC1091
    source "${CODES_ROOT}/provenance/checkpoint_hashes.env"
fi
: "${CVCI_EXTREME_CHECKPOINT_SHA256:=}"
: "${CVCI_ORDINARY_CHECKPOINT_SHA256:=}"
export CVCI_EXTREME_CHECKPOINT CVCI_ORDINARY_CHECKPOINT
export CVCI_EXTREME_CHECKPOINT_SHA256 CVCI_ORDINARY_CHECKPOINT_SHA256

# The package contains its own DriveTransformer/MMCV implementation.  Keep its
# paths ahead of any inherited PYTHONPATH so an unrelated host installation
# cannot shadow the submitted model stack.
if [[ -z "${PYTHON}" || ! -x "${PYTHON}" ]]; then
    echo "PYTHON must name an executable Python interpreter; set PYTHON or put python3 on PATH" >&2
    exit 2
fi
if [[ -z "${CARLA_ROOT}" ]]; then
    echo "CARLA_ROOT is required and must point to a CARLA installation" >&2
    exit 2
fi
export PYTHONPATH="${PACKAGE_ROOT}:${CODES_ROOT}/runtime_path:${CODES_ROOT}/code:${DT_ROOT}:$(dirname "${DT_ROOT}"):${CVCI_ROOT}:${CVCI_ROOT}/leaderboard:${CVCI_ROOT}/scenario_runner:${CARLA_ROOT}/PythonAPI:${CARLA_ROOT}/PythonAPI/carla:${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${OPENPCDET_ROOT}:${PYTHONPATH:-}"

test -x "${PYTHON}"
test -x "${CARLA_ROOT}/CarlaUE4.sh"
test -f "${ROUTES_FILE}"
test -f "${AGENT}"
test -f "${CODES_ROOT}/code/scene_runtime_dispatch.py"
test -f "${CODES_ROOT}/scene_variants/extreme_144/cvci_auxiliary_system.py"
test -f "${CODES_ROOT}/scene_variants/extreme_144/cvci_historical_dedicated_dispatch.py"
test -d "${SCENARIO_RUNNER_ROOT}/srunner/scenarios"
test -f "${CVCI_HISTORICAL_ASSETS_ROOT}/state/historical_source_reconstruction_verification.json"
test -f "${CONFIG}"
test -f "${EXTREME_CHECKPOINT}"
test -f "${ORDINARY_CHECKPOINT}"
test -f "${ORDINARY_CLASSIFIER_MODEL}"
test -f "${EXTREME_REFERENCE_XML}"
test -f "${ORDINARY_REFERENCE_XML}"

route_complete() {
    "${PYTHON}" - "$1" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    checkpoint = data.get("_checkpoint") or {}
    progress = checkpoint.get("progress") or []
    records = checkpoint.get("records") or []
    record = records[0] if records else {}
    status = str(record.get("status", "")).lower()
    global_record = checkpoint.get("global_record") or {}
    global_meta = global_record.get("meta") or {}
    exceptions = list(global_meta.get("exceptions") or [])
    route_facts = []
    for item in list(global_meta.get("challenge_common_facts") or []):
        fact = item.get("facts") or {}
        if "route_completed" in fact:
            route_facts.append(bool(fact.get("route_completed")))
    route_completed = bool(route_facts) and all(route_facts)
    scores = record.get("scores") or {}
    print(int(
        len(progress) >= 2
        and progress[0] == progress[1] == 1
        and len(records) == 1
        and status == "completed"
        and route_completed
        and not exceptions
        and scores.get("score_composed") is not None
    ))
except Exception:
    print(0)
PY
}

result_summary() {
    "${PYTHON}" - "$1" <<'PY'
import json
import sys

try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
    records = (data.get("_checkpoint") or {}).get("records") or []
    record = records[0] if records else {}
    scores = record.get("scores") or {}
    print("{}\t{}".format(record.get("status", "NO_STATUS"), scores.get("score_composed", "")))
except Exception:
    print("UNREADABLE_RESULT\t")
PY
}

route_is_skipped() {
    local route_id="$1"
    [[ ",${SKIP_ROUTES_CSV}," == *",${route_id},"* ]]
}

select_scored_attempt() {
    local route_dir="$1"
    local selected=""
    local selected_score=""
    local candidate
    local candidate_score
    local candidate_status
    for candidate in "${route_dir}"/attempt_*/result.json; do
        [[ -f "${candidate}" ]] || continue
        IFS=$'\t' read -r candidate_status candidate_score < <(result_summary "${candidate}")
        [[ -n "${candidate_score}" ]] || continue
        if [[ -z "${selected}" ]] || awk -v a="${candidate_score}" -v b="${selected_score}" 'BEGIN { exit !(a > b) }'; then
            selected="${candidate}"
            selected_score="${candidate_score}"
        fi
    done
    printf '%s\t%s\n' "${selected}" "${selected_score}"
}

carla_binary_snapshot() {
    # Match the executable name, not a free-form command-line substring.  The
    # latter can include this shell's own cleanup command and make ownership
    # checks unsafe.
    ps -eo pid=,comm= | awk '$2 == "CarlaUE4-Linux-Shipping" {print $1}'
}

cleanup_route_processes() {
    local result_path="$1"
    local route_port="$2"
    local carla_before="${3:-}"
    local pid
    local carla_pid
    local carla_after
    local pgid
    while read -r pid; do
        [[ -n "${pid}" ]] || continue
        kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f "leaderboard_evaluator.py.*${result_path}" || true)
    # CARLA may move to the next free RPC port when a prior process has left
    # the requested port occupied.  Identify only binaries created after this
    # attempt began; never kill a pre-existing external CARLA instance.
    carla_after="$(carla_binary_snapshot || true)"
    for carla_pid in ${carla_after}; do
        case " ${carla_before} " in
            *" ${carla_pid} "*) continue ;;
        esac
        pgid="$(ps -p "${carla_pid}" -o pgid= 2>/dev/null | tr -d ' ')"
        if [[ "${pgid}" =~ ^[0-9]+$ ]] && [[ "${pgid}" -gt 1 ]]; then
            kill -TERM -- "-${pgid}" 2>/dev/null || true
        else
            kill -TERM "${carla_pid}" 2>/dev/null || true
        fi
    done
    while read -r pid; do
        [[ -n "${pid}" ]] || continue
        kill -TERM "${pid}" 2>/dev/null || true
    done < <(pgrep -f "CarlaUE4-Linux-Shipping.*-carla-rpc-port=${route_port}" || true)
    sleep 2
    while read -r pid; do
        [[ -n "${pid}" ]] || continue
        kill -KILL "${pid}" 2>/dev/null || true
    done < <(pgrep -f "leaderboard_evaluator.py.*${result_path}" || true)
    carla_after="$(carla_binary_snapshot || true)"
    for carla_pid in ${carla_after}; do
        case " ${carla_before} " in
            *" ${carla_pid} "*) continue ;;
        esac
        pgid="$(ps -p "${carla_pid}" -o pgid= 2>/dev/null | tr -d ' ')"
        if [[ "${pgid}" =~ ^[0-9]+$ ]] && [[ "${pgid}" -gt 1 ]]; then
            kill -KILL -- "-${pgid}" 2>/dev/null || true
        else
            kill -KILL "${carla_pid}" 2>/dev/null || true
        fi
    done
}

mapfile -t ROUTE_IDS < <("${PYTHON}" - "${ROUTES_FILE}" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
ids = []
for route in root.iter():
    if str(route.tag).rsplit("}", 1)[-1] == "route":
        route_id = str(route.attrib.get("id", "")).strip()
        if not route_id:
            raise SystemExit("route is missing a non-empty id")
        ids.append(route_id)
if not ids:
    raise SystemExit("route XML contains no routes")
if len(ids) != len(set(ids)):
    raise SystemExit("route XML contains duplicate route ids")
for route_id in ids:
    print(route_id)
PY
)

if [[ "${#ROUTE_IDS[@]}" -lt 1 ]]; then
    echo "expected at least one route in manifest, found ${#ROUTE_IDS[@]}" >&2
    exit 2
fi

# Classify every current XML route before CARLA starts.  This uses the same
# official route fingerprints as the live radar gate, so a single-route or
# mixed XML remains safe even when route ids overlap between the two official
# manifests.  A preflight mismatch stops the run rather than launching a
# route with the wrong renderer/controller profile.
ROUTE_CLASSIFICATION="$("${PYTHON}" - "${ROUTES_FILE}" "${SCENE_REFERENCE_DIR}" <<'PY'
import sys

from radar_scene_gate import RadarSceneGate

route_xml, reference_dir = sys.argv[1:3]
rows = RadarSceneGate(
    route_xml=route_xml,
    reference_dir=reference_dir,
).classify_routes()
for row in rows:
    print("{}\t{}".format(row["route_id"], row["decision"]))
PY
)"
declare -A ROUTE_SCENES=()
while IFS=$'\t' read -r classified_route_id classified_scene; do
    [[ -n "${classified_route_id}" ]] || continue
    if [[ "${classified_scene}" != "extreme_144" && "${classified_scene}" != "ordinary_220" ]]; then
        echo "unsupported route-scene classification: route=${classified_route_id} scene=${classified_scene}" >&2
        exit 2
    fi
    ROUTE_SCENES["${classified_route_id}"]="${classified_scene}"
done <<< "${ROUTE_CLASSIFICATION}"
if [[ "${#ROUTE_SCENES[@]}" -ne "${#ROUTE_IDS[@]}" ]]; then
    echo "route classification count mismatch: ids=${#ROUTE_IDS[@]} classified=${#ROUTE_SCENES[@]}" >&2
    exit 2
fi

if [[ -n "${CVCI_ONLY_ROUTE:-}" ]]; then
    selected=()
    for route_id in "${ROUTE_IDS[@]}"; do
        if [[ "${route_id}" == "${CVCI_ONLY_ROUTE}" ]]; then
            selected+=("${route_id}")
            break
        fi
    done
    if [[ "${#selected[@]}" -ne 1 ]]; then
        echo "CVCI_ONLY_ROUTE not found in route XML: ${CVCI_ONLY_ROUTE}" >&2
        exit 2
    fi
    ROUTE_IDS=("${selected[@]}")
fi

# A recovery can start at an official route id without rewriting the XML or
# shifting the official indices recorded in ROUTE_STATUS.tsv.  Completed
# routes before this id are left untouched; the selected route itself is
# rerun when CVCI_FORCE_ROUTE=1 is supplied.
START_ROUTE_ID="${CVCI_START_ROUTE_ID:-}"
START_ROUTE_INDEX=0
if [[ -n "${START_ROUTE_ID}" ]]; then
    START_ROUTE_INDEX=-1
    for candidate_index in "${!ROUTE_IDS[@]}"; do
        if [[ "${ROUTE_IDS[${candidate_index}]}" == "${START_ROUTE_ID}" ]]; then
            START_ROUTE_INDEX="${candidate_index}"
            break
        fi
    done
    if [[ "${START_ROUTE_INDEX}" -lt 0 ]]; then
        echo "CVCI_START_ROUTE_ID not found in route XML: ${START_ROUTE_ID}" >&2
        exit 2
    fi
fi

# Choose the default output only after route filtering and scene classification.
# This keeps a CVCI_ONLY_ROUTE invocation or any custom/mixed XML separate from
# the two canonical full-set result directories.
extreme_count=0
ordinary_count=0
for route_id in "${ROUTE_IDS[@]}"; do
    case "${ROUTE_SCENES[${route_id}]:-}" in
        extreme_144) extreme_count=$((extreme_count + 1)) ;;
        ordinary_220) ordinary_count=$((ordinary_count + 1)) ;;
    esac
done

if [[ -n "${CVCI_OUTPUT_ROOT:-}" ]]; then
    OUT="${CVCI_OUTPUT_ROOT}"
elif [[ "${#ROUTE_IDS[@]}" -eq 144 && "${extreme_count}" -eq 144 ]]; then
    OUT="${RESULTS_ROOT}/144"
elif [[ "${#ROUTE_IDS[@]}" -eq 220 && "${ordinary_count}" -eq 220 ]]; then
    OUT="${RESULTS_ROOT}/220"
else
    if [[ "${extreme_count}" -gt 0 && "${ordinary_count}" -gt 0 ]]; then
        custom_label="mixed"
    elif [[ "${extreme_count}" -gt 0 ]]; then
        custom_label="extreme_${#ROUTE_IDS[@]}"
    else
        custom_label="ordinary_${#ROUTE_IDS[@]}"
    fi
    custom_stamp="$(date '+%Y%m%d_%H%M%S')"
    OUT="${RESULTS_ROOT}/custom_${custom_stamp}_${custom_label}"
    custom_suffix=1
    while [[ -e "${OUT}" ]]; do
        OUT="${RESULTS_ROOT}/custom_${custom_stamp}_${custom_label}_${custom_suffix}"
        custom_suffix=$((custom_suffix + 1))
    done
fi
OUT="$(make_absolute "${OUT}")"

validate_extreme_dependencies() {
    local missing=()
    [[ -n "${OPENPCDET_ROOT}" && -d "${OPENPCDET_ROOT}" ]] || missing+=("CVCI_OPENPCDET_ROOT (OpenPCDet directory)")
    [[ -n "${LIDAR_DETECTOR_CONFIG}" && -f "${LIDAR_DETECTOR_CONFIG}" ]] || missing+=("CVCI_LIDAR_DETECTOR_CONFIG (PointPillar YAML)")
    [[ -n "${LIDAR_DETECTOR_MODEL}" && -f "${LIDAR_DETECTOR_MODEL}" ]] || missing+=("CVCI_LIDAR_DETECTOR_MODEL (detector checkpoint)")
    if [[ "${#missing[@]}" -gt 0 ]]; then
        echo "extreme_144 requires OpenPCDet detector dependencies:" >&2
        printf '  - %s\n' "${missing[@]}" >&2
        echo "The checkpoint defaults to ${CODES_ROOT}/code/pointpillar_kitti.pth; configure only the OpenPCDet root/config or an explicit checkpoint override as needed." >&2
        exit 2
    fi
}

for route_id in "${ROUTE_IDS[@]}"; do
    if [[ "${ROUTE_SCENES[${route_id}]:-}" == "extreme_144" ]]; then
        validate_extreme_dependencies
        break
    fi
done

if [[ "${CVCI_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    extreme_count=0
    ordinary_count=0
    for route_id in "${ROUTE_IDS[@]}"; do
        case "${ROUTE_SCENES[${route_id}]:-}" in
            extreme_144) extreme_count=$((extreme_count + 1)) ;;
            ordinary_220) ordinary_count=$((ordinary_count + 1)) ;;
        esac
    done
    echo "portable package preflight passed: routes=${#ROUTE_IDS[@]} extreme_144=${extreme_count} ordinary_220=${ordinary_count}"
    echo "routes_file=${ROUTES_FILE}"
    echo "package_root=${PACKAGE_ROOT}"
    echo "codes_root=${CODES_ROOT}"
    echo "output_root=${OUT}"
    echo "ordinary_checkpoint=${ORDINARY_CHECKPOINT}"
    echo "ordinary_classifier_model=${ORDINARY_CLASSIFIER_MODEL}"
    echo "extreme_checkpoint=${EXTREME_CHECKPOINT}"
    exit 0
fi

mkdir -p "${OUT}/routes"
STATUS_TSV="${OUT}/ROUTE_STATUS.tsv"
SUPERVISOR_LOG="${OUT}/supervisor.log"
if [[ ! -f "${STATUS_TSV}" ]]; then
    echo -e "official_index\troute_id\tstatus\tscore_composed\tevaluator_rc\tattempts\tevidence" > "${STATUS_TSV}"
fi

echo "[$(date '+%F %T%z')] CVCI hybrid run started routes=${#ROUTE_IDS[@]} xml=${ROUTES_FILE} port=${PORT} profiles=route_reference" | tee -a "${SUPERVISOR_LOG}"

for index in "${!ROUTE_IDS[@]}"; do
    if [[ "${index}" -lt "${START_ROUTE_INDEX}" ]]; then
        continue
    fi
    route_id="${ROUTE_IDS[$index]}"
    route_dir="${OUT}/routes/route_${route_id}"
    final_result="${route_dir}/result.json"
    mkdir -p "${route_dir}"
    scene_family="${ROUTE_SCENES[${route_id}]:-}"
    if [[ -z "${scene_family}" ]]; then
        echo "route ${route_id} has no preflight scene classification" >&2
        exit 2
    fi
    if route_is_skipped "${route_id}"; then
        echo "[$(date '+%F %T%z')] skip-by-policy index=${index} route=${route_id} scene=${scene_family}" | tee -a "${SUPERVISOR_LOG}"
        continue
    fi
    route_env=()
    route_env_unset=()
    resume_value=True
    if [[ "${scene_family}" == "extreme_144" ]]; then
        route_env=("${EXTREME_144_ENV[@]}")
        route_env_unset=("${EXTREME_144_ENV_UNSET[@]}")
        # The reference 0725 full-144 evaluator starts from a fresh
        # checkpoint.  Keep the existing ordinary per-route recovery behavior
        # (resume=True) unchanged for ordinary_220.
        resume_value=False
    fi

    if [[ "${CVCI_FORCE_ROUTE:-0}" != "1" ]] \
        && [[ -f "${final_result}" ]] \
        && [[ "$(route_complete "${final_result}")" == 1 ]]; then
        IFS=$'\t' read -r route_status route_score < <(result_summary "${final_result}")
        echo "[$(date '+%F %T%z')] skip index=${index} route=${route_id} scene=${scene_family} status=${route_status} score=${route_score}" | tee -a "${SUPERVISOR_LOG}"
        continue
    fi

    # Preserve prior attempts so a later recovery invocation can add fresh
    # attempts without overwriting evidence from an interrupted run.  The
    # first full invocation still behaves as before (attempt_1, then
    # attempt_2); recovery can set CVCI_MAX_ATTEMPTS to a larger value.
    attempts=0
    for existing_attempt_dir in "${route_dir}"/attempt_*; do
        [[ -d "${existing_attempt_dir}" ]] || continue
        existing_attempt_name="${existing_attempt_dir##*/}"
        existing_attempt_number="${existing_attempt_name#attempt_}"
        if [[ "${existing_attempt_number}" =~ ^[0-9]+$ ]] && (( existing_attempt_number > attempts )); then
            attempts="${existing_attempt_number}"
        fi
    done
    complete=0
    evaluator_rc=0
    route_status="UNRESOLVED"
    route_score=""
    route_attempt_limit="${MAX_ATTEMPTS}"
    if [[ "${MAX_NEW_ATTEMPTS}" -gt 0 ]]; then
        route_attempt_limit=$((attempts + MAX_NEW_ATTEMPTS))
    fi
    while [[ "${attempts}" -lt "${route_attempt_limit}" && "${complete}" != 1 ]]; do
        attempts=$((attempts + 1))
        attempt_dir="${route_dir}/attempt_${attempts}"
        result="${attempt_dir}/result.json"
        live="${attempt_dir}/live_results.txt"
        log="${attempt_dir}/stdout.log"
        mkdir -p "${attempt_dir}"
        export CVCI_ROUTE_ID="${route_id}"
        export CVCI_SCENE_GATE_ROUTE_XML="${ROUTES_FILE}"
        export CVCI_SCENE_GATE_LOG_PATH="${route_dir}/scene_gate_attempt_${attempts}.json"
        export CVCI_SCENARIO_ID_LOG_PATH="${route_dir}/scenario_identification_attempt_${attempts}.csv"
        carla_before="$(carla_binary_snapshot || true)"
        echo "[$(date '+%F %T%z')] start index=${index} route=${route_id} scene=${scene_family} attempt=${attempts}" | tee -a "${SUPERVISOR_LOG}" | tee "${log}"

        set +e
        pushd "${CVCI_ROOT}" >/dev/null
        env "${route_env_unset[@]}" "${route_env[@]}" \
        timeout --signal=TERM --kill-after=30s "${ROUTE_TIMEOUT_SEC}s" \
            "${PYTHON}" -u "${CVCI_ROOT}/leaderboard/leaderboard/leaderboard_evaluator.py" \
            --routes="${ROUTES_FILE}" \
            --routes-subset="${route_id}" \
            --repetitions=1 --track=SENSORS \
            --checkpoint="${result}" \
            --debug-checkpoint="${live}" \
            --agent="${AGENT}" \
            --agent-config="${CONFIG}+None" \
            --debug=0 --port="${PORT}" --traffic-manager-port="${TM_PORT}" \
            --gpu-rank=0 --client-timeout="${CLIENT_TIMEOUT_SEC}" \
            --scenario-timeout="${SCENARIO_TIMEOUT_SEC}" \
            --agent-timeout="${AGENT_TIMEOUT_SEC}" --resume="${resume_value}" \
            >> "${log}" 2>&1
        evaluator_rc=$?
        popd >/dev/null
        set -e
        complete=0
        if [[ -f "${result}" ]]; then
            complete="$(route_complete "${result}")"
        fi
        cleanup_route_processes "${result}" "${PORT}" "${carla_before}"
        if [[ "${complete}" == 1 ]]; then
            cp "${result}" "${final_result}"
            IFS=$'\t' read -r route_status route_score < <(result_summary "${final_result}")
            echo "[$(date '+%F %T%z')] done index=${index} route=${route_id} status=${route_status} score=${route_score} rc=${evaluator_rc} attempts=${attempts}" | tee -a "${SUPERVISOR_LOG}"
        else
            echo "[$(date '+%F %T%z')] unresolved index=${index} route=${route_id} rc=${evaluator_rc} attempt=${attempts}; retrying if available" | tee -a "${SUPERVISOR_LOG}"
        fi
        sleep 2
    done

    if [[ "${complete}" != 1 ]]; then
        evidence="${route_dir}/attempt_${attempts}/stdout.log"
        echo "[$(date '+%F %T%z')] unresolved-final index=${index} route=${route_id} attempts=${attempts}" | tee -a "${SUPERVISOR_LOG}"
        echo -e "${index}\t${route_id}\tUNRESOLVED\t\t${evaluator_rc}\t${attempts}\t${evidence}" >> "${STATUS_TSV}"
        IFS=$'\t' read -r selected_result selected_score < <(select_scored_attempt "${route_dir}")
        if [[ -n "${selected_result}" ]]; then
            cp "${selected_result}" "${route_dir}/selected_result.json"
            # A route can finish with a genuine official score without
            # satisfying the acceptance-only `route_complete` predicate
            # (for example, collision plus partial completion).  Preserve
            # that terminal failure at the canonical path so the harness can
            # parse its score and deductions instead of reporting MISSING.
            # `route_complete` will still reject it on the next invocation,
            # so this does not turn a failed route into a resumable success.
            cp "${selected_result}" "${final_result}"
            echo -e "${index}\t${route_id}\t${selected_result##*/}\t${selected_score}" > "${route_dir}/FINAL_SELECTION.tsv"
            echo "[$(date '+%F %T%z')] selected-scored-attempt index=${index} route=${route_id} result=${selected_result} score=${selected_score}" | tee -a "${SUPERVISOR_LOG}"
        else
            echo "[$(date '+%F %T%z')] selected-scored-attempt index=${index} route=${route_id} result=none" | tee -a "${SUPERVISOR_LOG}"
        fi
    else
        echo -e "${index}\t${route_id}\t${route_status}\t${route_score}\t${evaluator_rc}\t${attempts}\t${route_dir}" >> "${STATUS_TSV}"
    fi
done

echo "[$(date '+%F %T%z')] CVCI hybrid run finished attempted=${#ROUTE_IDS[@]}" | tee -a "${SUPERVISOR_LOG}"
