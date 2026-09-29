# CVCI CARLA submission package (0929)

This repository is the relocatable delivery form of the current CVCI project. It contains the official evaluator wrapper, both route manifests, the frozen extreme-144 runtime, the current ordinary-220 online radar classifier and all 44 category rule entry points. The local delivery also contains every required checkpoint under `weights/`.

## Repository layout

```text
0929/
├── README.md
├── codes/
│   ├── run_all_routes.sh
│   ├── code/
│   ├── DriveTransformer/
│   ├── CVCI_BenchMark/
│   ├── scene_variants/extreme_144/
│   ├── historical_controllers/
│   ├── runtime_path/
│   ├── provenance/
│   ├── ordinary_rules/          # class_01.py ... class_44.py and router
│   ├── harness/                 # runtime radar feature/classifier modules
│   ├── data/classifier_model.json
│   ├── tools/verify_package.py
│   └── tests/
├── route/
│   ├── CVCI_BenchMark.xml
│   └── bench2drive220.xml
├── weights/
│   ├── iter_25000.pth
│   └── drivetransformer_large.pth
└── results/
    ├── 144/routes/route_<route_id>/result.json
    └── 220/routes/route_<route_id>/result.json
```

CARLA itself and the Python/CUDA environment are external platform dependencies. No project code depends on another checkout or any project directory outside this repository.

## Environment

The tested stack uses Linux, CARLA 0.9.15, Python 3.7, an NVIDIA CUDA runtime, PyTorch and the ABI-compatible compiled extensions already under `codes/`. The official requirements files are:

```text
codes/DriveTransformer/requirements.txt
codes/CVCI_BenchMark/scenario_runner/requirements.txt
codes/CVCI_BenchMark/leaderboard/requirements.txt
```

Because CARLA, PyTorch, CUDA, MMCV, xformers and OpenPCDet are ABI-coupled, reproduce with the competition environment or an equivalent Python 3.7/CUDA environment. If the packaged MMCV extensions do not match the target ABI, rebuild them with `codes/build_mmcv_extensions.sh`.

From the repository root, configure only the external environment:

```bash
REPO_ROOT="$(pwd)"
export PYTHON="$(command -v python)"
export CARLA_ROOT="${CARLA_ROOT:?set CARLA_ROOT to the CARLA 0.9.15 directory}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
```

`CARLA_ROOT/CarlaUE4.sh` must exist and be executable. The launcher derives all repository paths from its own location.

## Package and portable preflight

Run the repository completeness check first:

```bash
"${PYTHON}" codes/tools/verify_package.py

CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=220 \
  bash codes/run_all_routes.sh 2100 8100

CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=144 \
  bash codes/run_all_routes.sh 2100 8100
```

The preflight checks the selected route manifest, agent, checkpoint, classifier model, evaluator, scenario runner, reference manifests and CARLA launcher without starting a route.

## Reproduce ordinary-220 evaluation

Run all 220 routes:

```bash
CVCI_ROUTE_SET=220 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/reproduction_220" \
  bash codes/run_all_routes.sh 2100 8100
```

Fresh results are written beneath `results/reproduction_220/`, leaving the packaged `results/220/` evidence unchanged. Run one official route without editing XML:

```bash
CVCI_ROUTE_SET=220 \
CVCI_ONLY_ROUTE=1825 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/smoke_1825" \
  bash codes/run_all_routes.sh 2100 8100
```

The ordinary launcher uses `codes/code/ordinary_online_model_agent.py`, validates the first live radar packet, loads `weights/drivetransformer_large.pth`, reads `codes/data/classifier_model.json`, and attaches `ordinary_rules.router.OrdinaryRuleRouter`. Runtime category decisions do not read route IDs, filenames, test order or XML labels. The route XML is used by the official evaluator and launcher preflight only.

## Reproduce frozen extreme-144 evaluation

```bash
CVCI_ROUTE_SET=144 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/reproduction_144" \
  bash codes/run_all_routes.sh 2100 8100
```

Fresh results are written beneath `results/reproduction_144/`, leaving the packaged `results/144/` evidence unchanged. This path loads `weights/iter_25000.pth` and the frozen modules in `codes/scene_variants/extreme_144/`; the ordinary classifier/router is not attached to that branch.

## Output and validity

For each route, the canonical evaluator output is:

```text
results/<set>/routes/route_<route_id>/result.json
```

The repository contains 144 extreme-route and 220 ordinary-route final `result.json` files. The 144 set is copied unchanged from the reference package. In the 220 set, routes with a newer complete five-route official result set use those evaluator JSON files; all remaining routes retain the reference-package JSON files. Only terminal evaluator records with an official score are evidence. Missing, started, crashed or partial records are not silently removed or treated as successful. The launcher retains failed attempts, supports bounded retries and writes `ROUTE_STATUS.tsv` plus route logs.

## Model files and GitHub

The two required checkpoints are distributed through the same Baidu Netdisk package used by the reference project:

- Download: [Baidu Netdisk](https://pan.baidu.com/s/1x7SJgz9VWWaavOLNX09UBg?pwd=4a65)
- Extraction code: `4a65`

After downloading, place the files at:

```text
weights/iter_25000.pth
weights/drivetransformer_large.pth
```

They are ignored by normal Git because each is about 6 GB and cannot be stored as a regular GitHub blob. The local delivery already contains both files as ordinary files; a fresh GitHub clone must download them into the paths above before preflight. Do not replace them with links to another project directory.

## Provenance

The runtime source snapshot is identified in `docs/SOURCE_AUDIT.md`. Packaging changed only repository-path discovery and made the already-used classifier model a repository-relative launcher default; control, classifier and category-rule algorithms were not rewritten.
