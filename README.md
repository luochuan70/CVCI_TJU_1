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
│   └── provenance/
├── ordinary_rules/              # class_01.py ... class_44.py and router
├── harness/                     # runtime radar feature/classifier modules
├── data/
│   └── classifier_model.json
├── route/
│   ├── CVCI_BenchMark.xml
│   └── bench2drive220.xml
├── weights/
│   ├── iter_25000.pth
│   └── drivetransformer_large.pth
├── scripts/verify_package.py
├── tests/
└── results/                     # generated evaluator output
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

## Integrity and portable preflight

Run the repository audit first. The full form hashes both multi-gigabyte model files; `--quick` skips only those expensive hashes.

```bash
"${PYTHON}" scripts/verify_package.py

CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=220 \
  bash codes/run_all_routes.sh 2100 8100

CVCI_PREFLIGHT_ONLY=1 CVCI_ROUTE_SET=144 \
  bash codes/run_all_routes.sh 2100 8100
```

The preflight checks the selected route manifest, agent, checkpoint, classifier model, evaluator, scenario runner, reference manifests and CARLA launcher without starting a route.

## Reproduce ordinary-220 evaluation

Run all 220 routes:

```bash
CVCI_ROUTE_SET=220 bash codes/run_all_routes.sh 2100 8100
```

Results are written beneath `results/220/`. Run one official route without editing XML:

```bash
CVCI_ROUTE_SET=220 \
CVCI_ONLY_ROUTE=1825 \
CVCI_OUTPUT_ROOT="${REPO_ROOT}/results/smoke_1825" \
  bash codes/run_all_routes.sh 2100 8100
```

The ordinary launcher uses `codes/code/ordinary_online_model_agent.py`, validates the first live radar packet, loads `weights/drivetransformer_large.pth`, reads `data/classifier_model.json`, and attaches `ordinary_rules.router.OrdinaryRuleRouter`. Runtime category decisions do not read route IDs, filenames, test order or XML labels. The route XML is used by the official evaluator and launcher preflight only.

## Reproduce frozen extreme-144 evaluation

```bash
CVCI_ROUTE_SET=144 bash codes/run_all_routes.sh 2100 8100
```

Results are written beneath `results/144/`. This path loads `weights/iter_25000.pth` and the frozen modules in `codes/scene_variants/extreme_144/`; the ordinary classifier/router is not attached to that branch.

## Output and validity

For each route, the canonical evaluator output is:

```text
results/<set>/routes/route_<route_id>/result.json
```

Only terminal evaluator records with an official score are evidence. Missing, started, crashed or partial records are not silently removed or treated as successful. The launcher retains failed attempts, supports bounded retries and writes `ROUTE_STATUS.tsv` plus route logs.

## Model files and GitHub

The delivered folder contains the two required checkpoint files as ordinary local files. Their exact sizes and SHA-256 values are in `WEIGHTS.sha256`. They are ignored by normal Git because each is about 6 GB and cannot be stored as a regular GitHub blob. When publishing this repository, upload the verified files through an appropriately provisioned Git LFS or release/artifact service, then restore them to the exact `weights/` paths before preflight. Do not replace them with symlinks to another project directory.

## Provenance

The runtime source snapshot is identified in `docs/SOURCE_AUDIT.md`. Packaging changed only repository-path discovery and made the already-used classifier model a repository-relative launcher default; control, classifier and category-rule algorithms were not rewritten.
