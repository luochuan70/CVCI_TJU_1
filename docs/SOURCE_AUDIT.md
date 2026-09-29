# Source and packaging audit

## Audited source

- Current development source snapshot: commit `e613290ae68ed1e02dc704e75016b305ebc6f256`.
- Self-contained layout reference/base: `0824`, commit `9a39267`.
- Official rules: `Introduction.pdf`, reviewed in full before packaging.
- Runtime code was at the recorded commit when the delivery snapshot was created.

The source project contained two external links: `weights` pointed at another project, and `baseline_results_0904` pointed at historical results. Neither link was copied. Both model files were copied into this repository as regular files. All required vendored model/evaluator code was copied physically.

## Included

- Official launcher and evaluator stack.
- DriveTransformer, packaged MMCV extensions, OpenPCDet and PointPillar checkpoint.
- Frozen extreme-144 code, routes, historical controllers and checkpoint.
- Current ordinary online agent, radar features, classifier model, router, local planner and all 44 class modules.
- Both official route manifests.
- All 144 extreme-route result JSON files and all 220 ordinary-route result JSON files.
- Package verification script and focused tests.

## Intentionally excluded

- Runtime logs and intermediate evaluator output.
- Recorded videos, extracted frames and temporary visual diagnostics.
- Python caches, pytest caches and stale lock files.
- The historical baseline-results symlink.
- Intermediate development artifacts containing host-specific paths.

These exclusions are not runtime dependencies.

The 144 result set is unchanged from the reference package. For ordinary routes, newer terminal evaluator JSON files are used only where a complete five-route result set is available; the remaining route results are unchanged from the reference package.

## Logic preservation

The delivery copies the current runtime trees byte-for-byte. The only launcher edits are portability wiring:

1. keep all imported project modules beneath `codes/`, which is already on the launcher `PYTHONPATH`;
2. default `CVCI_ORDINARY_CLASSIFIER_MODEL` to `codes/data/classifier_model.json` inside the repository;
3. require that classifier file during preflight and print it in the preflight report.

These edits reproduce the environment values previously supplied externally. They do not change classifier, planner, controller, thresholds, state machines or class-rule logic.

The packaged MMCV and OpenPCDet shared objects have no absolute ELF RPATH or
RUNPATH. They resolve PyTorch and CUDA libraries from the active, ABI-compatible
Python/CUDA environment described in the README rather than from a build-host
Conda directory.

## Rules affecting packaging

The competition document requires a runnable agent, required weights/checkpoints, configuration and dependency instructions, a method description and inference/evaluation scripts. Evaluation is automated and must not rely on manual intervention or changes to the official runtime conditions. The ordinary benchmark score uses route completion multiplied by infraction penalty; the extreme benchmark uses scenario base score multiplied by collision penalty. The packaged launcher continues to call the official evaluator and preserves its raw JSON outputs.

## Frozen-144 protection

The frozen route XML, extreme checkpoint and extreme-only source files are unchanged. The ordinary runtime is attached only in the ordinary branch. Packaging did not edit the extreme-only rules, checkpoint, thresholds, model-selection profile or route definition.
