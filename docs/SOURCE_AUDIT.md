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
- Package verification script and focused tests.

## Intentionally excluded

- Runtime logs and generated evaluator output.
- Recorded videos, extracted frames and temporary visual diagnostics.
- Python caches, pytest caches and stale lock files.
- The historical baseline-results symlink.
- Intermediate development artifacts containing host-specific paths.

These exclusions are not runtime dependencies.

## Logic preservation

The delivery copies the current runtime trees byte-for-byte. The only launcher edits are portability wiring:

1. prepend the repository root to `PYTHONPATH`, so `harness` and `ordinary_rules` resolve after relocation;
2. default `CVCI_ORDINARY_CLASSIFIER_MODEL` to `data/classifier_model.json` inside the repository;
3. require that classifier file during preflight and print it in the preflight report.

These edits reproduce the environment values previously supplied externally. They do not change classifier, planner, controller, thresholds, state machines or class-rule logic.

## Rules affecting packaging

The competition document requires a runnable agent, required weights/checkpoints, configuration and dependency instructions, a method description and inference/evaluation scripts. Evaluation is automated and must not rely on manual intervention or changes to the official runtime conditions. The ordinary benchmark score uses route completion multiplied by infraction penalty; the extreme benchmark uses scenario base score multiplied by collision penalty. The packaged launcher continues to call the official evaluator and preserves its raw JSON outputs.

## Frozen-144 protection

The frozen route XML, extreme checkpoint and extreme-only source files retain their recorded hashes. The ordinary runtime is attached only in the ordinary branch. Packaging did not edit the extreme-only rules, checkpoint, thresholds, model-selection profile or route definition.
