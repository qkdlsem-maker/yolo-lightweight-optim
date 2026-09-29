# Evidence for the IEEE Access manuscript revision

Prepared 2026-09-29. Article: Reassessing Pruning and Feature Distillation for Lightweight YOLOv8 on BDD100K.

## What was completed

Nine archived detectors were freshly evaluated on all 10,000 validation images. Checkpoints were profiled directly, ten timing configurations were measured in ten shuffled blocks, and six ten-epoch continuations on all 70,000 training images finished. Six continuation outputs are execution records, not six independent statistical samples.

The installed Ultralytics data-loader used the same generator seed for all runs. All three control states are exactly equal. The three KD runs vary adapter initialization under that fixed data sequence. The primary result is one unique control and three KD adapter initializations. Do not describe the result as three independent paired training seeds. No p-value or confidence interval is supplied. All continuations scored below the starting baseline.

## Package map

- `results/`: original JSON and CSV records, including every continuation's configuration, losses, gradient check, parameter-update check, and metrics. `completed_run_manifest.json` records checkpoint-file and canonical tensor-state hashes and the installed loader source.
- `tables/`: convenient derived CSV summaries. Metrics are fractions; differences with suffix `pp` are percentage points.
- `figures/`: measured-data figures, exported at 400 dpi.
- `scripts/as_executed/`: scripts used for the completed reassessment. These preserve the fixed-data design actually run. They import helper functions from the project's original `scripts/` directory.
- `scripts/archived_helpers/`: original helper snapshots for reproducing that execution; these are intentionally not silently repaired.
- `scripts/revised/`: subsequent student-trainability, probe-state, checkpoint-saving, and explicit-data-seed corrections, plus regression tests. The explicit-data-seed revision was tested but was NOT used for the completed full training runs in this package.
- `student_training_fixes.patch`: changes relative to the original scripts. Corrected code and evidence have been published to the GitHub main branch; archived server scripts are retained separately.
- `tests/verification.txt`: both regression tests passed in the experiment environment.
- `reference_verification.json`: reference sources and scope of verification. This is not a comprehensive retraction-screening certificate.

## Important audit distinctions

In `kd_audit.json`, the early C2f probe deliberately omits the subsequent initialization call and differs from the full implemented conversion. The decisive full-pipeline test is `evaluation/yolov8n_baseline.json`, field `conversion_with_initialize_weights`: maximum absolute difference 0 and allclose true. Do not report the incomplete probe as a defect in the actual full conversion.

The two archived standalone KD artifacts are fused. Their JSON key `params_unfused` records the parameter count of the loaded graph before evaluation, but cannot undo pre-existing fusion. They are excluded from the manuscript's unfused complexity comparison. The baseline and pruned models in that comparison are unfused. Model file size is MiB = bytes / 2^20.

GPU timing excludes decode, preprocessing, transfer, and NMS. GPU 1 was training while GPU 0 performed timing; the server was shared and clocks were not fixed. SD describes ten block means, not per-image tail latency. FP16 accuracy was not reassessed.

## Reproduction prerequisites and commands

Use the original repository at commit `4f5be41e22ea8696800e94611688777a7cbb883a`, with the versions in `environment.txt`, the authorized BDD100K conversion and original checkpoint paths recorded in the JSON files. Dataset images and model weights are not redistributed in this package. Weights remain in the existing project and continuation output directories; hashes allow identification. Reproduction without those inputs is incomplete. Follow the dataset and original software licenses.

From the project root, place the as-executed scripts in `review_20260922/`, retaining the original helper scripts in `scripts/`. Output directories must be new or the controlled runner will stop; it never overwrites a completed run.

```sh
python review_20260922/remote_kd_audit.py
python review_20260922/review_evaluate.py
python review_20260922/review_benchmark.py
python review_20260922/controlled_kd.py --device 1 --arm control --seeds 0 1 2 --epochs 10
python review_20260922/controlled_kd.py --device 0 --arm kd --seeds 0 1 2 --epochs 10
```

For independent-data experiments, use the revised code in a separate checkout and run the tests from `scripts/` with the baseline checkpoint available at `../outputs/yolov8n_baseline/weights/best.pt`:

```sh
python test_kd_training_state.py
python test_seeded_loader.py
```

The patched training script accepts `--seed`. Its checkpoint selection and default training schedule differ from the fixed-final-epoch controlled runner; do not mix their results or claim that running it exactly reproduces Table IV.

## Submission status

The manuscript honestly reports the available bounded evidence, including unfavorable results. It does not establish a generally effective new compression method, independent training-seed robustness, an untouched test-set result, or embedded-device acceleration. Tae-Wan Kim's biography remains blank at the author's request. The author team must complete that biography and review all authorship, funding, AI disclosure, and scientific claims before submission. No acceptance guarantee or journal submission is represented.

## September 29 follow-up

The nine-run control/MSE/CWD comparison with independent data seeds is running; it has no completed final results in this package. Preflight and duplicate-audit records, protocol, and code are included under `followup/`. The original source is commit 4f5be41e22ea8696800e94611688777a7cbb883a; corrected code and audit records were published in commit efcd6484a4844b976ca63484bd1fd5b8d8d983f0. See DATA_PROVENANCE.md for distribution-source sample matching and unresolved annotation generation.
