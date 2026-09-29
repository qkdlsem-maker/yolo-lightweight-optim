# Evidence for the IEEE Access manuscript revision

Prepared 2026-09-29. Article: **Reassessing Pruning and Feature Distillation for Lightweight YOLOv8 on BDD100K**.

## Completed primary comparison

All nine runs finished: control / uniform MSE / CWD, each with independent data and augmentation seeds 0, 1, and 2, for 20 full epochs over 70,000 images. All start from one common trained nano detector. The primary endpoint is the fixed final epoch-20 EMA checkpoint, evaluated on the unchanged local validation labels (FP32, batch 32, 10,000 images, ten classes). No seed or epoch is selected for favorable accuracy.

| Arm | Mean AP50:95 | Sample SD | Mean paired difference from control (pp) | SD of paired difference (pp) |
|---|---:|---:|---:|---:|
| Control | 0.2591212631 | 0.0002555060 | 0 | 0 |
| MSE | 0.2588704316 | 0.0002787404 | -0.0250832 | 0.0239973 |
| CWD | 0.2588926446 | 0.0001409973 | -0.0228618 | 0.0147572 |

Neither objective gives a consistent positive primary-endpoint benefit. This descriptive comparison is conditional on one trained initialization and one continuation schedule. It does not establish equivalence, practical insignificance, or a general ranking of distillation methods. The three seeds are not three independently pretrained/fine-tuned baselines.

Sampled actual augmented inputs (the first two batches of every epoch) match across arms within a seed and differ across seeds. All three control states are distinct. All nine runs change 183 trainable student parameter tensors, preserve the fixed DFL projection, and pass finite-state checks. Each attempts 21,880 minibatches; AMP skips 4–6 updates, leaving 21,874–21,876 applied updates. EMA accounting matches. See `followup/AMP_NUMERICAL_AUDIT.md`; raw nonfinite median-gradient diagnostics are retained, not silently replaced.

## Completed annotation and secondary analyses

Full comparison with a public mirror of original-format **legacy 2018** annotations matches 185,526 positive-area validation boxes and 1,286,871 training boxes. The local conversion has 52 extra zero-area validation targets. Another 137 training images with 1,141 boxes are absent from the retrieved JSON and remain unverified. This is correspondence to a retrieved public mirror, not authentication against an official-host archive. See `DATA_PROVENANCE.md`.

Seven archived detectors and all nine final EMA students were evaluated on a separately reconstructed validation view: **16 completed secondary evaluations**, with exact agreement between recorded per-image aggregation and built-in validator AP. Global scores use ten classes; daytime/night/dawn-dusk summaries use the same nine road classes excluding sparse train. A fixed-prediction analysis isolates the zero-area-target denominator effect. Its inferred original-label baseline AP50 and AP50:95 exactly match an independent direct evaluation. The secondary batch size is 8, so changes from primary scores cannot all be attributed to labels.

## Historical evidence kept separate

Nine archived detectors were freshly evaluated, exact checkpoints were profiled, and ten model/precision settings were timed in ten shuffled blocks. Two archived standalone KD artifacts leave all 184 student parameter tensors unchanged and cannot estimate a learned KD effect.

The earlier **six ten-epoch probe executions** use one fixed data sequence. Their three controls have identical states, while three KD runs vary adapter initialization. They constitute one unique control and three adapter initializations, not three independent paired data seeds. They are separate from the final nine-run study and are not pooled with it. Their numerical records remain under `results/controlled/` and `tables/continuation.csv`.

## Package map

- `followup/confirmatory/`: every primary run's configuration, epoch records, gradient/parameter checks, EMA metrics, and completion record.
- `followup/confirmatory_summary.json` and `confirmatory_results.csv`: every final primary score and paired difference.
- `followup/final_analysis.json`, `final_secondary_paired.csv`, and `final_optimizer_updates.csv`: descriptive derived summaries and actual update counts.
- `followup/stratified/`, `secondary_metrics.csv`, and `annotation_sensitivity*.json`: all 16 secondary results and annotation checks.
- `followup/protocol.md` and `secondary_protocol.md`: prespecified primary and secondary designs, retained unchanged.
- `followup/scripts/`: executed training, validation, audit, aggregation, and analysis code. `confirmatory_kd.py` generated the nine-run results.
- `followup/publication.json`: public repository revision for the delivery. The executed training-code revision is `efcd6484a4844b976ca63484bd1fd5b8d8d983f0`.
- `results/`: historical reassessment and fixed-data probe records. `completed_run_manifest.json` identifies checkpoint and canonical state hashes.
- `scripts/as_executed/` and `scripts/archived_helpers/`: code and helpers actually used for the earlier probe; intentionally not silently repaired.
- `scripts/revised/`, `student_training_fixes.patch`, and `tests/verification.txt`: separate repairs and regression checks. This generic repaired runner is not the final nine-run runner.
- `tables/`, `figures/`, and `reference_verification.json`: convenient historical summaries, measured-data figures, and source-verification scope.
- `SHA256SUMS.txt`: checksums for exact packaged bytes (ZIP delivery only).

Metrics are fractions; differences ending in `pp` are percentage points. Sample SD describes the stated three seeds or ten timing blocks, not an unstated population confidence interval. Raw epoch JSON follows the executed Python writer, which may encode a diagnostic as `NaN`; strict JSON consumers must explicitly handle that extension. Final aggregate analysis uses finite strict JSON values.

## Reproduction

Use the versions in `environment.txt`, the original repository helpers, authorized BDD100K inputs, and the exact source/teacher checkpoints identified in run configurations. Dataset images, full source-label archives, model weights, and per-image NPZ prediction statistics are not redistributed. Hashes identify inputs but do not replace them; independent reproduction is incomplete without obtaining them. Follow original dataset and software licenses.

From the original project root, use a **new output directory** for any replication. Do not overwrite the recorded runs. The final training script SHA-256 is `f3baa82158110bbbc46bcc9053f78101d3fd8e0c18ebd4e5751d5dbf7d96cae1`; this identifies the unchanged executed source. All hyperparameters, seed rules, inputs, and endpoint rules are in the protocol and per-run configurations. Example for one of the nine specified runs:

```sh
python experiments/confirmatory_kd.py --device 0 --arm mse --seed 0 --epochs 20 --out-root replication/confirmatory
```

Repeat only when intentionally replicating, for each arm and seed. The nine completed experimental runs already exist; no replication is required to read these results. The final source used batch-size-scaled KD, no mosaic, a cosine schedule, and a final EMA endpoint; do not substitute the earlier fixed-data probe or generic repaired script.

After all prescribed outputs exist, `experiments/summarize_confirmatory.py` checks their final metrics and sampled input pairing. `experiments/audit_confirmatory_numerics.py` checks saved numerical states. Use their `--help` options for project/output locations. `experiments/analyze_final_results.py artifacts/ieee_access_20260929/followup` regenerates the published descriptive tables from the recorded JSON without training. Source-label reconstruction and secondary-evaluation commands are in `DATA_PROVENANCE.md`.

## Interpretation boundaries

The complete C2f conversion **including** `initialize_weights` gives zero maximum absolute output difference on the tested input. The earlier incomplete probe in `kd_audit.json` must not be reported as a defect in the implemented full conversion. Already fused standalone KD checkpoints are excluded from the manuscript's unfused complexity comparison. File size is MiB = bytes / 2^20.

GPU timing excludes decoding, preprocessing, transfer, and NMS. GPU 1 was training while GPU 0 performed timing; clocks were not fixed. Timing SD describes block means, not per-image tail latency. FP16 accuracy was not reassessed. Structural reductions did not produce acceleration in the measured workload; this does not predict every device.

The study does not establish an untouched test-set result, current Detection 2020 benchmark comparability, general KD failure, embedded-device acceleration, or vehicle safety. The 20,000 externally supplied test labels have unknown origin and were excluded. Tae-Wan Kim's biography remains blank at the author's request. The author team must complete that biography and review the scientific claims, authorship, funding, and AI disclosure before journal submission. No acceptance guarantee or journal submission is represented.
