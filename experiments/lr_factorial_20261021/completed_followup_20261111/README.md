# Completed learning rate follow up

All six new base trainings and 27 new forty-epoch continuations are complete. The fixed follow-up retains 84 evaluation endpoints (60 new and 24 reused), 72 fixed-model gradient batch probes, and 384 metric-definition cells. Source-bound scores and every prespecified contrast passed the [independent local checks](evidence/INDEPENDENT_LOCAL_VERIFICATION.json).

The primary question is whether changing the initial continuation rate from 0.0001 to 0.001 changes the paired distillation effect at epoch 40. MSE and CWD both use coefficient 0.3. An interaction is `(KD − control at 0.001) − (KD − control at 0.0001)`, measured in AP percentage points on native ten-class BDD AP50:95. A positive interaction does not by itself mean that KD improved on control.

## Primary results

| Method | Six-pair mean | Sample SD | Descriptive 95% t interval | Historical three mean | New three mean |
| --- | --- | --- | --- | --- | --- |
| MSE | −0.0249 | 0.0263 | −0.0525 to +0.0026 | −0.0106 | −0.0392 |
| CWD | −0.0017 | 0.0674 | −0.0724 to +0.0691 | +0.0177 | −0.0210 |

| Pair seed | Stratum | MSE interaction | CWD interaction |
| --- | --- | --- | --- |
| 101 | historical | +0.0139 | +0.0269 |
| 202 | historical | −0.0431 | +0.0813 |
| 303 | historical | −0.0027 | −0.0549 |
| 404 | new | −0.0210 | +0.0627 |
| 505 | new | −0.0505 | −0.0795 |
| 606 | new | −0.0462 | −0.0463 |

| Method | Initial rate | Six-pair mean KD minus control | Historical three mean | New three mean |
| --- | --- | --- | --- | --- |
| MSE | 0.0001 | −0.0225 | −0.0292 | −0.0158 |
| MSE | 0.001 | −0.0474 | −0.0398 | −0.0550 |
| CWD | 0.0001 | −0.0329 | −0.0497 | −0.0161 |
| CWD | 0.001 | −0.0346 | −0.0320 | −0.0371 |

Absolute native BDD AP was higher at the larger rate in 18 of the eighteen linked arm-by-pair comparisons. At the same rate, 4 of the four mean KD-minus-control effects were negative, ranging from −0.0474 to −0.0225 pp. A higher absolute score therefore does not establish a distillation advantage. Secondary evaluation definitions have different paired outcomes and remain secondary; these observations do not support a general method ranking.

These are six matched starting-pair replications, with shared COCO pretraining and datasets. Epochs, metric definitions and the two diagnostic batches do not add independent training replications. The three historical pairs were inspected before the extension was designed. The 95% t intervals use five degrees of freedom and are descriptive; no significance, equivalence or best-setting decision is made.

## Complete numerical records

- [Summary and primary interactions](results/SUMMARY.json)
- [All 84 endpoints and eighteen metrics](results/ALL_84_ENDPOINTS.json)
- [All eighteen metric-specific contrast sets](results/contrasts): 324 summaries and 1,944 pair-level differences
- [All 36 continuation paths and six new bases](results/TRAINING_36_PATHS_AND6_NEW_BASES.json): actual optimizer counters, missingness and 1,440 epoch records
- [All 72 gradient probes](results/GRADIENT_72_SUMMARY.json): raw final states, fixed FP32 inputs, no optimizer updates and explicit undefined values
- [All 384 metric cells and factorial effects](results/METRIC_384_CELLS_AND_EFFECTS.json): the same saved predictions under eight controlled definitions
- [Historical strength-grid reference](results/HISTORICAL_STRENGTH_GRID_REFERENCE.json)
- [Manuscript, supplement and figures](../../../manuscript/learning_rate_followup_20261111/README.md)

All machine-readable AP scores are fractions; differences are percentage points. The original 48-endpoint strength study is preserved. Its 24 overlapping endpoints give 108 distinct endpoints across both studies, not 132 independent observations.

## Verification and reuse

The [evidence archive](evidence/final030_text_evidence_20261111.zip) contains selected scientific text and code. Its [index](evidence/INDEX.json) maps original research paths to content-addressed files and records every size and SHA-256. The [public file manifest](PUBLIC_FILE_MANIFEST.json) identifies the delivered result, evidence, verification and manuscript files.

To repeat the scalar review, extract the evidence archive into a new empty directory and run `python verification/verify_downloaded_results.py /path/to/extracted_evidence` from this directory. It uses the Python standard library, verifies the entire index, and creates a new verification record only after all checks pass. It checks 1,512 scores, all paired contrasts, 1,008 metric-effect scalars, 96 exact source-score reference cells, raw training records and gradient summary identities. It does not perform model inference or independently reimplement native detection matching. Synthetic verifier fixtures are separate from research measurements.

Training diagnostics retain 120 historical undefined epoch medians and 342 native explicit-null medians from the new runs. New nulls have positive recorded nonfinite-gradient batch counts, as permitted by the frozen native instrumentation checks. The legacy reason field describes historical-literal conversion only; the local verification report separately explains every unavailable median and checks native batch partitions and cumulative counters. The [local verification schema review](evidence/LOCAL_VERIFICATION_SCHEMA_REVIEW.json) records the export and checker corrections. Neither the original records nor the completed analysis was rewritten. These are training diagnostics, distinct from the 72 fixed final-model probes.

The figure builder uses the verified extracted evidence and the versions in `verification/PLOT_REQUIREMENTS.txt`. Run `python verification/build_result_figures.py /path/to/extracted_evidence /path/to/new_figures` after successful scalar verification. Published figure hashes and source versions are in the manuscript's `FIGURE_MANIFEST.json`. OpenAI Codex assisted with workflow and validation code, figure preparation and drafting; this assistance is disclosed in the manuscript and figure captions.

Large model, image, label and prediction binaries are not included in this new text archive. Their server-side bindings and audit records are retained, but this is not a completed public deposit of all reproduction inputs. Durable binary access and third-party dataset permissions require author review before journal submission.

## Interpretation limits

BDD validation and the public 7,481-image KITTI training partition were previously examined. BDD labels were reconstructed from legacy 2018-format material and were not authenticated as official Detection 2020 labels. The controlled metric factorial retains COCO-style matching and is not native KITTI equivalence.

The server kernel changed from Linux 6.17.0-40 to 7.0.0-38 between historical and current execution. Source, package and input checks passed, but numerical equivalence across kernels was not established and historical scores were not rerun. This remains an unisolated execution-period difference, particularly for historical low-rate versus current high-rate comparisons. Student and adapter learning rates also changed jointly.

Two disclosed pair606 high-rate cold restarts and prior integration failures remain in the retained project records. The first synthetic gradient preflight stopped before any forward because an exact-dict check rejected saved OrderedDict containers. A container-only correction preserves tensors and state metadata; all 36 CPU assemblies and three full-size synthetic arms were checked before research probes. A later CPU admission hit the unchanged five-second disk-scan timeout; the correction uses four available CPU cores while retaining the aggregate CPU, wall, GPU and disk caps. Failure records and consumed resource charges remain included. Completed evaluation was explicitly retained under its original manifest and exit evidence.

The metric stage stopped before scoring because a request directory was missing. Five additional filesystem checks, including 66 actual atomic publications, checked the corrected request and contrast paths. Completed evaluation and gradient stages were retained under their original manifests and actual exit records; only unfinished metrics and analysis were dispatched. A conservative CPU reservation for the stopped attempt reduced the remaining allowance, with the original global deadline unchanged.

The selected text archive provides bound evidence and references; it is not a complete copy of every operational file. Completed results were not rerun for a favorable outcome. Fixed-model gradient probes describe final states and do not reconstruct stochastic mixed-precision training gradients. No new architecture, inference-speed improvement or general method ranking is established.
