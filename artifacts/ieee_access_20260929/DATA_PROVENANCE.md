# Dataset provenance and full annotation comparison

The historical campaign used an external YOLO conversion with 70,000 training, 10,000 validation and 20,000 test images. The original download was not recorded contemporaneously. Five byte-identical configuration/label/image samples identify the likely distribution as [a7madmostafa/bdd100k-yolo, version 4](https://www.kaggle.com/datasets/a7madmostafa/bdd100k-yolo). This sample trace is distinct from the full annotation comparison below.

## Full comparison to original-format legacy JSON

On 2026-09-29, training and validation JSON were retrieved from the [SoleSensei public mirror](https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k), a distribution predating the 2020 revision. The [official BDD100K documentation](https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/download.rst) distinguishes legacy 2018 annotations from revised Detection 2020 labels and recommends the latter. **These historical experiments correspond to the legacy 2018 annotation set, not Detection 2020.** The comparison authenticates correspondence to the retrieved mirror, not an official-host archive signature.

| Scope | Source images | Source boxes matched | Extra local zero-area boxes | Unverified local data |
|---|---:|---:|---:|---|
| Validation | 10,000 | 185,526 | 52 in 51 images | No unmatched images or positive-area boxes |
| Training | 69,863 | 1,286,871 | 393 in 385 matched images | 137 additional images with 1,141 boxes |

All source boxes and all positive-area local boxes in the matched images agree after normalizing class ID and coordinates to six decimals. Row order is ignored while duplicate multiplicity is retained. It would be incorrect to claim that all 70,000 training images were authenticated. The original files were not changed and the existing trained baseline already reflects the unverified subset.

SHA-256 of the uncompressed original-format JSON:

- Validation: `c835be652f0c002897be51e74deedd16a1dcee0b7230f19c0456fba772f694af`
- Training: `c1998792e385fd79213187576ca07d5249563d0ab83ae976688ad60668baca9e`

See `followup/legacy_full_audit.json`, `legacy_val_download.json`, and `legacy_train_download.json` for counts, examples, URLs, and archive metadata. The source JSON itself is not redistributed. The dataset's original terms remain applicable; a third-party package's license label must not be assumed to replace them.

## Accuracy sensitivity and illumination

A separate reconstructed validation view contains exactly the 185,526 source boxes and the same images. Seven archived operating points were evaluated at FP32, batch 8, image size 640. Per-image statistics exactly reproduce the built-in validator AP. The illumination analysis uses source-JSON metadata: daytime 5,258, night 3,929, dawn/dusk 778, undefined 35 images. Global AP averages all ten classes; group AP averages the fixed nine road classes excluding sparse train, and is null if support is incomplete. Do not compare these different class averages directly.

Holding predictions fixed and adding/removing only the 52 zero-area targets isolates the annotation effect. Baseline AP50 changes by +0.009036 percentage points after their removal. The inferred original-label AP50 and AP50:95 exactly agree with a separate direct original-label baseline evaluation at the same batch size. The small target-denominator effect does not explain the much larger observed pruning losses. Full per-model and subgroup outcomes are in `followup/stratified/`, `secondary_metrics.csv`, `annotation_sensitivity.json`, and `annotation_sensitivity_check.json`.

This is a secondary analysis. The prespecified nine-run training comparison retains the unchanged original validation conversion as its primary endpoint; it will also evaluate all final checkpoints on the reconstructed view. No results from unfinished final runs are imputed. The secondary evaluations share GPU resources with training and their elapsed times are not latency benchmarks.

## Duplicate and test-label limitations

SHA-256 comparison of all 100,000 encoded images found no exact train/validation overlap. Training contains 69,999 unique image hashes, validation 9,998, and one image is shared by train and test. No exact cross-split label-file duplicates were found. This does not exclude perceptual or video-level overlap. See `followup/data_hash_audit.json`.

The external test-label generation process is still unknown. [BDD100K maintainers describe withheld test annotations and server evaluation](https://github.com/bdd100k/bdd100k/discussions/52). The 20,000 local test labels are excluded from all accuracy claims. The reconstructed view does not create an untouched test set or establish contemporary benchmark comparability.

## Reproduction

Use the versions in `environment.txt` and obtain the authorized images/checkpoints separately. The audit script `experiments/audit_legacy_conversion.py` expects `work/reference_labels/legacy_val.json`, `legacy_train.zip` (one original JSON member), and `local_train_labels.zip` / `local_val_labels.zip` with flat `<image-stem>.txt` members. It streams the large training JSON, records source hashes, and writes reconstructed-label archives and attributes without rewriting existing data. The coordinate conversion assumes the BDD100K 1280 by 720 frames.

Place the resulting `legacy_val_reconstructed.zip` and `legacy_val_attributes.json` under `review_20260929/` in the project. Run `experiments/setup_verified_validation.py` from the project root to create the isolated view. Evaluate checkpoints with `experiments/stratified_validation.py --checkpoint <path> --name <artifact>`; archived pruned checkpoints require the repository's `scripts/c2f_v2_utils.py`. Use the supplied secondary protocol for thresholds and all seven model paths.

Run `experiments/build_zero_area_manifest.py` against the same flat local validation-label archive, then put its manifest under `review_20260929/`. After evaluations, run `experiments/annotation_sensitivity.py`. A direct baseline run with `--data configs/bdd100k.yaml --annotation-set original-local --out-root review_20260929/original_label_check` enables `experiments/verify_annotation_sensitivity.py`. Per-image NPZ statistics remain in the experiment outputs and can be regenerated; published summaries and hashes are not a substitute for the required inputs.
