# Clean-label replication and external evaluation protocol

Specified on 2026-09-29 before new training or KITTI model evaluation. This is a new follow-up phase prompted by the remaining annotation and held-out evaluation limitations. Prior completed experiments remain unchanged and are not pooled with the new runs.

## Data and label uncertainty

The retrieved original-format legacy BDD100K JSON contains 69,863 training image annotations and 10,000 validation annotations. The 137 additional local training images, with 1,141 boxes of unknown origin, are excluded from the new training population. All retained labels are reconstructed from the retrieved JSON: 1,286,871 training boxes and 185,526 validation boxes, with none of the extra local zero-area boxes. The source JSON and converted labels are hashed. Original datasets are never overwritten.

This exclusion resolves exposure to the unverified labels for **newly trained models**; it cannot erase their influence on the archived models. It also does not authenticate the mirror against an unavailable official-host archive or reclassify legacy 2018 annotations as Detection 2020.

Preflight clarification before full training: native Ultralytics loading removes one exact repeated box in image `75055858-7d04a650`. The reconstructed source contains 1,286,871 rows; the native training loader retains 1,286,870 boxes on the same 69,863 images. Preserve the source label file and native deduplication behavior. This count reconciliation is recorded rather than silently changing the source labels.

## Fresh student and teacher

Train a new YOLOv8n baseline and YOLOv8l teacher from official Ultralytics COCO-pretrained checkpoints, not from any previous BDD100K-trained checkpoint. Record input-file SHA-256 and source release URLs. Each runs 100 full epochs, image size 640, seed 0, deterministic setting, native Ultralytics 8.2.103 training with explicit SGD (native Nesterov behavior), initial learning rate 0.01, final multiplier 0.01, linear learning-rate schedule, momentum 0.937, weight decay 0.0005, warm-up 3 epochs, AMP, eight workers, default augmentation, and mosaic disabled for the final ten epochs. The nano minibatch is 64; the large-teacher minibatch is 16 with the native trainer's nominal batch size 64 and gradient accumulation (including its warm-up accumulation behavior). This memory choice is fixed before training, not selected by accuracy. Early stopping is disabled (patience 0). Both starting-model endpoints are the final epoch-100 EMA weights (`last.pt`), not the best validation epoch.

Validation is diagnostic. The BDD100K validation split has been repeatedly consulted in the historical campaign; this phase does not relabel it as untouched. Fresh training is essential: simply deleting 137 label files and continuing from the old BDD100K checkpoint would retain their historical influence. Technical failures may be resumed under the same declared configuration; no outcome-driven extension or substitution is permitted.

## Nine-run objective replication

After both fresh models complete, repeat the predefined control/MSE/CWD comparison on the 69,863 verified-source training images: three independent data/augmentation seeds (0,1,2), 20 epochs per arm, batch 64, 640-pixel setting, SGD momentum .937 and weight decay .0005, cosine learning rate .0001 to .00001, AMP initial scale 128, gradient norm clipping 10, no mosaic/mixup/copy-paste. Use the unchanged `confirmatory_kd.py` runner with new explicit `--source`, `--teacher`, `--data`, and `--out-root` arguments. MSE/CWD share weight .3 on the per-image scale; CWD temperature is 4. The endpoint is the final epoch-20 EMA student. All nine outcomes are retained.

Preserve explicit sampler and worker seeding and first-two-actual-batch hashes in every epoch. Within-seed inputs must pair across objectives and between-seed inputs must differ. Require distinct controls, 183 changed trainable student parameter tensors, preserved DFL projection, no teacher gradients and finite final states. Report actual AMP-skipped/applied update counts. The expected full-epoch minibatch count changes to ceil(69863/64) = 1092; this is a declared data change, not a relaxed test. Primary internal metric remains ten-class BDD100K AP50:95 on reconstructed validation labels; AP50, per-class and illumination metrics are secondary. Three seed differences are descriptive, conditional on one fresh base-model initialization.

## External KITTI evaluation

Use all 7,481 publicly labeled KITTI object-detection **training-partition** images as an external test population for these BDD100K models. None is used for this study's training, adaptation, early stopping, hyperparameter choices or checkpoint selection. This is cross-dataset testing on public ground truth, not a KITTI official hidden-test leaderboard submission. Record the archive hashes against TensorFlow Datasets' public KITTI checksum registry and check exact image-file overlap with BDD100K. KITTI frames may be correlated within sequences; disclose this rather than claiming independent image sampling. COCO pretraining provenance is recorded, without asserting a full image-level audit of COCO pretraining.

Before observing KITTI scores, fix the compared artifacts: the archived nano baseline, three archived pruned operating points, archived recovery-KD model, all nine earlier final EMA students, the fresh clean-label nano baseline and teacher, and all nine new clean-label final EMA students (25 models total). The small/large archived teachers are not needed for the main external comparison. All selected model checkpoints are frozen before their KITTI inference. Do not choose an arm or epoch by KITTI accuracy.

Primary external categories are **Car and Pedestrian**, mapped from BDD100K car (class 2) and person (class 0). Cyclist is excluded because a KITTI cyclist box can encompass the rider and bicycle whereas BDD100K separates rider/bike, making a direct mapping inappropriate. Keep the native KITTI ignored-ground-truth rules (Van for Car, Person_sitting for Pedestrian, DontCare regions), difficulty filters, and IoU thresholds of .7 for Car and .5 for Pedestrian. Report bounding-box AP at 40 recall positions (R40) for easy/moderate/hard; the primary external summary is the mean of the two moderate APs, accompanied by both class values. This is not the ten-class BDD100K AP metric.

Use the official KITTI object devkit evaluation logic with only documented adaptations needed for local labeled-partition paths/frame count and R40 aggregation if its bundled output retains the older 11-position display. Preserve the downloaded source and a patch; validate evaluator behavior with synthetic perfect, missing, false-positive, DontCare and ignored-neighbor cases before real scores. If the available evaluator cannot be verified, stop external scoring rather than substitute an unreported metric. Inference is fixed at image size 640, square letterboxing (rect=False), batch 8, FP32, confidence .001, class-aware NMS IoU .7 and max 300 detections across the original ten classes, without augmentation or test-time adaptation. Retain class 0/2 detections after native NMS. Do not treat inference duration as a latency benchmark.

## Reporting and boundaries

Report the clean replication and external results alongside, not as silent replacements for, the prior nine-run study. Give every seed and paired objective difference, data/source hashes, protocol, failures/resumptions, and exact evaluator changes. No significance, equivalence, general KD failure, automotive safety or acceptance guarantee follows automatically from these additions. An unfavorable result is retained. Update the manuscript, review, evidence and GitHub only with verified completed results; keep the completed earlier manuscript available while this phase runs.

## Sources

- BDD100K documentation distinguishes legacy and revised labels: https://raw.githubusercontent.com/bdd100k/bdd100k/master/doc/source/download.rst
- Original-format mirror used and audited: https://www.kaggle.com/datasets/solesensei/solesensei_bdd100k
- KITTI evaluation, ignored regions/difficulty and R40 policy: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d
- Public archive URLs and checksums: https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/kitti/checksums.tsv
