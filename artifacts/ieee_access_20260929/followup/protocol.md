# Prespecified continuation comparison

Written before full-run results on 2026-09-29. The September 22 frozen-student audit and fixed-data continuation are retained as historical evidence; they are not overwritten.

## Primary comparison

Three arms (no-KD control, uniform feature MSE, and channel-wise spatial KL distillation/CWD) with data and adapter seeds 0, 1, and 2: nine runs. Each starts from the same existing YOLOv8n BDD100K checkpoint and uses the same YOLOv8l teacher where applicable. CWD follows Shu et al., ICCV 2021, with temperature 4 and a learned student-to-teacher 1x1 adapter on each neck scale. This is an adaptation to YOLOv8, not a claim to reproduce the paper's original RetinaNet scores. The CWD value and gradient are checked against the OpenMMLab reference formula.

All arms: full 70,000-image training split, batch 64, image size 640, 20 epochs, SGD momentum .937 and weight decay .0005, initial learning rate .0001 with cosine decay to .00001, AMP, gradient norm clipping at 10, and the same default geometric/photometric augmentations except mosaic/mixup/copy-paste disabled throughout. The common student is already converged; this conservative schedule addresses the performance loss in the earlier constant-rate mosaic continuation. These schedule changes are not attributed to KD.

Loss: native batch-summed detection loss plus batch size times lambda (.3) times the sum of the three scale-wise mean feature losses. Multiplication by batch size makes lambda a per-image weighting. This differs explicitly from the historical loss normalization. An equal numerical weight does not imply equal gradient strengths or optimal tuning for each method; the first-step loss ratio and gradient norm are recorded.

## Seed and checkpoint checks

AMP uses an initial gradient scale of 128. An eight-batch preflight verifies actual student updates and no skipped optimizer steps in all three arms before full runs. A technical preflight at the default scale of 65536 skipped its first two optimizer updates; those probes are not experimental results. Interrupted validation is recoverable from the saved EMA training state.

The DataLoader generator and all workers use seed + 100003*epoch. First two augmented input batches of every epoch are hashed. Hashes must agree across arms for the same seed and differ between distinct seeds. Three identical controls will fail the independence audit, not be averaged as independent evidence. Every run must show nonzero student gradients and changed student weights; the DFL integral projection stays fixed. Dimension probes must not alter BN buffers. Teacher parameters have no gradients.

Primary endpoint: final epoch EMA AP50:95 on all 10,000 validation images. AP50 and all class APs are secondary. Epochs 5 and 10 are recorded only for learning curves; no best-score checkpoint is selected. Every completed run, including unfavorable results, is retained. Paired differences across the three independent data seeds are reported with raw values and descriptive mean/SD. Three pairs do not justify strong claims of equivalence or general superiority.

## Supporting evaluation

Check whether an existing disjoint labeled split is available and document provenance before calling it held-out evaluation. If available, do not use it for training or parameter selection. Timing must distinguish device-only and end-to-end inference and retain raw measurements. Repeat timing after training jobs finish if an idle-server comparison is needed. Additional evaluations cannot retroactively turn the repeatedly used validation set into an untouched test set.

## Stopping and reporting

Each training run stops at 20 epochs. It may stop earlier only for a technical failure (nonfinite loss, invalid gradients, storage/device error), which is recorded. Do not extend or select a run because its score is favorable. Any later hyperparameter exploration must be labeled exploratory, and its full attempted matrix must be disclosed. Acceptance or revision remains an editorial decision; additional experiments strengthen evidence, not guarantee a decision.

## Sources

- Shu et al., Channel-Wise Knowledge Distillation for Dense Prediction, ICCV 2021: https://openaccess.thecvf.com/content/ICCV2021/html/Shu_Channel-Wise_Knowledge_Distillation_for_Dense_Prediction_ICCV_2021_paper.html
- OpenMMLab reference formula: https://github.com/open-mmlab/mmrazor/blob/main/mmrazor/models/losses/cwd.py
