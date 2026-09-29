# New follow-up: training in progress

This phase was requested after the earlier nine-run analysis and manuscript were completed. It has **no final training or external test results yet**. Do not confuse software smoke tests with study outcomes.

The 137 images have not acquired authenticated labels. They are excluded from a fresh 69,863-image training population. Both nano and large models restart from official COCO-pretrained weights, removing exposure to those labels in this new BDD training phase. Old models and their limitations remain unchanged. Original-format legacy 2018 JSON is from an audited public mirror, not a newly authenticated official-host archive.

Both native one-epoch software smoke runs and all three two-minibatch objective checks passed. Full native loading found one exact duplicate source row; the one-box count reconciliation was documented before full training. The native loader retains 1,286,870 boxes from 1,286,871 source rows. All 69,863 images remain. Actual paired input hashes and 183 changed student parameter tensors were checked in the tiny objective runs. These are implementation checks, not final KD effectiveness estimates.

Two new 100-epoch base trainings have started, followed automatically by nine new 20-epoch continuations. Final endpoints are fixed, and earlier completed experiments are preserved. KITTI input download and preparation are in progress. Its official 2D scoring core passed eight synthetic cases; class/coordinate serialization and FP32 inference were checked without real KITTI images. All 25 model choices are fixed in advance; actual checkpoint hashes will be frozen after all trainings and before any real KITTI inference.

KITTI uses its 7,481 publicly labeled training-partition images as an external holdout for this study. It is not an official hidden-test leaderboard result. No KITTI fitting, adaptation or checkpoint selection is planned. Car and Pedestrian are evaluated with native ignored-label and difficulty rules and R40. Neither dataset independence nor clean training can guarantee acceptance or a revision decision.

See the [full protocol](../../experiments/clean_followup/protocol.md). The completed earlier paper must be revised only after these new outcomes and all audits are actually complete.
