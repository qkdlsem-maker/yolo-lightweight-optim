# Completed clean-source and external evaluation

Both fresh100-epoch base trainings, all nine20-epoch continuations, all25frozen KITTI evaluations and all11new reconstructed-label class/illumination evaluations are complete. No completed experiment was repeated to obtain a preferred outcome. This record is separate from the earlier campaign.

The fresh phase excludes137images with source-unverified labels and restarts student/teacher from official COCO checkpoints. Native loading retains1,286,870targets over69,863images from1,286,871source rows, preserving one exactly duplicated source row in the input files. These are reconstructed legacy2018annotations from an audited mirror, not authenticated Detection2020data.

Primary ten-class BDD AP50:95 means: control0.2591236363, MSE0.2588043595, CWD0.2583395409. Mean paired differences: MSE-0.03192768pp; CWD-0.07840954pp. Every fresh internal paired difference is negative. Three data/augmentation seeds share one newly trained student/teacher pair; they are not three independent base trainings.

External moderate Car/Pedestrian AP_R40 means: control0.7016589328, MSE0.7020294180, CWD0.7030743041. Mean paired differences are+0.03704852pp and+0.14153713pp, positive in every fresh seed. All nine remain below fresh nano0.7081111667. All25models and all six class/difficulty values are retained. This is external use of KITTI's publicly labeled training partition, not an official hidden-test score; no fitting, selection or extension uses KITTI outcomes. Dataset, class population and metric change together, preventing a pure domain-shift causal interpretation.

Independent checks reproduce every primary per-class mean and descriptive seed aggregate, all150external precision-curve aggregates and every new full-set/illumination/class AP from per-image statistics. All187,025KITTI prediction files and frozen checkpoints are independently rehashed. Each fresh run attempts21,840minibatches; four or five AMP skips are disclosed, and applied/EMA counts match exactly. Raw nonfinite gradient diagnostics in skip epochs remain preserved; all final model/optimizer tensors are finite.

See protocol.md in experiments/clean_followup, all per-run JSON, external_all_models.csv, independent_training_record_verification.json, external_record_verification.json and stratified_statistics_verification.json. Inference checkpoint and prediction/statistics archives are prepared with source/release SHA-256 mappings; their release publication state is recorded separately. No acceptance, revision or completed journal submission is claimed.
