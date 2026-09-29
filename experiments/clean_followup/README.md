# Clean-source training and external evaluation

Status: **in progress**, as described in [the evidence status](../../artifacts/clean_followup_20260929/STATUS.md). This directory adds a separate phase; it does not overwrite earlier results.

Read [protocol.md](protocol.md) first. The selected data population, budgets, seed pairing, endpoints, external classes and all 25 artifact choices were fixed before full training and external model scoring. This is a documented protocol, not a claim of prospective registration with an independent registry. Protocol/source hashes are stored in executed configurations.

## Inputs

Use the source-reconstructed label ZIPs from the earlier original-format audit, the original BDD image files under `data/raw/{train,val}/images`, and the published excluded-image list. Do not use the 137 unverified label files or the locally converted test labels. Data images and full annotation archives are not redistributed here. `setup_clean_replication.py` creates a new symlink/label view with no changes to originals.

`download_clean_pretrained.py` records official Ultralytics v8.2.0 COCO artifact URLs and hashes. `download_external_kitti.py` uses official KITTI S3 URLs with TensorFlow Datasets' published archive hashes. Source correspondence and archive identity are distinct from an audit of all pretraining imagery or sequence-level independence.

## Execution recorded for this phase

The scripts execute from the repository root with copies under `review_20260929_clean/`; copy this directory's Python files and `protocol.md` there in a deliberately new reproduction workspace. Keep the training script and protocol bytes immutable after full training begins. Reproduction requires the project's archived checkpoint artifacts for the historical comparison; the new clean student/teacher do not initialize from them.

1. Build the isolated data view and obtain fresh COCO weights.
2. Run `clean_base_train.py --model large --device 1 --smoke`, then its nano/device0 smoke. Run `clean_phase_preflight.py` to check full input counts and all three objectives on tiny batches. Inspect failures before proceeding.
3. Run `queue_clean_training.py` after preflight. It trains two fresh bases, runs all nine continuations and verifies paired inputs and finite saved states. `summarize_clean_confirmatory.py` expects 1,092 batches per epoch, and `test_clean_summary.py` tests effect calculation and rejection of incomplete/mismatched records.
4. Download KITTI inputs; build and test the official local scoring harness, and run the blank-image inference I/O test. `build_local_kitti_evaluator.py` preserves the upstream source and exact patch. It changes local I/O, removes unused 3D/mail/plotting code and improves precision serialization; all native 2D matching/ignore/threshold/precision-envelope blocks are asserted unchanged.
5. `queue_clean_postprocess.py` waits for input checks and all training audits, freezes the 25 actual checkpoints, evaluates every fixed model on KITTI, and performs source-reconstructed BDD class/illumination evaluation for all 11 new final models. No outcome-dependent arm/epoch selection is allowed.

These queues do not write the paper. Scientific analysis, complete result disclosure, manuscript editing and page-by-page visual QA still follow. A completion marker for training or inference alone does not mean submission readiness.
