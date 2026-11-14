Public research outputs 15 November 2026

https://github.com/qkdlsem-maker/yolo-lightweight-optim/releases/tag/plos-one-underlying-outputs-20261115

This release contains 108 distinct inference checkpoints: 84 follow-up evaluations plus 24 non-overlapping historical evaluations. It also includes 36 final raw states, two fixed gradient-input archives, and 33,804 saved native BDD statistics files.
Download Documentation_and_verification_20261115.zip for the current instructions, exact evaluator sources, input provenance and verification reports. The model/statistic ZIP files preserve the bytes authenticated on 13 November. Their archive manifests map each file to its original research path and SHA-256 digest.

Reproducing the native BDD AP calculation
Extract this documentation beside Native_BDD_statistics_108.zip, install NumPy 1.26.4 in a Python 3.12 environment, and run python replay_native_bdd.py. The verified run reproduced all 10,800 class-by-IoU AP values exactly, without loading models, rerunning inference, training, or using a GPU. See REPLAY_INSTRUCTIONS.txt.

Scope and third-party conditions
The two fixed gradient-input archives contain BDD-derived image pixels. Retain the Regents data terms and the accompanying copyright/disclaimer notices. KITTI data retain their provider terms (CC BY-NC-SA 3.0). Ultralytics 8.2.103 retains AGPL-3.0. This release does not replace any third-party terms or create a new blanket project license.
Complete original image/annotation distributions and cached per-image BDD/KITTI prediction JSONs are not included. Provider access routes, version/hash information, inference/evaluation configurations, numerical table/figure data and analysis code are documented in the repository. This is not a claim that every prediction-level analysis has been rerun from the public package alone.

Historical preparation records
Preparation_records_20261113.zip preserves the earlier documentation and audit snapshot. Its draft/private/publication-pending wording, and similar flags inside the four unmodified binary archive manifests, record the state on 13 November. The repository owner authorized publication on 15 November; use this current document and PUBLICATION_STATUS.json for publication status. File hashes and scientific results were not changed.
Public availability does not assert journal acceptance, completion of author declarations, or final submission readiness.
