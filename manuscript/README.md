# Reassessing Pruning and Feature Distillation for Lightweight YOLOv8

Submission-prepared research manuscript by **Hyerim Choi and Tae-Kook Kim**, dated 1 October 2026. This repository does not claim journal submission, review or acceptance.

- [Main manuscript PDF](IEEE_Access_Revised_Manuscript.pdf) and [editable Word source](IEEE_Access_Revised_Manuscript.docx): 15 pages, 10 tables, 5 figures and 27 references, using the IEEE Access template.
- [Supplementary methods and complete results PDF](IEEE_Access_Supplementary_Materials.pdf) and [Word source](IEEE_Access_Supplementary_Materials.docx): 5 pages with every fixed external model and every continuation seed.
- [Evidence and code](Evidence_and_Code.zip): numerical records, source versions, protocol and checksum manifest.
- [Versioned model, prediction and per-image statistics release](https://github.com/qkdlsem-maker/yolo-lightweight-optim/releases/tag/ieee-access-reassessment-20260930): three public archives totaling approximately 1.03 GB. [Exact sizes, SHA-256 hashes and download links](release_record.json).
- [Document verification](verification.json): direct table-to-record checks and rendered-page review.

The immutable completed-results commit cited by the paper is `c54d758d431ec3dc07927d2a94c63dc4b3031132`. The final-document commit follows it; the results have not been altered to match the prose. Author-only cover letter and submission/account checklist are provided separately to the authors.

Author correction on 1 October 2026: the funding acknowledgment is removed. Hyerim Choi's supplied ORCID `0009-0006-7311-4107` appears on the first manuscript page. The AI assistance disclosure and all experimental results remain unchanged.

Typography correction on 1 October 2026: references [1]-[27] were verified in first-citation order. English automatic hyphenation reduces stretched word spacing while preserving the justified two-column body. References and the two Code/Data paragraphs containing long URLs are left aligned. All 15 manuscript pages were rendered and visually checked; scientific wording and table values are unchanged.

## Findings and interpretation

The fresh phase excludes 137 source-unverified images, trains nano and large bases for 100 epochs, and completes nine matched 20-epoch control/MSE/CWD continuations. The earlier nine-run study is retained separately. All 25 frozen KITTI evaluations and all 27 reconstructed-label evaluations across the two phases are reported.

Fresh paired internal AP50:95 differences average -0.03193 and -0.07841 percentage points for MSE and CWD. On the two-class moderate KITTI AP_R40 summary, the corresponding differences are +0.03705 and +0.14154 points, positive for every fresh seed. All fresh continuations remain below fresh nano externally. The dataset, class set and scoring rules change jointly, so neither a universal KD ranking nor a causal domain-shift mechanism is established.

The source-unverified labels remain unidentified; their exposure is eliminated for the new models. The legacy 2018 mirror is not authenticated as an official Detection 2020 archive. BDD validation is reused, each phase shares one trained starting pair, and KITTI uses the publicly labeled training partition as this study's external holdout, not the official hidden-test leaderboard. Historical pruning budgets, shared-desktop runtime and unaudited COCO image overlap remain limitations. See the full discussion before reusing these results.

## Reproduction

Use the pinned source/environment and fresh output directories specified in the experiment protocols. Obtain BDD and KITTI images/labels separately under their dataset terms. The inference checkpoint manifest maps original frozen model hashes to exported files with identical tensors, architectures and class names; only training-location/history metadata was removed. All three release archives were publicly downloaded and hash-verified. For archived pruned models, retain the repository's `scripts/c2f_v2_utils.py` import module. The evidence and release archives include further manifests and instructions. Source and model license notices continue to apply.
