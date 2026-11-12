Publication reproduction supplement prepared 12 November 2026

This folder accompanies the fixed learning-rate follow-up by Hyerim Choi and
Tae-Kook Kim. It preserves the numerical data already verified and provides
the revised A/B panel plotting code used in the PLOS ONE preparation.
This release does not add new experiments or establish submission acceptance.

Required public evidence (existing immutable commit):
https://github.com/qkdlsem-maker/yolo-lightweight-optim/tree/407f4b3d3a52e72bda5022a61b5df6dbd38473e5/experiments/lr_factorial_20261021/completed_followup_20261111
Download evidence/final030_text_evidence_20261111.zip and its INDEX.json.
Extract into a new empty directory. INDEX.json must be in the extraction root.
The expected INDEX SHA-256 is
c50308c238bd2fb774af6c9e3dd15cd5941ffb393dc00a45f10706ce61ac75d7.

Run from this folder with Python 3.12 or later:
python verify_release.py S1_Dataset.zip /path/to/extracted_evidence
The verifier checks dataset bytes, all bound evidence files and the original
arithmetic/identity checks. It does not execute a model or create/change the
research evidence. Standard library suffices for the numerical verification.

For figures, install the versions in PLOT_REQUIREMENTS.txt in your environment.
An installed Arial font is required to match the supplied layout; no fonts are
redistributed. Font substitution can change pixel output without changing data.
python build_result_figures.py /path/to/extracted_evidence /path/to/new_figures
The builder requires a successful INDEPENDENT_LOCAL_VERIFICATION.json beside
INDEX.json. To create one, run the original verify_downloaded_results.py on a
fresh extracted evidence folder that does not already contain that report.
The figure output directory must not already exist. Five diagnostic plots are
produced; only the first four are the main manuscript figures. The fifth plot
is an additional diagnostic; the numerical records remain authoritative.
REVIEWED_FIGURE_MANIFEST.json records the previously reviewed plot bytes.
The portable builder removes one machine-specific Python path; calculations
are unchanged. PDF hashes can vary with embedded creation-time metadata.

CLAIM_DATA_MAP.json links each table/figure to data. DATA_DICTIONARY.txt defines
units, missing values, paired units and limitations. S1_Dataset.zip contains
all numerical tables and unchanged original-study numerical data.

Coverage: scalar/statistical and plot reproduction is supplied. Original
third-party images and labels, model checkpoints, raw final model/adaptor
states and prediction binaries are not in this supplement. Native inference
and prediction-level rescoring require those inputs, their actual versions,
and the separately recorded scientific environment. These remaining inputs
must have verified access before claiming fully open reproduction.
GitHub availability is not a grant of third-party rights. A repository-wide
license was not found on 12 November 2026; rights-holder license approval is
pending. No blanket license is asserted by this supplement.

AI assistance: OpenAI Codex assisted with workflow and validation code,
plot preparation and drafting. The corresponding disclosure is in the draft
manuscript. The authors remain responsible for review and approval.
