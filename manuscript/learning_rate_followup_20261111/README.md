# Learning rate follow up manuscript and supplement

These author-review drafts report the completed fixed follow-up on learning rate sensitivity after detector fine tuning. They include every prespecified evaluation endpoint and preserve the original strength-grid study as a separate companion.

- Manuscript: [Word](Manuscript_Learning_Rate_Followup.docx) and [PDF](Manuscript_Learning_Rate_Followup.pdf)
- Supplement: [Word](Supplement_All_Fixed_Followup_Results.docx) and [PDF](Supplement_All_Fixed_Followup_Results.pdf)
- [Five figures in PNG and PDF](figures), [figure identities](FIGURE_MANIFEST.json) and [document page review](DOCUMENT_VERIFICATION.json)
- [Complete numerical records, interpretation and verification instructions](../../experiments/lr_factorial_20261021/completed_followup_20261111/README.md)
- [Unchanged original strength-grid manuscript and supplement](../independent_study_20261021/README.md)

## Primary learning rate interactions

Values are AP percentage points for `(KD − control at 0.001) − (KD − control at 0.0001)` at epoch 40 on native ten-class BDD AP50:95.

| Method | Six-pair mean | Sample SD | Descriptive 95% t interval | Historical three mean | New three mean |
| --- | --- | --- | --- | --- | --- |
| MSE | −0.0249 | 0.0263 | −0.0525 to +0.0026 | −0.0106 | −0.0392 |
| CWD | −0.0017 | 0.0674 | −0.0724 to +0.0691 | +0.0177 | −0.0210 |

An interaction alone does not establish improvement over control. All pair values, simple effects and historical/new strata are reported, along with small-sample uncertainty, shared pretraining and datasets, previously examined evaluation populations, legacy annotation provenance and the unisolated server-kernel transition.

The calculations and rendered-page checks are complete; authorship contributions, competing interests, final interpretation and venue-specific AI disclosure still require the listed authors' review. OpenAI Codex assistance is disclosed. No journal submission or author approval is asserted. This release provides scientific text and code, while durable access to full model and prediction binaries remains an author decision.
