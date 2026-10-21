# Independent starting pair study 2026 10 21

Author-review draft of a new bounded empirical study following the final desk rejection of the previous IEEE Access manuscript. No submission, acceptance, author approval or completed AI-disclosure review is claimed.

- [Manuscript PDF](Manuscript_Independent_Distillation_20261021.pdf) and [editable Word](Manuscript_Independent_Distillation_20261021.docx)
- [Supplement PDF](Supplement_All_Endpoints_20261021.pdf) and [editable Word](Supplement_All_Endpoints_20261021.docx)
- [All 48 endpoints and all contrasts](../../experiments/independent_replication_20261017/final_all_pair_analysis.json)
- [Training diagnostics](../../experiments/independent_replication_20261017/completed_training_diagnostics_summary.json)
- [Independent arithmetic check](../../experiments/independent_replication_20261017/final_analysis_independent_verification.json)
- [Document verification](../../experiments/independent_replication_20261017/final_document_verification.json)
- [Contribution and journal review](../../experiments/independent_replication_20261017/contribution_and_journal_review.json)

All six 100-epoch base runs, 21 forty-epoch continuations, 48 fixed endpoint evaluations and 48 numerical audits are complete. The three independently fine-tuned pairs are the replication unit. Epoch 40 is primary; epoch 20 is nested in the same schedule. All strengths and both common KITTI variants are retained alongside native metrics. No new training was added to obtain positive results.

The nine-page manuscript and ten-page supplement were exported with native Microsoft Word and all pages visually reviewed. The author-facing Korean review is retained locally. Full prediction and model binaries remain on the research server and are not deposited here. Public JSON provides every endpoint metric and hash-bound audit evidence, not a complete raw-data release. Dataset provenance/reuse, incomplete gradient instrumentation and missing base AMP counters remain limitations. Author review of contributions, conflicts, data access and full AI disclosure is still required.

The published proof chain and every summary can be checked locally without datasets, GPUs or model loading:

```bash
python experiments/independent_replication_20261017/recheck_published_analysis.py
```

This replays arithmetic from saved metric proofs. It does not repeat inference or independently reimplement detector matching. The figures and documents can be generated with `build_research_package.py` after installing matplotlib, numpy and python-docx; native Word/Poppler were used for the released PDF layout.
