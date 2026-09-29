# Status on 2026-09-29

Completed: archived checkpoint audit, nine full validation evaluations, model-only timing, six fixed-data ten-epoch continuation executions. The fixed-data controls are identical and do not constitute independent repetitions.

Follow-up: nine independent-data twenty-epoch runs (control/MSE/CWD x three seeds), using the protocol in `experiments/protocol_20260929.md`. Results are pending. There is no claim of submission readiness or acceptance.

The student/loader regression checks, class-order regression, CWD formula/gradient check, and actual augmented-input pairing preflight passed. All three preflight arms changed 183 unfrozen student parameter tensors without skipped optimizer steps. The nine-run training queue has started on two GPUs; final results are pending. See DATA_PROVENANCE.md for unresolved data origin and exact-duplicate findings. The complete manuscript has not been published here as a preprint.
