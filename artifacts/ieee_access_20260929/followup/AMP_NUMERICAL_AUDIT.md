# Final AMP numerical-state audit

The final read-only audit at 2026-09-29 10:27:21 UTC covers all nine completed epoch-20 training states. It does not alter the training recipe, seeds, budget, raw diagnostics, or final endpoint. No completed run was restarted, extended, or selected by accuracy.

Each run attempts 21,880 minibatches. Final counts are:

| Run | AMP skips | Applied optimizer steps |
|---|---:|---:|
| control_seed0_e20 | 5 | 21875 |
| control_seed1_e20 | 5 | 21875 |
| control_seed2_e20 | 4 | 21876 |
| cwd_seed0_e20 | 4 | 21876 |
| cwd_seed1_e20 | 5 | 21875 |
| cwd_seed2_e20 | 4 | 21876 |
| mse_seed0_e20 | 4 | 21876 |
| mse_seed1_e20 | 6 | 21874 |
| mse_seed2_e20 | 5 | 21875 |

Every EMA counter exactly equals the applied optimizer-step count. All saved student, adapter, EMA, and optimizer tensors are finite; every recorded epoch objective mean is finite. All runner hashes match the per-run configurations.

Some epoch-level median gradient-norm diagnostics are nonfinite because the norm list includes an AMP overflow and numpy.median does not ignore nonfinite members. Every such epoch has a recorded skipped optimizer update. The unchanged runner rejects nonfinite objectives, uses GradScaler to skip overflowed updates, and updates EMA only after an applied optimizer step. These diagnostics therefore do not establish corrupted final states or a frozen student. Raw Python JSON logs may contain NaN; strict JSON readers must handle this diagnostic explicitly instead of inventing a finite replacement.

Report equal **planned minibatch and epoch budgets**, with actual applied counts disclosed. Do not assert zero skips or exactly equal applied updates. The final audit covers all completed states; earlier monitoring snapshots were provisional and are superseded by this report. The recorded 4–6 skipped steps are a limitation of exact update-count matching, not a reason to select or rerun only unfavorable seeds.

Implementation: audit_confirmatory_numerics.py. Evidence: numerical_state_audit.json and final_optimizer_updates.csv. The prespecified protocol and executed runner remain unchanged.
