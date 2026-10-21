# Learning-rate interaction and new starting-pair replication

This extension was fixed on 2026-10-21 after the original three-pair study was completed and inspected. It is not retroactive preregistration. Its purpose is to test whether the continuation optimizer learning rate modifies the paired MSE/CWD effect, with three additional independently fine-tuned starting pairs. No new result is claimed yet.

- Preserve all original 48 endpoints and all original strengths.
- Add seeds 404/505/606, each nano/large for 100 epochs: six new bases.
- Use fixed initial learning rates 0.0001 and 0.001, detection-only control and MSE/CWD at the original grid's log-midpoint coefficient 0.3. This coefficient is not the observed best condition.
- Reuse the nine completed low-rate paths for seeds 101/202/303. Run their nine high-rate paths and 18 paths for the new pairs: 27 new 40-epoch continuations.
- Report within-pair learning-rate-by-distillation interactions at epoch40. Keep all epoch20 endpoints and old/new cohorts separate as well as the full six-pair descriptive summary.
- Fixed final-state gradient probes and controlled rescoring of cached KITTI predictions will address gradient alignment and metric-definition sensitivity. These are diagnostics, not causal proof of teacher quality or a promise of improved AP.
- Maximum 220 allocated GPU-hours across two GPUs, maximum ten days, no favorable-result stopping or added search. See protocol.json for exact scope, budgets and stop rules.

Native AMP update counters are observational. The initial 128-image nano smoke fixture had only two optimizer attempts and no applied updates because both were skipped. That failed preflight and all four original smoke outputs are retained. One separately recorded 2,048-image fixture per architecture/seed checks actual raw-parameter updates; native training/scaler/EMA rules are unchanged. No full training starts before the base preflight and public protocol-byte verification pass. Continuations require their own separate preflight; the base queue cannot launch them.

The legacy data provenance, reused evaluation populations, one detector family, previous IEEE Access final desk rejection and pending author/AI-disclosure review remain explicit. New results may support or weaken the proposed explanation. No paper acceptance, new compression algorithm or inference speedup is promised.

## Execution evidence (2026-10-21)

The base preflight passed after the separately recorded fixed-length AMP fixture. Exact native optimizer/scaler/EMA parity was checked, including an injected skipped optimizer step. All four extended native fixtures applied real updates, changed raw model parameters and retained finite final states. The failed short fixture remains recorded.

Full base training has started. The initial simultaneous launch encountered a native shared-label-cache temporary-file rename race in seed505 before any input batch or optimizer update. The failed config/args/log were archived, full train/validation caches were warmed serially and the identical source/seed/settings were relaunched. Seed404 continues unchanged; the recovery worker owns only the five other bases and uses GPU0 only after the original worker exits. A separate guard enforces the original disk, GPU-hour and wall-clock caps. No scientific result is yet claimed; continuation execution still requires its own passed preflight and completed-base audit.

[protocol_interpretation.json](protocol_interpretation.json) makes the explicit new factorial take precedence over copied historical roster/count fields in the original protocol. It preserves the protocol bytes and design, documents the duplicated-text ambiguity, and was recorded before any new completed base or continuation result. Read both files together.

The `technical_failures` records preserve the short AMP fixture, cache race and an empty-file transport error. No failure is excluded to improve a reported result. Raw training states and larger operational logs remain on the research server; the JSON evidence release is not a claim that complete model/prediction binaries are publicly deposited.
