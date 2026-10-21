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
