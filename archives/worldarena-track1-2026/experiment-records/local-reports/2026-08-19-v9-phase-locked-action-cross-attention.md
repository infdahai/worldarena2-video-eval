# Wan v9 Phase-Locked Action Cross-Attention — GPU6 Bounded Run Report

Date: 2026-08-18 to 2026-08-19 (Asia/Shanghai)  
Remote source: `/home/huazhi/nlh/baseline`  
Artifact root: `/data/di/worldarena2_track1_20260815`  
Run root: `/data/di/worldarena2_track1_20260815/runs/v9-phase-locked-action-cross-attention/gpu6`

## Executive decision

**Stop at step100. Do not start step250 or Phase T.**

The run was numerically and operationally healthy, but it failed the action-separation mechanism gate. Correct action did not consistently beat reverse, either temporal shift, or arm swap on the fixed matched audit20. The ranking loss stayed approximately at `ln(2)`, so the run did not learn a robust preference for correct action content.

This result rejects this exact v9 recipe as a promotion candidate. It does not support RGB decoding or trajectory evaluation for this checkpoint.

## Architecture actually trained

- Parent: pinned `clean-gated-step10`.
- Six Wan blocks: `8–13`.
- Four action sub-tokens per arm and interval:
  - translation;
  - SO(3) log-map rotation;
  - compact image motion;
  - gripper state.
- Strict phase lock: latent `t` reads only interval `t-1`; latent0 is zero.
- Left/right head-bank isolation; missing arm output is zero.
- Trainable families:
  - native Wan q/k/v/o in the selected blocks;
  - action tokenizer and cross-attention;
  - zero-initialized channel gates.
- Trainable parameters: `257,423,488`.
- Objective: correct weighted FM plus one rotating counterfactual ranking negative per step.
- Final checkpoint calibration: `lambda_cf=2.4163222312927246`, `tau=0.1`.

## Data and leakage contract

- Clean manifest SHA256: `190a45509cc283b8c3415861d19013f34897b804f35a37389d62e4b08b8e80af`.
- Optimizer pool: 1,765 samples.
- Audit set: fixed 20 samples, excluded from optimizer replay.
- Dev-fast20 remains evaluator-only and excluded from training.
- Replay SHA256: `92fbc5f463a8f831898301c8dad40d0fae2ebcedd20f50117fd34c504ba49800`.
- Audit SHA256: `6461e395bf5bd716d4b9db3d50a3aba4d6144fe32a6eb20a02bc1d7926057e60`.
- Transition normalization SHA256: `64fe000724204652a58b109dd360b3d6698102933386d2d0f9b2d3b72eccfc3e`.
- Final transition receipt SHA256: `8f53535b169482b9fb88942e1b781fb9c80ff21ebb05acdfd50d452960fcc164`.
- Final source closure SHA256: `23982bcf128317f12ea3297cea665eaf6105d67cdb6068d37cae1fee822799c0`.

Transition cache completion:

- optimizer: `8,825 / 8,825` files (`1765 × 5`);
- audit: `100 / 100` files (`20 × 5`);
- variants: correct, reverse, shift+1, shift-1, swap.

## Verification and smoke

- Remote focused Torch suite: `48 passed`, zero skipped.
- Current-source preflight passed.
- Current-source three-iteration production smoke passed.
- Resolution/shape contract: 81 frames, 480×640, cached latent + text; no T5/VAE in the training hot path.
- GPU: only GPU6 was exposed to the process.
- Peak allocated: `15,985,696,256` bytes = 14.89 GiB.
- Peak reserved: `16,661,872,640` bytes = 15.52 GiB.
- Limit: 22 GiB.
- All trainable gradient families were nonzero: native q/k/v/o, tokenizer, cross q/k/v/o, channel gates.
- End state: GPU6 at 6 MiB and 0% utilization; no v9 compute process remains.

## Training health

- Completed steps: `1–100`.
- Mean step time: `6.4461 s`.
- Mean FM loss: `0.35593`.
- Mean ranking loss: `0.693063` (`ln(2)=0.693147`).
- Mean per-step margin: `1.785e-05`.
- Maximum observed gradient norm: `4.99753`.
- No OOM, NaN, optimizer failure, scheduler drift, or topology drift was observed.

The run therefore failed for semantic efficacy, not for numerical instability.

## Matched audit results

### Step25 health gate

Decision: **continue=true**. This was only a health gate, not a mechanism promotion.

| Negative | Correct wins | Win rate | Mean margin |
|---|---:|---:|---:|
| reverse | 11/20 | 55% | `+4.820e-05` |
| shift+1 | 10/20 | 50% | `-2.640e-05` |
| shift-1 | 10/20 | 50% | `+5.032e-05` |
| swap | 6/20 | 30% | `-3.216e-05` |

- FM regression versus frozen parent: `-0.00418%` approximately; no FM regression.
- Routing retention: `100%`.

### Step100 hard mechanism gate

Decision: **continue=false, pass=false**.

| Negative | Correct wins | Win rate | Mean margin |
|---|---:|---:|---:|
| reverse | 8/20 | 40% | `-1.642e-05` |
| shift+1 | 10/20 | 50% | `-4.098e-05` |
| shift-1 | 9/20 | 45% | `+2.546e-06` |
| swap | 9/20 | 45% | `-5.875e-06` |

Hard-gate failure reasons:

- borderline shift continuation rule failed;
- reverse mean margin was not positive and wins were below 12/20;
- shift+1 mean margin was not positive and wins were below 12/20;
- shift-1 wins were below 12/20;
- swap mean margin was not positive and wins were below 12/20.

Guardrails remained healthy:

- FM regression: `-0.01439%` approximately;
- routing retention: `100%`;
- gate-enabled FM was no worse than the control.

## Interpretation

The strict four-token phase lock solved the architectural ambiguity of a single K/V token, but this training recipe still did not produce stable action semantics. More training did not improve the matched separation curve: reverse and shift performance weakened from step25 to step100, while swap recovered only from 30% to 45%, still below chance-level promotion evidence.

The key pattern is:

1. gradients exist;
2. FM remains healthy;
3. routing remains intact;
4. correct-vs-wrong ranking remains near random.

So the limiting issue is not memory, optimizer execution, or basic branch connectivity. The current counterfactual energy signal is too weak/noisy relative to the generative objective, or the selected trainable interface still does not make action correctness identifiable at the final visual prediction boundary.

## Important limitations discovered

### 1. Probe aggregation invalidity

Only 17/20 audit samples have finite position and velocity probe values. Three samples return infinity. The current reducer propagates these into:

- `position_mean = Infinity`;
- `velocity_mean = Infinity`;
- improvement values = `NaN`.

Therefore this run provides **no valid claim of position or velocity improvement**. The evaluator must report finite coverage and aggregate only the predeclared observable subset, while retaining invalid-count penalties separately.

### 2. Fresh initialization is not globally seeded

Replay noise and timestep seeds are fixed, but the trainer does not set a global model-initialization seed before creating the fresh v9 branch. Independent fresh launches produced different lambda calibrations (`1.8566`, `2.4163`, and earlier values). The completed checkpoint is internally bound to `lambda_cf=2.4163222312927246`, but the experiment is not exactly reproducible from source + replay alone.

Before any repeat, a source-pinned initialization seed must be applied before model/branch construction and recorded in checkpoint lineage.

These limitations weaken reproducibility and trajectory-probe interpretation, but they do not reverse the observed action-separation failure on this completed run.

## Bugs repaired during execution

1. Audit samples originally had only `correct.npz`; audit required four negative variants. The cache contract now creates and validates all five variants for optimizer and audit samples.
2. Cache rerun originally risked rebuilding all entries. Valid caches are now provenance-checked and reused; only the missing 80 audit negatives were created.
3. `train100` originally resumed raw step25 and could bypass audit25. It now requires `step-000025-gated.pt`, and the trainer independently requires `gates.step25.continue=true`.
4. The source closure changed after these fixes. The old step10/25 files were recoverably archived, and step1–25 was rerun under the final source lineage instead of falsifying checkpoint provenance.

Archived pre-fix artifacts:

`/data/di/worldarena2_track1_20260815/runs/v9-phase-locked-action-cross-attention/gpu6/pre-audit-cache-fix-20260818`

## Final artifacts

- Step25 gated checkpoint: `step-000025-gated.pt`  
  SHA256: `28e14cc083e600a599286e2ad09c19cf1ff276a7e14bdbd8173670ba73371dc0`
- Step100 raw checkpoint: `step-000100.pt`  
  SHA256: `0e832ee36a6aaa2e03168a55aa6da53db3fb283eacf600973911bb27cd91c41b`
- Step100 gated checkpoint: `step-000100-gated.pt`  
  SHA256: `d030f00a932ef1b55f3a07c7c03beebd8cd446a5c754857ba28e88c0976bcb59`
- Step25 report: `audit-025.json`.
- Step100 report: `audit-100.json`.
- Training log: `training.jsonl`.
- Current-source preflight: `preflight.json`.
- Current-source smoke: `production-smoke.json`.

## Recommendation

Do not tune v9 longer and do not add trajectory supervision on top of this checkpoint. First fix only the two experiment-contract defects:

1. deterministic global initialization/calibration;
2. finite observable-subset probe aggregation.

Then make one decision: either run a single exact-seed confirmation of v9 to rule out initialization variance, or accept the present failure and move to a stronger objective that directly supervises decoded gripper trajectory / partially unfreezes the visual prediction path. Do not start a broad hyperparameter sweep.
