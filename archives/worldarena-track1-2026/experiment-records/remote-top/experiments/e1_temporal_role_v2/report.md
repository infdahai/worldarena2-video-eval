# E1v2 Five-State Temporal Role Report

Date: 2026-08-17

## Decision

E1v2 is rejected as a Track-1 trajectory candidate. It is not eligible for
official SAM3 evaluation or further training steps.

The five-state temporal-role contract is conceptually valid and fixes the
four-state prototype's bimanual assumptions, but the resulting checkpoint
mainly improves arm detectability rather than matched trajectory ranking.

## Corrected contract

- `LEFT_ONLY`: correct-vs-swap ranking plus right-arm quiet penalty.
- `RIGHT_ONLY`: correct-vs-swap ranking plus left-arm quiet penalty.
- `BOTH`: normal flow matching only.
- `QUIET`: normal flow matching only.
- `AMBIGUOUS`: normal flow matching only.
- Arm identity remains bound to the kinematic action channel, not image side.
- Left/right support may overlap. Overlap is retained in conditioning and is
  excluded only from the inactive-arm penalty to avoid contradictory loss.
- Motion roles use a three-latent centered window and dual thresholds:
  active `0.2`, quiet `0.05`.

## Train-1k role audit

The frozen thresholds were audited on 1,000 cached episodes (20,000 future
latent periods):

| Role | Fraction |
|---|---:|
| LEFT_ONLY | 31.845% |
| RIGHT_ONLY | 26.415% |
| BOTH | 19.645% |
| QUIET | 13.395% |
| AMBIGUOUS | 8.700% |

Thus 28.345% of periods are explicitly protected from single-arm exclusivity
loss by `BOTH` or `AMBIGUOUS`.

## Verification

- Remote focused PyTorch tests: 35 passed.
- Crossed-image arm identity is covered: a kinematic left arm remains left even
  when its projected position lies to the right of the right arm.
- Overlapping support is preserved and is not quiet-penalized.
- Production smoke: 7/7 ranks passed three forward/backward/step cycles.
- Peak allocated: about 7.79 GiB/rank.
- Peak reserved: about 9.43 GiB/rank.
- GPU7 was excluded and retained the colleague-owned process.

## Training

- Parent: support-gated step10,
  SHA256 `521f2bfe3540da74a50e674f337a47095e24f2c3b8b58d8c5a23c67e947d80fc`.
- Adapter-only, raster-only, LR `5e-6`, warmup 2.
- 10 optimizer steps; checkpoints at 5 and 10.
- E1v2 step5 SHA256:
  `0b8061bdb78286f6c3b0faca1ee49d36b695299e0e906ed079c6d1d59431ff82`.

Step5 passed the discovery8 causality gate. Relative to the old E1 prototype,
left/right gradient routing improved from `3.03/2.75` to `3.13/2.99` and
left/right isolation improved from `2.04/2.30` to `2.05/2.36`.

## Matched fast20 result

All videos use identical samples, seeds, 81 frames, 480x640, and 50 sampling
steps. The S1A125 videos were reused.

| Metric | S1A125 | E1v2 step5 | Decision |
|---|---:|---:|---|
| Raw mean DTW | 18.0911 | 17.9704 | 0.67% better, below gate |
| Trajectory score | 0.08087 | 0.06828 | 15.56% worse |
| Paired win rate | - | 50% | below >55% gate |
| Catastrophic/detection proxy | 65% | 25% | better |
| Black video rate | 0% | 0% | pass |

## Post-hoc role-stratified diagnosis

This split is non-official and small-sample; it is for direction finding only.

| Group | n | Raw DTW delta | Win rate | Catastrophic rate |
|---|---:|---:|---:|---:|
| Single-dominant | 5 | -2.48% | 40.0% | 60% -> 0% |
| Bimanual | 6 | +24.41% | 50.0% | 66.7% -> 50% |
| Mixed | 9 | -9.83% | 55.6% | 66.7% -> 22.2% |

The important failure is the bimanual regression. Even though `BOTH` periods
receive no exclusivity loss, updating one shared Adapter on single-arm periods
still changes rollout behavior globally. Therefore this objective must not be
extended or made stronger in the current shared-Adapter architecture.

## Next architecture decision

Stop E1 and Adapter/LoRA hyperparameter search. Preserve the five-state role
schedule as an audit/conditioning contract, but move the next experiment to a
more explicit per-arm geometry interface:

1. independent left/right temporal action streams;
2. kinematic SE(3) identity retained through camera projection;
3. image-space EEF trajectory/flow cues;
4. robot-region trajectory supervision;
5. separate single-arm isolation, bimanual coordination, and temporal leakage
   gates.

Do not add Depth, V-JEPA, GAN, DMD, or a new temporal network in this step.

## Remote evidence

- `/data/di/worldarena2_track1_20260815/experiments/e1_temporal_role_v2/role-audit.train1000.json`
- `/data/di/worldarena2_track1_20260815/experiments/e1_temporal_role_v2/reports/step5.v1.json`
- `/data/di/worldarena2_track1_20260815/eval/s1a125-vs-e1v2-step5-fast20/s1a125-vs-e1v2-step5.arm-proxy.json`
- `/data/di/worldarena2_track1_20260815/experiments/events/e1v2-five-role.fast20-complete.v1.json`
- `/data/di/worldarena2_track1_20260815/experiments/events/e1v2-five-role.group-analysis.v1.json`
