# v4 clean data-scale preparation — 2026-08-17

## Hard contract

- Dataset leakage is a release-blocking failure.
- Training source, cached manifest, every trainable parent ancestor, and every
  locally available evaluation identity are revalidated before Wan is loaded.
- The currently available evaluation identity set is `dev-fast-20`. Hidden
  blind/test identities are not claimed verified until their manifests exist.
- Previous `v2-1000` and v4 results are contaminated diagnostics because 13 of
  20 `dev-fast-20` episodes occur in the old training manifest.

## Clean datasets

| split | episodes | source SHA-256 | dev-fast20 overlap |
|---|---:|---|---:|
| clean-1000 | 1000 | `fc54f851099ea213431efcb64a2fcbfd5c01335e18404f685768f00ece89b289` | 0 |
| clean-1785 | 1785 | `6bcc85830fa2c6ccb45b2c3f7f593102674efce6a76a7a6e5d8871fcccdbae35` | 0 |

`clean-1000` is a strict subset of `clean-1785`. It preserves 987 old rows and
replaces all 13 leaked rows. The large split excludes 15 of 1800 eligible rows.

## Parent lineage

- Formal parent: `s1A_branch_0125_ws7/step-000125.pt`.
- Parent SHA-256:
  `59b0de933abc5229cd81ad8507b3f7ab550f996d29038496ae8a58e373fb6df3`.
- Parent data: four-task Stage-1 manifest, 200 episodes, zero overlap with
  `dev-fast-20`.
- The old gated step10 parent is rejected because its training lineage resolves
  to the leaked `v2-1000.cached.jsonl`.

## Cache gate

The shared `clean-1785` cache is complete and supports both experiments:

- action conditions: 1785/1785;
- VAE latents: 1785/1785;
- T5 contexts: 1785/1785;
- cache binding SHA-256:
  `f4fdc8c94be6234b297eace3a71f2dbc24faeba4b51f1ccbc2e6a4ad72e183ed`;
- validated URDF SHA-256:
  `097c59fb19a7b482249c6097df8319586ea7cfd268c015516f103b289a7e761a`.

## Comparison protocols

- Target mix: single-dominant 45%, bimanual-heavy 30%, mixed 15%, quiet 10%.
- Quiet means at least 30% of latent action windows are BOTH_QUIET. The
  threshold leaves enough independent episodes for the 10% lane.
- Sampling is without replacement inside an effective epoch. A lane that
  cannot meet its quota fails closed.
- A balanced epoch is the largest without-replacement draw satisfying the
  target mix. This is 856 for clean-1000 and 1323 for clean-1785.
- The currently running `clean-1785` run is retained as a **proportional
  training-budget / source-coverage-LR** result. Its checkpoint grid is aligned
  by balanced epochs, while its existing LR schedule is a function of source
  dataset coverage. It must not be called a pure data-scale result.
- Its checkpoints are aligned at 5/10/15 balanced epochs:
  - clean-1000: steps 612 / 1223 / 1835;
  - clean-1785: steps 945 / 1890 / 2835.
- A later **fixed-exposure confirmation** is registered separately. Both source
  scales will use steps 612 / 1223 / 1835, hence exactly 4,284 / 8,561 /
  12,845 samples seen, and a shared samples-seen warmup/cosine LR schedule.
  It is launched only if the proportional-budget run wins matched video
  evaluation.
- Every checkpoint stores actual per-lane counts/fractions, `samples_seen`,
  `unique_episode_seen`, and `effective_epochs`.

## Seven-GPU smoke

The clean-1000 production smoke passed on GPU0–6.

- Peak allocated: 8,161–8,163 MiB per rank.
- Peak reserved: 9,871–9,903 MiB per rank.
- Three post-warmup steps: 4.00–4.38 seconds each.
- All ranks are below the 22 GiB hard limit; GPU7 was not used.

## clean-1000 completion

Remote formal tests and the seven-GPU production smoke passed. `clean-1000`
completed from the clean S1A125 parent at step 1835 (15.006 balanced effective
epochs), with all seven training ranks released afterward.

| checkpoint | effective epochs | samples seen | unique episodes | checkpoint SHA-256 |
|---|---:|---:|---:|---|
| 612 | 5.005 | 4,284 | 998 | `584c8fb9eb4ab955246b4719dcad9cd0204293528b08e7fd8ab6376eac1e49b1` |
| 1223 | 10.001 | 8,561 | 1,000 | `5f7674f3acaffd159a961d4e38dc4dcaf203e1f0bdfc4f6fecc0883e160761b6` |
| 1835 | 15.006 | 12,845 | 1,000 | `fbfcc15195d11dbf07adcbe2fc727d2c22e4918d1e6a0536dbff37deadb09446` |

The final frozen action audit remains healthy: `correct_error=1.0165` is lower
than `swap_error=1.0262`; neither arm stream collapsed; coordination residual
ratio is `0.0954`; and both-active coverage is `0.0816`. The 0.95% correct-vs-
swap separation is an eligibility signal only: it shows no inversion/collapse,
not strong action-following proof. The actual completed
sampling mix is 44.97% single-dominant, 30.03% bimanual-heavy, 14.95% mixed,
and 10.05% quiet, matching the contract.

`bimanual-heavy` is an episode-level sampling stratum, whereas `both-active`
is the fraction of action windows where both commanded arms are simultaneously
active. They are intentionally different statistics; 30% and 8.16% therefore
do not conflict.

These are training-health and causality eligibility signals, **not a
Trajectory win claim**. Checkpoint selection remains blocked on matched
clean-lineage video evaluation.

Remaining sequence:

1. run an independent clean-1785 seven-GPU smoke, then train it from the same
   clean parent;
2. evaluate only clean-lineage checkpoints on dev-fast20;
3. retain all checkpoints, exposure records, audit logs, and evaluator output
   in the experiment registry for later analysis.
