# Wan v8 Direct Action Band — bounded mechanism run

## Executive decision

The v8 Phase-M run completed its approved 100 optimizer steps on physical
GPU6. Production smoke passed. The held-out audit20 **failed the mandatory
step100 mechanism gate**, solely and decisively on temporal shift separation.
Training therefore stopped at step100. Phase T, RGB decoding, and matched
fast20 were not launched.

This is a useful negative result rather than a runtime failure: direct native
Q/K/V/O adaptation learned strong reverse and swap separation, but did not
learn a stable `+1/-1` latent-step temporal preference.

## Frozen experiment contract

- Parent: `clean-gated-step10`
- Parent SHA256: `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`
- Dataset: clean-1785 cached latents/text/actions
- Optimizer pool: 1,765 identities after held-out audit20 removal
- Replay: one rank, 250 pre-generated records; only records 1–100 consumed
- Audit: fixed 20 probe-heldout, multi-task identities; overlap with replay 0
- Trainable Wan band: blocks 8–13
- Trainable tensors: native self-attention Q/K/V/O plus six FP32 channel gates
- Trainable parameters: 226,584,576
- Frozen: raster parent, all other Wan blocks/families, probe, T5, VAE
- Phase-M loss: correct weighted FM plus sequential correct-vs-one-wrong ranking
- Wrong cycle: reverse, shift ±1, swap
- Native LR: `1e-6`; gate LR: `5e-5`; 25-step warmup and cosine schedule
- Calibrated `lambda_cf`: `0.1100865660`
- Calibration QKVO norms: FM `0.0461398913`, CF `0.2095618608`

## Data readiness and isolation

- Correct SE(3) cache: 1,785 / 1,785
- Complete negative caches: 1,785 / 1,785 for reverse, shift+1,
  shift-1, and swap
- audit20: 20 / 20
- replay/audit collision: 0
- optimizer/dev-fast20 collision: 0 by frozen v8 data receipt
- Official test: unavailable and not accessed
- Persistent outputs: `/data/di/worldarena2_track1_20260815` only

## Production smoke

Three complete correct/wrong iterations ran at production latent/video
geometry with backward, optimizer step, and zero-grad.

| Metric | Result | Gate |
|---|---:|---:|
| Peak allocated | 15,494,590,976 bytes (14.43 GiB) | <22 GiB |
| Peak reserved | 16,141,778,944 bytes (15.03 GiB) | <22 GiB |
| Step time | 6.58–6.60 s | finite/stable |
| OOM / NaN / Inf | 0 | 0 |
| Smoke decision | PASS | PASS |

Smoke receipt SHA256:
`90a6338cf1a25bcf8004349d7206e573c8ae6c381d2ee27f7fbc0ca64b0d1525`.

## Phase-M training curve

- Completed updates: 100 / 100
- Mean FM, steps 1–25: `0.437458`
- Mean FM, steps 76–100: `0.267854`
- Mean FM, all steps: `0.357227`
- No OOM, NaN, Inf, missing trainable gradient, or topology drift

Training-replay margins are diagnostic only, not the promotion decision:

| Negative | Positive-margin steps | Mean margin | Last-10 family mean |
|---|---:|---:|---:|
| reverse | 29 / 34 | +0.0181035 | +0.0197865 |
| shift | 22 / 33 | +0.0000380 | +0.0000646 |
| swap | 27 / 33 | +0.0061336 | +0.0090873 |

The curve already predicts the held-out failure: shift remains effectively at
zero while reverse and swap become separable.

## Fixed held-out audit20

Promotion threshold was at least 11/20 wins and positive mean margin for each
negative family.

| Negative | Wins | Mean margin | Result |
|---|---:|---:|---|
| reverse | 16 / 20 | +0.00563915 | pass |
| hard shift | 5 / 20 | -0.00016099 | **fail** |
| swap | 17 / 20 | +0.00544123 | pass |

Other evidence:

- Routing retention: `1.0`
- Correct FM regression vs fresh parent: `-0.00007996` (improved)
- Gate-zero reverse: 16/20, `+0.00567876`
- Gate-zero shift: 6/20, `-0.00019996`
- Gate-zero swap: 17/20, `+0.00548013`

The SE(3) channel gates therefore contributed little to the separation at
step100; most reverse/swap behavior came from the adapted native Q/K/V/O band.
The frozen probe returned non-finite parent/candidate position and velocity
aggregates on some audit rows. They are recorded as unavailable and must be
fail-closed in future audit summaries. They do not change this run's decision,
because hard shift already fails both the win and margin gates.

## Gate decision

Step100: **FAIL**

Reasons:

1. `shift_wins_below_11` (5/20)
2. `shift_margin_not_positive` (-0.00016099)
3. position/velocity aggregate contains non-finite rows; no positive
   trajectory claim is permitted

Consequences enforced:

- no Phase-T optimizer update;
- no trajectory-lambda calibration;
- no step150/200/250;
- no RGB decode;
- no matched fast20;
- GPU6 released.

## Artifacts

| Artifact | Size | SHA256 |
|---|---:|---|
| `step-000010.pt` | 1,359,685,921 | `e08fb51ed73f6fd00274364f8772c5cfd8c3e12e371f890cabc38669434e1dac` |
| `step-000025.pt` | 1,359,685,921 | `79d9b8ea42c0b748c3a5ceb34dea345aebbb4eb05c12ccd68999c7857358d97b` |
| `step-000050.pt` | 1,359,686,113 | `d54e43ea48a5026293f96e2d00c2016d440de1242b3cf88e138c7c452a3877a4` |
| `step-000100.pt` | 1,359,686,113 | `e269667e45c65236a6eb6ba9c88395af1acf3bec1d954803f1944b3fa57609e6` |
| failed-gate receipt checkpoint | 1,359,686,101 | `2d4382235ae4147244936b44c72b1fcd3224993f609e95664bbe3818b06b4692` |
| `training.jsonl` | 19,064 | `f3fc3869d479f8e5e6beef6746cecd8e8a37bab04e93ec2dc5f268368e2391ee` |
| `audit-100.json` | 2,022 | `0eb0821c2dde783f96754b99a69b5034f77367b5c8adb211db2a554451590cfe` |

Run directory:
`/data/di/worldarena2_track1_20260815/runs/v8-direct-action-band/mechanism-gpu6`

## Implementation/runtime issues encountered

Three fail-fast integration issues were found before or during audit and were
fixed without changing the trained Phase-M math:

1. audit identities were compared in file order instead of as a set;
2. a passed smoke branch lacked an explicit return and fell into train-mode
   argument checking after writing its valid receipt;
3. the v7 frozen-probe metric adapter required `target_raster`, `loss`, and
   `support_energy` aliases in the v8 audit result.

Training steps 1–100 used one consistent implementation lineage. Audit-only
fixes were applied after the immutable step100 checkpoint was written.

## Final technical interpretation

v8 rejects the hypothesis that a six-block native Q/K/V/O band plus the
current weighted-FM/counterfactual objective is sufficient for complete
action semantics. It can recognize direction reversal and arm identity, but
not one-latent-step temporal displacement. More training is not authorized:
the weak signal is structural/objective-alignment evidence, not an unfinished
curve.

The next design should target temporal alignment explicitly rather than widen
this run or continue it past its failed gate.
