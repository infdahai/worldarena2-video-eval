# Wan-Action v7 SE(3) Mechanism Probe Design

## Decision

The first v7 experiment is a 24-hour mechanism probe, not a six-block
competition candidate. It tests one hypothesis:

> Does inserting known per-arm relative SE(3) transforms into Wan
> self-attention make the frozen clean parent respond more correctly to the
> commanded trajectory than to swapped, reversed, or shifted trajectories?

Stage A changes the action-to-model interface while holding the parent,
training data, selected blocks, loss, and replay lineage as close as possible
to v6. A six-block trainable geometry branch is a gated Stage B and must not
start automatically.

## Fixed boundaries

- Formal artifacts live under `/data/di/worldarena2_track1_20260815`.
- Remote source lives at `/home/huazhi/nlh/baseline`; Wan source lives at
  `/home/huazhi/nlh/Wan2.2`.
- Training uses GPU0-6 only. GPU7 remains excluded.
- The parent is the immutable clean-gated-step10 checkpoint with SHA256
  `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- Wan, the existing support-gated Action Adapter, Wan Q/K/V/O projections,
  and the frozen gripper probe remain frozen.
- Stage A uses weighted flow matching only. It adds no trajectory loss,
  gripper bias, QKV LoRA, Depth, V-JEPA, DMD, GAN, causal action mask, or new
  training data.
- T5 and VAE remain outside the training hot path.
- The video contract remains 81 RGB frames at 480x640 and 21 latent frames.
- Stage A modifies blocks 8, 16, and 24 only.

## Data and zero-leakage contract

Stage A uses the same clean-1000 eligible training pool as v6. It does not use
clean-1785 so that data scale cannot be mistaken for an attention-interface
effect.

With seven ranks and micro-batch one, the maximum exposure is:

| Checkpoint | Episode draws | Approximate clean-1000 epochs |
|---|---:|---:|
| step 10 | 70 | 0.070 |
| step 25 | 175 | 0.175 |
| step 50 | 350 | 0.350 |

The sample, noise, timestep, and dropout replay should match v6 wherever its
immutable replay contract is compatible. Any unavoidable replay difference
must be recorded and prevents a claim of an exact matched v6 comparison.

The Stage A training manifest, discovery manifest, dev-fast20 manifest, and
official test manifest must be hash-bound and pairwise checked before launch.
Discovery episodes are not legal training rows. Dev-fast20 and official test
episodes are not legal training or discovery rows. A failed or missing leakage
receipt blocks smoke and training.

Stage B may use clean-1785 only after Stage A promotion and a new explicit
approval. It receives a new replay and exposure-based checkpoint contract.

## SE(3) condition contract

### Source

The SE(3) condition is generated from the original action end-effector poses
or the already validated FK source. It is not reconstructed from the
standardized 11D pose cache. The 11D cache contains camera-relative and
normalized features and is not the authoritative rigid-transform source.

The v7 cache stores, per sample:

- `arm_transform`: float32 `(2, 21, 4, 4)`;
- `arm_present`: bool `(2, 21)`;
- the source episode, action/FK source, temporal-packing, and cache hashes;
- the selected anchor identity and motion scale;
- a schema version dedicated to the v7 SE(3) condition.

`arm_present` describes whether the action/EEF stream is valid. It is never
derived from SAM3 visibility or RGB observability. An arm outside the camera
can remain present. RGB observability remains an evaluator/supervision mask
only.

### Temporal packing

The 81 source poses use the existing explicit 81-to-21 contract:

- latent 0 receives frame 0;
- latent 1 receives frames 1-4;
- ...;
- latent 20 receives frames 77-80.

Rigid poses are sampled with translation interpolation and quaternion SLERP,
not by averaging matrices or quaternion components. Every visual token is
mapped to its latent frame using Wan's time-major, row-major, column-major
flattening contract.

### Shared reference frame

Let `G_t^k` be the homogeneous end-effector transform for arm `k`. The
episode anchor is selected deterministically:

1. the left pose at latent frame 0 when valid;
2. otherwise the right pose at latent frame 0 when valid;
3. otherwise identity, with both arm-presence masks false and the entire
   geometry branch forced to zero.

The shared relative frame is:

`Gbar_t^k = inverse(G_ref) @ G_t^k`.

This anchor selection is stored in the cache. It makes the condition invariant
to a common left-multiplication of all source poses by a global SE(3)
transform, up to the declared numeric tolerance.

### Motion normalization

For valid arms, compute:

`motion_scale = max_{k,t} ||pbar_t^k - pbar_0^k||_2`.

If `motion_scale <= epsilon`, use scale one. Otherwise divide the translation
component of every `Gbar_t^k` by the shared episode scale. Both arms therefore
use the same scale. Rotation is not rescaled. Non-finite values, invalid
homogeneous rows, or non-rotation matrices fail closed.

The action matrix consumed by geometric attention is the inverse of the
normalized relative transform. Missing-arm entries are stored as identity for
numeric safety, but their outputs are also zeroed by `arm_present`.

## Geometry attention operator

Wan2.2-TI2V-5B has width 3072 and 24 attention heads, hence head dimension
128. Stage A asserts these values at runtime and separately requires
`head_dim % 4 == 0`.

For arm `k`, latent frame `t`, and head dimension `d_h`, construct the group
representation conceptually as:

`D_t^k = I_(d_h/4) kron A_t^k`.

Production code must not materialize a 128x128 matrix for every token.
Instead, it reshapes each head channel from `128` to `32x4` and applies the
4x4 transform with a broadcasted contraction. Transform construction and
inversion use float32; the attention inputs are cast back to the production
attention dtype after finite checks.

Heads 0-11 permanently belong to the left arm and heads 12-23 permanently
belong to the right arm. Head ownership never depends on image half, projected
EEF position, arm crossing, or RGB detection.

## Raw-QKV fork and RoPE boundary

The frozen Wan projections are evaluated once:

1. apply frozen `q`, `k`, and `v` linear projections;
2. apply the existing frozen Q/K normalization;
3. fork the resulting pre-RoPE Q, K, and V.

The original path is unchanged:

`pre-RoPE Q/K -> Wan 3D RoPE -> original attention -> frozen Wan O`.

The geometry path does not apply Wan 3D RoPE:

`pre-RoPE Q/K/V -> arm-group SE(3) maps -> geometry attention -> inverse
output map -> presence mask -> channel gate -> frozen Wan O`.

For token `i` assigned to arm `k` and latent frame `n(i)`:

- `Q'_i = D_i^T Q_i`;
- `K'_i = D_i^-1 K_i`;
- `V'_i = D_i^-1 V_i`;
- `Ogeom_i = D_i Attention(Q', K', V')_i`.

The original and geometry attention calls use the same legal sequence lengths
and padding semantics. The geometry branch is parallel; it never consumes
Q/K after Wan RoPE and never alters the original Q/K/V tensors in place.

## Trainable gate and residual placement

Each selected block owns one float32 gate with shape `(24, 128)`. The first 12
heads and last 12 heads are independently trainable because they occupy
disjoint slices of the gate. Total Stage A trainable parameters are:

`3 blocks * 24 heads * 128 channels = 9,216`.

The gate is exactly zero initialized. It is applied to the presence-masked,
inverse-mapped geometry-head output before flattening and before the frozen
Wan output projection. Therefore:

- step 0 is exactly equal to the clean parent;
- absent-arm heads remain exactly zero even after training;
- one frozen Wan O call maps the concatenated gated geometry heads back to
  model width;
- head ownership is inspectable before Wan O mixes channels.

The geometry residual is added to the pretrained self-attention output before
the existing Wan self-attention modulation multiplier. No parameter or buffer
on the original path becomes trainable.

## Phase 0 hard gates

All gates below are fail closed:

1. Production Wan configuration is exactly width 3072, 24 heads, head
   dimension 128, and head dimension divisible by four.
2. The raw-QKV fork occurs after Q/K normalization and before Wan RoPE.
3. With the v7 gate disabled, the refactored original attention path is
   bitwise equal to the unmodified parent.
4. With all channel gates zero, the complete v7 model is bitwise equal to the
   parent on the same production input.
5. Constant action poses make every pairwise relative SE(3) transform identity;
   this does not assert that attention itself is an identity function.
6. Applying one common global SE(3) transform to every valid source pose leaves
   the anchored relative sequence unchanged within tolerance.
7. Swapping pose, presence, and arm assignment together changes only which
   fixed head group receives each action tuple. No final-output swap equality
   is asserted.
8. `arm_present=false` makes the corresponding pre-O geometry heads exactly
   zero for arbitrary hidden states.
9. The geometry implementation matches an explicitly materialized Kronecker
   reference on small tensors for Q, K, V, and inverse output mapping.
10. Non-finite, singular, malformed, wrong-shape, wrong-time, or wrong-schema
    conditions fail before model execution.
11. Wan Q/K/V/O and the parent Adapter receive no gradients; all and only the
    three channel gates are trainable.
12. Correct, reverse, shifted, and swapped actions produce non-identical
    geometry features on a deterministic fixture.

## Phase 1 gradient audit

Before optimizer training, use fixed parent inputs, noise, and timestep to
evaluate correct, reverse, shifted, and swapped actions. At gate zero, record:

- the geometry feature digest before the gate;
- per-block and per-arm feature RMS;
- the weighted-FM gradient with respect to each channel gate;
- cosine similarities and norm ratios between correct and counterfactual gate
  gradients.

This phase is a degeneracy test, not a success claim. It requires finite,
nonzero gate gradients and non-identical geometry features. It does not require
the zero-gate correct action to have lower FM loss, because the branch is still
silent.

## Stage A training

- Seven ranks, micro-batch one, production cached latents/text/action inputs.
- Optimizer updates: 50 maximum.
- Checkpoints: 10, 25, and 50.
- Trainable parameters: the 9,216 channel-gate parameters only.
- Loss: existing weighted flow matching only.
- No gripper injection in Stage A.
- Optimizer and learning rate are calibrated for the channel gates by a short
  gradient-scale smoke; they are frozen in the run contract before step 1.
- Smoke runs three complete forward/backward/step/zero-grad iterations.
- Every rank must report peak allocated and peak reserved memory below 22 GiB,
  finite loss/gradients, and nonzero gate gradients.

Step 10 is a health checkpoint and is not eliminated for weak mechanism
separation. It requires nonzero finite gates, bounded residual RMS, finite FM,
correct head attribution, and unchanged original-path parameters.

Step 25 is the first mechanism decision. On the fixed discovery set, compare
correct action with reverse, temporal shift, and left/right swap under matched
sample/noise/timestep. If correct shows no paired separation against every
counterfactual and the frozen trajectory probe has no positive direction, stop
without step 50 or RGB generation.

Step 50 requires:

- correct wins at least 6 of 8 discovery episodes under the frozen aggregate
  counterfactual criterion;
- frozen-probe position improvement is positive;
- frozen-probe velocity improvement is positive;
- spatial-routing retention is at least 90%;
- weighted-FM regression is at most 2%;
- no absent-arm output, head-ownership violation, NaN, Inf, or residual
  domination occurs.

An improvement of 3-5% is sufficient to establish a Stage A mechanism signal
when counterfactual ordering is systematic. It is not sufficient to claim a
video or leaderboard improvement.

## RGB anti-exploitation gate

Only a passing step-50 candidate may decode 4-8 fixed RGB episodes. Compare it
with the clean parent under the same first frame, instruction, action, seed,
sampler, sampling steps, CFG, and output contract. Latent trajectory direction
and RGB trajectory direction must agree. Black/broken video, detector coverage
loss, or obvious visual corruption blocks Stage B.

Dev-fast20 is not consumed during the mechanism probe. It becomes legal only
after the RGB anti-exploitation gate passes.

## Stage B boundary: six-block architecture

Stage B is permitted only after all Stage A and RGB gates pass and the user
explicitly approves the promotion. It never launches as an automatic
continuation of Stage A.

The intended Stage B architecture is:

- geometry branches in blocks 4, 8, 12, 16, 20, and 24;
- the same validated SE(3), head-ownership, presence, and RoPE contracts;
- geometry-only Q/K/V LoRA, initially rank 8 and promotable to rank 16 only by
  an explicit gate;
- per-arm gripper-state injection;
- clean-1785 eligible training data with a new zero-leakage receipt;
- original Wan attention and the existing parent Adapter still frozen;
- no full dedicated QKV/O unless the six-block LoRA version first demonstrates
  matched video improvement.

Stage B requires a separate design amendment covering optimizer groups,
exposure-based checkpoints, replay, CFG, fast20 promotion, and official
evaluation. Stage A approval does not approve those details.

## Stop rules and interpretation

Stop Stage A immediately for a mathematical-contract failure, original-path
drift, wrong trainable parameters, cache provenance mismatch, leakage, OOM,
non-finite values, absent-arm leakage, or persistent counterfactual
non-separation at the step-25 gate.

A Stage A success proves only that analytic arm-grouped relative SE(3)
attention is compatible with the frozen Wan representation and creates useful
action discrimination under the bounded probe. A failure rejects this shared
raw-QKV, frozen-O, gate-only interface; it does not by itself reject all
dedicated PRoPE-style architectures.

No Stage A result is described as an official WorldArena improvement until a
promoted model passes matched video generation and the official evaluator
pipeline.
