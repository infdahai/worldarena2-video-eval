# Wan v10 Action-Relational Native Attention Design

## Decision

The next primary experiment is `v10-action-relational-native-attention`.

v10 ends the phase-locked side-attention line tested by v9. It retains the
frozen support-gated raster parent, but action conditioning now changes the
relationship computed by native Wan self-attention heads. The implementation
must remain compatible with Wan's fused attention path and must never
materialize a visual-token pairwise action-bias tensor.

The experiment tests one principal hypothesis:

> A factorized, arm-specific action relation inside a twelve-block native Wan
> attention band, trained with direct temporal and trajectory supervision, can
> improve action separation and image-space gripper trajectory without
> regressing the native video objective.

This is an architectural experiment, not a continuation of v9. It starts from
the immutable clean-gated-step10 parent with fresh optimizer state, fresh v10
modules, a source-pinned global initialization seed, and a new clean-5k replay.

## Evidence motivating the design

v8 directly adapted native Q/K/V/O in blocks 8-13. Its fixed audit20 result was:

- reverse: 16/20 correct wins;
- swap: 17/20 correct wins;
- one-latent shift: 5/20 correct wins.

v9 added phase-locked four-token cross-attention, but its step100 audit did not
produce stable separation:

- reverse: 8/20;
- shift+1: 10/20;
- shift-1: 9/20;
- swap: 9/20.

Both runs preserved FM and routing. The evidence therefore favors modifying
native self-attention relations and directly supervising temporal trajectory,
not adding another residual or action-token side branch.

## Fixed operational boundaries

- Formal artifacts live only under
  `/data/di/worldarena2_track1_20260815`.
- Remote source lives at `/home/huazhi/nlh/baseline`.
- Wan source is read only at `/home/huazhi/nlh/Wan2.2`.
- Initial topology is one opportunistic RTX 4090, GPU6. A different topology
  requires a separate smoke and a separate lineage.
- No job stops, restarts, or occupies another user's process.
- The video contract remains 81 RGB frames at 480x640 and 21 latent frames.
- T5 and VAE are offline cache producers only. They are forbidden in the
  training hot path.
- The parent checkpoint remains clean-gated-step10 and is frozen.
- Depth, V-JEPA training loss, DMD, GAN, PRoPE, a coordination branch, and
  FFN/text-attention unfreezing are out of scope.

## Data contract

### Current availability

The formal server currently contains approximately 5,500 raw RoboTwin HDF5
episodes and 1.1 TiB of available disk. Only 1,785 episodes currently have the
complete provenance-bound cache used by v8/v9. A clean-5k optimizer dataset is
therefore feasible but is not currently ready.

### Clean-5k construction

The source pool is the formal 5,500-episode RoboTwin extraction. Before any
cache or optimizer replay is produced, every identity must be classified as
one of:

1. optimizer eligible;
2. v10 audit20;
3. dev-fast20 evaluator only;
4. rejected because source provenance or required media is incomplete.

The fixed v10 audit20 is rebuilt from training-domain samples and must contain
20 probe-observable, multi-task episodes. It is removed before optimizer
selection. Dev-fast20 remains disjoint from both optimizer and audit.

From the remaining source, exactly 5,000 identities are deterministically
selected using task and action-role strata. Selection uses content-hashed
inputs and a source-pinned algorithm version. It must not depend on filesystem
enumeration order.

The target realized optimizer distribution is:

| Stratum | Target share |
|---|---:|
| single-dominant | 40% |
| bimanual-heavy | 35% |
| mixed | 20% |
| quiet | 5% |

The receipt records configured and realized counts. Missing strata, identity
overlap, or a changed selection hash fails before cache generation.

### Cached inputs

Every optimizer and audit sample requires:

- video latent;
- text context;
- correct frozen-parent raster/support condition;
- anchored left/right SE(3) states at 21 latent times;
- left/right UV, gripper, presence, and motion descriptors;
- reverse and swap relational-action counterfactuals;
- RGB-observable left/right trajectory labels and validity masks.

Shift supervision does not need another video forward or a dense shifted
cache. It compares the predicted time-indexed trajectory with adjacent valid
targets while keeping destination time slots fixed.

### Zero-leakage proof

The data receipt must prove all of the following before CUDA initialization:

- optimizer and audit20 overlap is zero;
- optimizer and dev-fast20 overlap is zero;
- audit20 and dev-fast20 overlap is zero;
- every selected sample belongs to the approved training-domain source;
- official test identities are unavailable and were not read;
- normalization statistics use optimizer samples only;
- frozen-probe fitting identities do not overlap audit20;
- latent, text, action, label, and manifest hashes match their sidecars.

## Action-relational native attention

### Native block band

The trainable band is exactly blocks 6-17 inclusive.

For each selected block:

- native self-attention Q is trainable;
- native self-attention K is trainable;
- native self-attention V is trainable;
- native self-attention O is trainable;
- the v10 relation encoders and relation gates are trainable.

All FFNs, normalization, text cross-attention, blocks outside 6-17, the raster
parent, T5, VAE, and trajectory teacher are frozen.

### Head ownership

Wan TI2V-5B has 24 self-attention heads. Each selected block has immutable head
ownership:

- heads 0-7: LEFT action-relational;
- heads 8-15: RIGHT action-relational;
- heads 16-23: GLOBAL native visual.

LEFT and RIGHT identity is kinematic. It is never inferred from image half,
color, bounding-box order, or relative horizontal position. Crossing supports
may overlap. `arm_present=false` makes that arm's relation features exactly
zero. GLOBAL heads never receive relation features.

### Per-token action state

For arm `k`, latent time `t`, and visual patch `(x,y)`, v10 constructs a compact
state:

```text
z[t,x,y,k] =
    anchored SE3 log coordinates
  + local translational and angular velocity
  + normalized EEF uv
  + gripper opening and delta
  + arm presence and motion activity
  + support/swept-corridor spatial affinity at (x,y)
  + destination visual time encoding
```

The SE(3) anchor is the arm's own valid initial state:

`Gbar[t,k] = inverse(G[0,k]) @ G[t,k]`.

Counterfactual shift operations move action content only. Destination visual
time and patch location remain unchanged, preventing source-phase identity
from becoming a shortcut.

### Factorized relation bias

v10 must not evaluate an MLP for every visual-token pair and must not create a
tensor shaped like `[heads, visual_tokens, visual_tokens]`.

Instead, each LEFT/RIGHT head computes rank-16 relation factors:

```text
phi_i = relation_q(z_i)
psi_j = relation_k(z_j)
```

A single frozen-shape input normalizer and two-layer state encoder are shared
across blocks and arms. Each selected block owns separate LEFT/RIGHT rank-16
Q/K projections. LEFT and RIGHT projections may share weights within a block,
but their input banks and head destinations remain disjoint. The production
choice is shared weights within each block to avoid encoding arm identity in
parameters rather than data routing.

The conceptual attention score is:

`score(i,j,h) = native_q_i dot native_k_j / sqrt(128)
                + gate_h * phi_i dot psi_j / sqrt(128)`.

The fused implementation augments Q and K with the rank-16 factors:

```text
Q_aug = concat(native_Q, gate_h * phi)
K_aug = concat(native_K, psi)
V_aug = concat(native_V, zeros(rank=16))
```

Attention uses the native `1/sqrt(128)` scale, and the extra V dimensions are
discarded before native O. GLOBAL heads append zeros. The relation gate is a
signed FP32 scalar per action-aware head and is exactly zero initialized.

This construction has three required properties:

1. at initialization, output is numerically equivalent to native Wan;
2. relational scores are computed by the fused attention kernel without an
   explicit quadratic bias allocation;
3. setting every relation gate to zero provides an exact inference-time
   ablation while retaining learned native Q/K/V/O.

If the production fused kernel does not support augmented head width 144 and
an explicit native scale, implementation stops. It must not fall back to an
unfused 25,200-by-25,200 attention matrix.

## Training-only hidden EEF heads

Blocks 11 and 17 each own a small training-only per-token head that predicts
left and right EEF heatmaps on the 21-by-30-by-40 latent grid.

Targets come from RGB-observable GT trajectory labels, not directly from the
commanded action projection. This prevents the auxiliary head from satisfying
the loss by merely decoding its own action input.

The head loss is applied only where:

- the arm is RGB observable;
- the label passed geometric and detector consistency checks;
- the latent timestep is valid;
- the sample is not quiet or motion-ambiguous for the supervised term.

High-noise diffusion timesteps are downweighted using the frozen v6 sigma
contract. The heads are checkpointed for exact resume and audit, but removed
from the final inference package.

## Objective

### Correct weighted FM

Correct weighted FM remains the visual-generation guardrail and uses the
existing fixed correct-action mask. It is never silently averaged over invalid
latent padding.

### Counterfactual action ranking

Each mechanism-phase optimizer step evaluates:

- one correct forward;
- one wrong relation-action forward.

The frozen raster parent remains on the correct condition in both forwards.
Only the v10 relational action state changes. This isolates whether native
relational attention learns action semantics instead of measuring the already
trained raster parent.

Wrong families alternate deterministically between reverse and swap. Ranking
energy is unreduced FM over the fixed correct robot/support region:

`L_cf = softplus((E_correct - E_wrong) / tau)`.

### Phase ranking

The predicted clean trajectory at latent time `t` is compared against the GT
target at `t`, `t-1`, and `t+1` using only motion-discriminative valid entries:

```text
D(pred[t], target[t]) < D(pred[t], target[t-1])
D(pred[t], target[t]) < D(pred[t], target[t+1])
```

Boundaries use validity masks; they are not padded into artificial wins.

### Position and velocity

Position and velocity losses use the frozen validated trajectory probe on the
predicted clean latent. They operate per arm and latent timestep with explicit
visibility and validity masks.

### Loss schedule

The architecture and optimizer lineage remain fixed, but supervision is staged:

#### Mechanism phase, steps 1-250

`L = L_FM + lambda_cf L_cf + lambda_phase L_phase
     + lambda_hidden L_hidden`.

#### Trajectory phase, steps 251-500

Only after the step250 gate passes:

`L = L_FM + lambda_cf L_cf + lambda_phase L_phase
     + lambda_hidden L_hidden
     + lambda_pos L_pos + lambda_vel L_vel`.

No loss is added at step500 or later without a new approved design.

All lambda values are calibrated once using FP32 native-Q/K/V/O gradient norms
on a fixed source-pinned calibration batch and fixed RNG state. Target norms
relative to correct FM are:

- counterfactual ranking: `1.0 × FM`;
- phase ranking: `1.0 × FM`;
- combined block-11/17 hidden EEF loss: `0.5 × FM`;
- position loss after step250: `0.5 × FM`;
- velocity loss after step250: `0.5 × FM`.

Each calibrated lambda must be finite, positive, and within the source-pinned
safety interval `[1e-4, 100]`; otherwise training stops. Calibration is stored
in the checkpoint and is never recomputed on resume.

## Determinism contract

Before model or v10 module construction, the trainer sets and records:

- Python RNG seed;
- NumPy RNG seed;
- Torch CPU seed;
- Torch CUDA seed;
- deterministic initialization seed for relation encoders, gates, and hidden
  heads.

The replay fixes sample identity, noise seed, timestep seed, negative family,
and dropout decisions. Two fresh preflights using the same source and replay
must produce byte-identical initialized v10 trainable state and identical loss
calibration. Failure blocks smoke.

## Probe validity and aggregation

The v9 behavior of propagating infinity into aggregate means is prohibited.

Every trajectory report contains:

- total episodes;
- finite/observable episodes;
- invalid episodes and reasons;
- finite-only mean and median;
- failure-aware score where invalid episodes receive the declared penalty.

Audit20 must be 20/20 finite before training. Optimizer data may include
FM-only samples, but at least 60% of optimizer identities must have eight or
more valid arm-time trajectory targets. Realized supervised exposure is logged
per checkpoint.

## Optimizer and checkpoint contract

The optimizer has explicit disjoint parameter groups:

1. native Q/K/V/O;
2. relation Q/K encoders;
3. FP32 relation gates;
4. block-11/17 hidden EEF heads.

The formal AdamW learning rates are:

- native Q/K/V/O: `1e-6`;
- relation state encoder and per-block Q/K projections: `1e-4`;
- FP32 relation gates: `5e-5`;
- hidden EEF heads: `1e-4`.

All groups use AdamW betas `0.9/0.999`, epsilon `1e-8`, and gradient clipping
at global norm `1.0`. Gates use zero weight decay; other groups use `0.01`.
The schedule has 50 warmup steps followed by cosine decay to 20% of each base
rate at step500. A step1000 extension requires a separately frozen continuation
schedule in its gate receipt. A parameter outside these families, a missing
parameter, an alias, or an incomplete AdamW state fails closed.

Checkpoint metadata records:

- global completed step and phase;
- global initialization seed;
- source closure and config hashes;
- parent, data, cache, replay, audit, probe, and calibration hashes;
- topology and physical GPU mapping;
- exact trainable parameter inventory;
- optimizer and scheduler state;
- loss calibration;
- gate decisions and parent checkpoint hash;
- cumulative samples seen and realized stratum counts.

Raw checkpoints cannot start the next phase. Continuation always resumes the
corresponding `-gated` checkpoint.

## Runtime gates

### Source and cache gates

- recursive source closure is committed, clean, and hash matched;
- clean-5k cache and label receipts validate completely;
- audit20 is 20/20 finite and excluded from replay;
- no training process starts if a persistent path escapes the formal root.

### Production smoke

Smoke uses GPU6, production 81-frame/480x640 latent geometry, formal dtype,
activation checkpointing, and the exact training optimizer. After warmup and a
peak reset it executes three complete iterations:

`correct + wrong forward -> backward -> optimizer.step -> zero_grad`.

All iterations must satisfy:

- peak allocated and reserved below 22 GiB;
- no unfused quadratic attention allocation;
- all four trainable families have finite nonzero gradients;
- zero relation gates reproduce native attention within the declared numeric
  tolerance;
- no NaN, Inf, OOM, unexplained memory growth, or cross-GPU process.

### Step50 health gate

- finite optimizer and all required gradients;
- FM regression no more than 2%;
- relation gates and hidden heads have updated;
- routing retention at least 90%;
- no source, replay, topology, or scheduler drift.

Failure stops the run.

### Step250 mechanism gate

On fixed audit20:

- reverse correct wins at least 11/20 with positive mean margin;
- swap correct wins at least 11/20 with positive mean margin;
- phase+1 correct wins at least 11/20 with positive mean margin;
- phase-1 correct wins at least 11/20 with positive mean margin;
- finite-only position and velocity improvement are positive;
- routing retention at least 90%;
- FM regression no more than 2%;
- enabled relation is better than the exact gate-zero ablation on the combined
  action-separation score.

Only a passing gated step250 may enter the trajectory phase.

### Step500 hard gate

On the same audit20:

- reverse, swap, phase+1, and phase-1 each achieve at least 14/20 wins and a
  positive mean margin;
- finite-only position improvement exceeds 5%;
- finite-only velocity improvement exceeds 5%;
- routing retention is at least 90%;
- FM regression is no more than 2%;
- the relation-enabled model beats its gate-zero ablation.

Passing step500 decodes exactly eight matched RGB samples covering single-arm,
bimanual, crossing, and sequential-role tasks. Advancement additionally
requires zero black/broken videos, no detectability regression, and proxy
trajectory improvement in the same direction as the latent audit.

### Step1000 extension

Step1000 is not automatic. It is allowed only if the step500 RGB proxy improves
at least 5%, paired wins exceed 55%, visual guardrails pass, and the step250 to
step500 mechanism curve is still improving. Otherwise step500 is final.

## Failure interpretation

- Reverse/swap improve but phase fails: relational identity/direction works;
  temporal supervision or factorization is insufficient. Do not add more
  blocks before inspecting phase targets.
- Mechanism passes but trajectory fails: action semantics exist, but final
  visual trajectory supervision is ineffective. Inspect probe/hidden-head
  coupling before changing attention.
- Gate-zero equals enabled relation while native Q/K/V/O improve: gains come
  from native fine-tuning, not v10 relation features.
- FM regresses while action metrics improve: do not promote; visual-generation
  damage exceeds the competition objective tradeoff.
- Production augmented-head Flash Attention is unsupported: stop v10 as
  designed. Do not substitute an explicit pairwise bias implementation.

## Deliverables

The implementation phase must produce:

1. clean-5k manifest, zero-leakage receipt, and cache completeness report;
2. deterministic initialization and calibration receipt;
3. fused factorized relational-attention module and exact gate-zero ablation;
4. RGB-observable hidden EEF supervision with finite validity masks;
5. guarded single-GPU launcher and checkpoint validator;
6. step50, step250, and step500 audit reports;
7. a final experiment report recording positive and negative results without
   silently excluding invalid samples.
