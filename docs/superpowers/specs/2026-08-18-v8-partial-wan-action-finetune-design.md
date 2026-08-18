# Wan-Action v8 Partial Wan Action Fine-Tune Design

## Decision

v8 ends the frozen-backbone small-branch program. The approved first experiment
tests one remaining mechanism hypothesis:

> Can a continuous band of native Wan self-attention learn that a complete,
> correct robot action must predict the future better than a physically
> consistent reversed, shifted, or arm-swapped action?

The experiment partially unfreezes native Wan attention while retaining the
validated SE(3) geometry path as a structural prior. It does not enlarge the
old frozen branch, increase its rank, or continue v7.1-CF from step 25.

The experiment is named `v8-partial-wan-action-finetune`. It starts from the
immutable clean-gated-step10 parent and is initialized from scratch. A v7 or
v7.1 checkpoint is not a legal parent.

## Evidence and scope

v7.1-CF established all of the following:

- the geometry branch was connected and trainable;
- Q/K/V/O geometry LoRA and the channel gates received gradients;
- explicit counterfactual ranking produced small sampled energy margins;
- correct global flow matching remained healthy;
- fixed audit-20 action separation and position/velocity probes did not
  improve.

Those results do not prove that every possible frozen-backbone adapter is
mathematically incapable of action conditioning. They are sufficient for the
deadline-driven engineering decision to stop spending experiments on small
frozen-backbone branches.

v8 changes only the representation capacity and counterfactual completeness
needed to test partial native adaptation. It does not add trajectory loss in
Stage A.

## Fixed boundaries

- Formal artifacts live under `/data/di/worldarena2_track1_20260815`.
- Remote source lives at `/home/huazhi/nlh/baseline`; Wan source lives at
  `/home/huazhi/nlh/Wan2.2`.
- The parent is the clean-gated-step10 checkpoint with SHA256
  `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- The existing support-gated raster parent remains frozen.
- T5 and VAE remain outside the training hot path. Training consumes cached
  latents, cached text context, action conditioning, and SE(3) conditioning.
- The video contract remains 81 RGB frames at 480x640 and 21 latent frames.
- Native attention V, FFN/MLP, LayerNorm, QK normalization, text
  cross-attention, VAE, and T5 remain frozen in Stage A.
- Stage A adds no position loss, velocity loss, trajectory loss, gripper bias,
  Depth, V-JEPA, DMD, GAN, or coordination branch.
- The training topology is selected once before smoke and remains immutable
  through step 250. A running job never changes world size.
- GPU ownership is checked before launch. v8 never stops or reuses another
  user's process.

## Data and zero-leakage contract

### Stage A data

Stage A uses the already cached, provenance-bound `clean-1785` eligible source
with the existing balanced sampler. It does not wait for clean-5k. The 20
fixed audit episodes are removed from the optimizer-eligible rows before the
balanced replay is built.

The balanced target mix remains:

| stratum | target share |
|---|---:|
| single-dominant | 45% |
| bimanual-heavy | 30% |
| mixed | 15% |
| quiet | 10% |

The replay receipt records both configured and realized counts. A missing
stratum, a changed sample identity, or a mismatch between configured and
realized replay bytes fails closed.

`clean-1000` is retained only for deterministic smoke/debug comparison. It is
not the formal Stage A training pool.

### Evaluation isolation

- `dev-fast20` has zero sample overlap with training, block scan, calibration,
  and audit data.
- The fixed audit-20 has zero sample overlap with optimizer replay. It is a
  deterministic held-out subset of clean-1785 and remains identical at steps
  0, 100, and 250.
- Official test data remains unavailable and is never accessed by Stage A. If
  an official frozen manifest later becomes available, its identities must be
  checked before any test generation, not during training selection.
- Block-scan discovery and audit-20 are disjoint from each other.
- Block-scan discovery is train-only and is not used for checkpoint gates.
- Audit-20 is fixed before training and is not used to select the six-block
  band.

Every manifest and derived subset is schema-versioned, content-hashed, and
bound into the run receipt before CUDA initialization.

### Later scale-up

clean-5k is a gated Stage B artifact, not a Stage A prerequisite. It may be
prepared in parallel, but Stage B cannot launch until it has:

- an immutable source manifest;
- a zero-overlap receipt against dev-fast20 and an authoritative source-split
  receipt proving that every row belongs to the train partition rather than an
  official-test partition; if a frozen official-test manifest is available,
  its identities are additionally checked explicitly;
- complete action, SE(3), latent, and text-context caches;
- the same balanced-stratum schema as Stage A;
- explicit approval after the Stage A mechanism result.

## Continuous action-control band

### Train-only block scan

The band contains exactly six consecutive self-attention blocks. Selection
uses a deterministic 32-episode train-only discovery set covering multiple
tasks and all four sampler strata.

For each Wan self-attention block, the scan measures normalized native Q/K/O
counterfactual gradient RMS under complete correct-versus-wrong action pairs.
It reports separate reverse, shift, and swap scores. It performs no optimizer
step and writes no checkpoint.

Each six-block window receives the median normalized score across episodes and
the minimum score across the three negative families. The default window is
blocks `8-13`. Another window is selected only if:

1. its aggregate score is at least 25% greater than blocks `8-13`;
2. it is greater for at least two of the three negative families; and
3. its advantage is present in both halves of the discovery set.

If no window meets all three conditions, blocks `8-13` are selected. The
selection receipt records per-block scores, the exact decision, and all input
hashes.

### Native attention trainability

Within the selected band:

- native Q is trainable;
- native K is trainable;
- native O is trainable;
- native V is frozen;
- native attention execution and Wan 3D RoPE remain production-identical.

Outside the selected band all Wan parameters remain frozen.

For Wan width 3072, six blocks of full Q/K/O contain approximately 170 million
trainable parameters before projection biases. The runtime records the exact
parameter names, shapes, dtypes, and count. Any trainable parameter outside the
approved whitelist blocks training.

## SE(3) geometry path in the same band

Every selected block contains the validated arm-grouped SE(3) geometry path:

```text
raw hidden
  |
  +-- shared native Q/K/V projections
  |     -> native Q/K normalization
  |     -> Wan 3D RoPE
  |     -> native attention
  |     -> shared native O
  |
  +-- shared native pre-RoPE Q/K/V
        + geometry-specific rank16 Q/K/V LoRA
        -> fixed left/right arm-grouped SE(3)
        -> geometry attention
        -> shared native O
        + geometry-specific rank16 O LoRA
        -> zero-initialized channel gate
        -> residual
```

There is no frozen duplicate of native Q/K/O. The geometry path shares the
same native Q/K/O projections that the native path uses; its rank16 LoRA is the
branch-specific delta. Native V stays frozen and shared. This avoids two
diverging base representations and avoids a second full projection copy.

The existing SE(3) mathematical contract remains unchanged:

- the fork occurs after native Q/K normalization and before Wan 3D RoPE;
- per-arm identity is kinematic and never derived from image half;
- arm transforms are anchored relative to each arm's legal initial pose;
- missing-arm geometry output is exactly zero;
- motion-scale normalization remains episode-shared;
- left/right support may overlap in bimanual and crossing scenes.

Each geometry channel gate is exactly zero initialized. Therefore geometry
LoRA is allowed to have zero step-0 gradient. Native Q/K/O must receive finite,
nonzero gradients at step 0.

## Complete-action counterfactual contract

Each optimizer step uses one correct forward and one wrong forward with the
same RGB latent, text context, diffusion noise, timestep, GT future, and fixed
energy mask.

The wrong family cycles deterministically:

```text
reverse -> shift+1 -> swap -> reverse -> shift-1 -> swap -> ...
```

Unlike v7.1-CF, the complete action changes coherently:

- action raster;
- EEF heatmaps;
- flow-x and flow-y;
- arm occupancy/support used as model conditioning;
- gripper opening;
- SE(3) transforms and arm presence.

Transform semantics are:

- **reverse:** reverse relative motion while preserving legal initial anchors,
  then recompute flow and all time-derived channels;
- **shift:** shift the complete action by one latent step, alternate `+1/-1`,
  and use hold padding at the exposed boundary;
- **swap:** exchange left/right relative motion and arm-local channels while
  preserving each arm's own initial pose and kinematic identity.

All transformed payloads pass the same finite, shape, temporal-packing, and
SE(3)-inverse validation as correct inputs.

## Conditioning support and ranking energy mask

The implementation exposes two distinct tensors:

1. `conditioning_support` is part of the action input and changes with the
   complete negative action.
2. `ranking_energy_mask` is derived once from the correct action support and
   correct valid-latent mask. It is identical in correct and wrong energy
   evaluation.

The two concepts must not share an overloaded argument. The wrong action never
changes its comparison denominator.

For each sample, the ranking mask is:

```text
valid latent mask * union(correct left support, correct right support)
```

The existing correct `loss_weight` is applied inside that support. A sample
with zero finite mask mass fails closed rather than falling back to global
background energy.

The per-sample energy is unreduced robot/action-region flow-matching error,
normalized by fixed correct-mask mass:

```text
E(action) = sum(mask * correct_loss_weight * FM_error(action))
            / sum(mask * correct_loss_weight)
```

## Stage A objective and calibration

Stage A uses:

```text
L = weighted_FM(correct) + lambda_cf * L_cf

L_cf = softplus((E_correct - E_wrong) / tau)
tau = 0.1
```

Only one wrong family is graph-bearing per optimizer step. The implementation
may use the validated sequential two-forward analytic-VJP method to avoid
holding two checkpoint graphs simultaneously, but its gradients must match a
direct pairwise oracle on a small non-checkpoint fixture.

`lambda_cf` is calibrated once on a deterministic calibration batch so that:

```text
RMS(lambda_cf * grad_native_QKO(L_cf))
    = 0.5 * RMS(grad_native_QKO(weighted_FM))
```

The common active set is native Q/K/O only. Zero-gated geometry LoRA and gates
are not included in the calibration denominator. Non-finite, zero, or
disconnected calibration gradients block launch. The calibrated scalar is
then frozen for the entire Stage A run; it is never dynamically adjusted from
training metrics.

## Optimizer and schedule

Stage A uses one optimizer with explicit parameter groups:

| parameter family | initial LR | weight decay |
|---|---:|---:|
| native Q/K/O in selected band | `1e-6` | `0.01` |
| geometry Q/K/V/O LoRA rank16 | `5e-5` | `0` |
| geometry channel gates | `5e-5` | `0` |

- optimizer: AdamW;
- warmup: 25 optimizer steps;
- schedule: cosine decay to 20% of each group's initial LR at step 250;
- global gradient clipping: L2 norm `1.0`;
- micro-batch: one per rank;
- activation checkpointing: production Wan policy;
- precision: production bfloat16 where supported, with FP32 optimizer states,
  LoRA parameters, gates, SE(3) construction, and ranking reductions.

The checkpoint stores all optimizer groups, scheduler state, calibrated
`lambda_cf`, selected band, topology, replay cursor, source closure, parent
hash, dataset hash, and update count. Resume rejects any mismatch.

If the gated step-500 extension is approved, the same scheduler continues
without reset from 20% at step 250 to 10% at step 500. It does not introduce a
second warmup or restore a higher LR.

## Checkpoints and gates

### Step 25: health gate

Step 25 is not an action-separation elimination point. It verifies:

- every approved native Q/K/O family has finite, nonzero gradient history;
- geometry LoRA and gates begin updating after the zero-gate boundary;
- native V and all other frozen parameters have no gradients;
- correct FM and ranking loss are finite;
- attention, geometry residual, and block-output RMS remain within 10x of the
  preflight reference and show no monotonic explosion;
- routing retention is at least 90%;
- no NaN, Inf, OOM, or optimizer/scheduler mismatch occurred.

Failure stops immediately. Weak counterfactual wins alone do not stop step 25.

### Step 100: trend gate

Fixed audit-20 requires:

- reverse wins at least 11/20;
- hard `shift(+1,-1)` wins at least 11/20;
- swap wins at least 11/20;
- all three average ranking margins are positive;
- routing retention is at least 90%;
- correct FM regression is below 2%;
- position and velocity do not regress by more than 2% relative to the fresh
  parent.

If any counterfactual family remains at or below 10/20, has non-positive mean
margin, or the health gates fail, Stage A stops. A passing run continues to
step 250.

### Step 250: mechanism hard gate

Fixed audit-20 requires:

- reverse wins at least 14/20;
- hard shift wins at least 14/20;
- swap wins at least 14/20;
- all three average ranking margins are positive;
- routing retention is at least 90%;
- correct FM regression is at most 2%.

This gate is intentionally separate from the trajectory-benefit gate.

If the mechanism hard gate fails, v8 partial Q/K/O Stage A stops. It does not
continue to step 500, unfreeze V, increase rank, widen the band, or add
trajectory loss.

### Trajectory-benefit decision

After the mechanism gate passes, compare position and velocity probes with the
fresh parent:

- if both improve by more than 5%, decode 4-8 fixed RGB sanity samples;
- if counterfactual separation passes but either trajectory probe improves by
  5% or less, the mechanism is considered learned but incomplete, and the run
  becomes eligible for a separately approved Stage B with v6 position and
  velocity supervision;
- if RGB shows black/broken output, severe arm disappearance, or obvious
  quality collapse, do not run fast20.

Step 500 is allowed only when step 250 passes the mechanism gate, both
trajectory probes improve by more than 5%, RGB is healthy, and audit metrics
are still improving from step 100 to step 250. It uses the same objective and
does not change the trainable whitelist.

## Production smoke and memory gate

The selected topology runs a production-shape smoke before Stage A:

```text
81 frames
480x640
cached latent and text context
complete action conditioning
six-block native Q/K/O plus geometry path
one correct and one wrong forward
backward -> optimizer.step -> zero_grad
three consecutive iterations
```

After initialization and warmup, reset CUDA peak statistics. Every rank must
record allocated/reserved peak memory and step time. Every rank must remain
below 22 GiB allocated and 22 GiB reserved. Any OOM, unexplained memory growth,
foreign trainable parameter, or inactive required parameter fails closed.

Smoke also proves that T5 and VAE are not loaded in the training process.

## Stage B boundary

Stage B is not automatically launched by this specification. It requires the
Stage A mechanism gate and a new explicit approval.

The allowed Stage B direction is:

```text
L = weighted_FM
    + calibrated counterfactual ranking
    + calibrated v6 position loss
    + calibrated v6 velocity loss
```

Initial gradient budgets relative to native-Q/K/O FM gradient are:

- counterfactual: 50-100%;
- position: 20-30%;
- velocity: 10-20%.

Stage B may use clean-5k after its independent cache and zero-leakage contract
passes. Native V may be unfrozen only if counterfactual separation is already
stable but trajectory improvement plateaus. MLP, norms, cross-attention, VAE,
and T5 remain frozen.

## RGB and matched evaluation

The first matched video comparison contains only:

- clean S1A125;
- clean-gated-step10;
- v8 best-1;
- v8 best-2, only if a second checkpoint has a distinct mechanism/quality
  trade-off.

Generation conditions are identical across candidates. Promotion requires:

- failure-aware trajectory improvement of at least 5%;
- paired wins of at least 12/20;
- detector coverage drop no greater than 5 percentage points;
- no detector-failure regression hidden by survivor-only averaging;
- black/broken count equal to zero;
- bimanual and temporal-leakage guardrails not worse than clean-gated-step10.

VLM and JEPA remain no-regression evaluators for later full development
evaluation; they do not enter Stage A training.

## Provenance and artifact contract

The run records:

- source commit and recursive source-closure SHA256;
- parent path and SHA256;
- Wan model-manifest and source hashes;
- clean-1785 manifest/cache and sampler hashes;
- block-scan discovery and decision hashes;
- audit-20 hash;
- replay hash and per-step negative identity;
- exact trainable names/counts and optimizer groups;
- calibrated `lambda_cf`, gradient norms, and calibration batch hash;
- topology, physical GPU mapping, world size, dtype, and software environment;
- checkpoint lineage, optimizer/scheduler state, and step count;
- smoke, audit, RGB, and matched-evaluation reports.

Persistent outputs are written atomically under a dedicated v8 directory in
`/data/di/worldarena2_track1_20260815`. Existing v7/v7.1 artifacts are
read-only provenance inputs and are never overwritten.

## Required tests

Implementation must provide focused tests for:

1. exact native trainable whitelist and frozen V/MLP/norm/cross-attention;
2. six consecutive blocks and deterministic scan fallback to `8-13`;
3. scan/audit/dev-fast20/official-test identity disjointness;
4. shared native projections with no duplicate full Q/K/O copy;
5. zero-gate parent equivalence before any optimizer step;
6. complete reverse, shift+1, shift-1, and anchored swap transformations;
7. fixed ranking-energy mask under changed conditioning support;
8. direct pairwise versus sequential analytic-VJP gradient equivalence;
9. calibration over native Q/K/O only and frozen `lambda_cf` thereafter;
10. optimizer-group LR, weight decay, scheduler, and resume validation;
11. step25/100/250 gate semantics and prohibition of premature step50/500;
12. checkpoint rejection after source, data, band, topology, or replay drift;
13. production smoke memory, gradient, T5/VAE-absence, and telemetry checks;
14. zero persistent writes outside the authoritative `/data/di` root.

## Final decision tree

```text
clean-gated-step10
        |
train-only six-block scan
        |
continuous native Q/K/O band
+ shared SE(3) geometry LoRA
        |
full-action counterfactual objective
+ clean-1785 balanced
        |
step25 health
        |
step100 trend
        |
step250 mechanism gate
        |
        +-- CF separation fails
        |      -> stop partial-Q/K/O Stage A
        |
        +-- CF separation passes, trajectory <=5%
        |      -> eligible for approved trajectory-supervision Stage B
        |
        +-- CF separation passes, trajectory >5%
               -> RGB sanity -> optional step500 -> matched fast20
```

This experiment answers one question only: whether partial native Wan
attention adaptation allows complete action semantics to influence the future
video prediction. It does not attempt to solve data scale, trajectory
supervision, and visual-quality optimization in the same first run.
