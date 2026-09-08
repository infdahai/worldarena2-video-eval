# Wan-Action v8 Direct Action Band Design

## Decision

The approved experiment is `v8-direct-action-band`.

It ends the frozen-backbone small-branch program and directly adapts one
continuous band of native Wan self-attention. It tests two ordered hypotheses
in one immutable lineage:

1. **Phase M:** complete-action counterfactual supervision can make native Wan
   reliably prefer the correct action over reverse, shift, and swap.
2. **Phase T:** once action separation exists, the validated v6 position and
   velocity supervision can turn that separation into better image-space
   gripper trajectory.

v8 starts from the immutable clean-gated-step10 parent. It does not warm-start
from v7 or v7.1, and it does not reuse their optimizer states.

## Why this design replaces the previous v8 draft

The previous draft combined native Q/K/O unfreezing, a six-window gradient
scan, and four geometry-specific LoRA families. That was safer but left two
learned representations and an ambiguous attribution path.

The direct-action-band design is more aggressive and simpler:

- blocks `8-13` are fixed before training;
- native Q/K/V/O are fully trainable in those six blocks;
- geometry-specific Q/K/V/O LoRA is removed;
- deterministic SE(3) attention shares native Q/K/V/O;
- one channel gate per block controls the geometry residual;
- action separation and trajectory supervision are measured at a fixed phase
  boundary rather than mixed from step 0.

The design deliberately unfreezes V. Q/K can change attention relations, but V
is needed to change the motion and visual content carried through those
relations. O allows the adapted result to be written back into Wan.

## Fixed boundaries

- Formal artifacts live under `/data/di/worldarena2_track1_20260815`.
- Remote source lives at `/home/huazhi/nlh/baseline`; Wan source lives at
  `/home/huazhi/nlh/Wan2.2`.
- The parent is clean-gated-step10 with SHA256
  `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- The existing support-gated raster parent is frozen.
- The trainable native band is exactly blocks `8-13`.
- All Wan blocks outside `8-13` remain frozen.
- FFN/MLP, LayerNorm, QK normalization, text cross-attention, VAE, and T5
  remain frozen everywhere.
- T5 and VAE remain outside the training hot path. Training consumes cached
  latents and cached text context.
- The video contract remains 81 RGB frames at 480x640 and 21 latent frames.
- Phase M uses only weighted FM and complete-action counterfactual ranking.
- Phase T adds only the already validated v6 position and velocity losses.
- No geometry LoRA, block scan, trajectory auxiliary network, gripper bias,
  Depth, V-JEPA, DMD, GAN, coordination branch, or MLP unfreezing is allowed.
- The selected world size and physical rank mapping remain fixed from smoke
  through step 250. No running job changes topology.
- GPU ownership is checked before every launch. v8 never stops or occupies
  another user's process.

## Data and zero-leakage contract

### Eligible source and optimizer pool

v8 uses the already cached, provenance-bound `clean-1785` source. It does not
wait for clean-5k.

A deterministic 20-episode audit set is selected before replay construction
from the frozen v6 probe's 177-row held-out split. It covers multiple tasks,
both arm identities, single-dominant scenes, bimanual scenes, and valid
position/velocity probe targets. Those 20 identities are removed from
optimizer-eligible rows. They were never used to fit the frozen probe.

The remaining rows use the existing balanced sampler:

| stratum | target share |
|---|---:|
| single-dominant | 45% |
| bimanual-heavy | 30% |
| mixed | 15% |
| quiet | 10% |

The replay receipt records configured and realized stratum counts. A changed
sample identity, missing stratum, replay-order drift, or mismatch between
configured and realized bytes fails closed.

`clean-1000` is retained only as a smoke/debug reference. It is not the formal
v8 optimizer pool.

### Isolation rules

- audit-20 has zero overlap with optimizer replay;
- dev-fast20 has zero overlap with optimizer replay, calibration, and
  audit-20;
- official test data is unavailable and never accessed by training or
  development selection;
- all manifests and derived subsets are schema-versioned and content-hashed;
- source-split provenance must prove that every optimizer row belongs to the
  training partition;
- every receipt is validated before CUDA initialization.

clean-5k may be prepared in parallel, but it is not used in this experiment.
Scale-up requires a separate approval after step250 passes and requires its own
cache-completeness and zero-leakage receipts.

## Direct native action-control band

### Fixed block selection

The action-control band is exactly blocks `8-13`.

There is no block-gradient scan. Fixing the band removes selection noise,
avoids consuming another discovery set, and places the trainable sequence
immediately after the validated block-8 action injection point.

### Native trainability

For each selected block:

- native Q is trainable;
- native K is trainable;
- native V is trainable;
- native O is trainable;
- native Wan 3D RoPE and attention implementation remain unchanged.

For width 3072, six full Q/K/V/O projection families contain approximately
226.5 million weights before projection biases. The runtime records the exact
parameter names, shapes, dtypes, and total count from the production model.

The trainable whitelist is exact. One missing Q/K/V/O family, one extra Wan
parameter, or a trainable parent-adapter parameter blocks smoke and training.

## Shared deterministic SE(3) path

Each selected block evaluates two attention paths from one shared native
representation:

```text
raw hidden
  |
  +-- trainable native Q/K/V
        |
        +-- native Q/K normalization
        |     -> Wan 3D RoPE
        |     -> native attention
        |     -> trainable shared O
        |
        +-- pre-RoPE Q/K/V
              -> fixed arm-grouped SE(3)
              -> geometry attention
              -> trainable shared O
              -> zero-initialized channel gate
              -> geometry residual
```

There is no geometry-specific Q/K/V/O LoRA and no duplicate full projection
copy. The native and geometry paths share the same trainable Q/K/V/O.

The shared O projection is evaluated separately on native-attention output and
geometry-attention output. The channel-gated geometry projection is added to
the native self-attention output at the same pre-modulation residual location
validated by v7/v7.1; it does not bypass or duplicate the Wan block modulation
contract.

The validated SE(3) mathematical contract remains unchanged:

- geometry forks after native Q/K normalization and before Wan 3D RoPE;
- head identity is kinematic, never inferred from image half;
- left and right head groups are fixed;
- each arm preserves its legal initial anchor;
- `arm_present=false` makes that arm's geometry output exactly zero;
- motion-scale normalization is shared within an episode;
- left and right support may overlap in crossing and bimanual scenes.

Each block owns one FP32 channel gate with width 3072. It is exactly zero
initialized. Step 0 is therefore output-equivalent to the clean parent even
though native Q/K/V/O are marked trainable. Native Q/K/V/O must receive finite,
nonzero gradients at step 0; geometry contribution begins after its gate moves
away from zero.

For audit attribution, every checkpoint is evaluated twice without retraining:

1. learned geometry gates enabled;
2. all geometry gates forced to zero.

This ablation determines whether the deterministic SE(3) prior contributes
beyond native full-action adaptation.

## Complete-action counterfactual contract

Each optimizer step uses one correct forward and one wrong forward with
identical:

- RGB latent;
- text context;
- diffusion noise and timestep;
- GT future;
- valid-latent mask;
- ranking energy mask.

The wrong family cycles deterministically:

```text
reverse -> shift+1 -> swap -> reverse -> shift-1 -> swap -> ...
```

The complete wrong action coherently changes:

- raster occupancy;
- EEF heatmaps;
- flow-x and flow-y;
- gripper opening;
- conditioning support;
- SE(3) transforms;
- arm presence.

Transform rules are:

- **reverse:** reverse relative motion while preserving legal initial anchors,
  then recompute every time-derived raster and flow channel;
- **shift:** shift the full action by one latent step, alternate `+1/-1`, and
  use hold padding at the exposed boundary;
- **swap:** exchange left/right relative motion and arm-local channels while
  preserving each arm's own initial pose and kinematic identity.

Every wrong payload passes the same shape, finite, temporal-packing, SE(3)
inverse, and arm-presence validation as a correct payload.

## Conditioning support and fixed ranking mask

The model interface exposes two different tensors:

1. `conditioning_support` is action input and changes with the wrong action.
2. `ranking_energy_mask` is derived from correct support and never changes
   between correct and wrong energy evaluation.

The wrong action cannot change its comparison denominator.

For each sample:

```text
ranking_energy_mask
    = valid latent mask
      * union(correct left support, correct right support)
```

The existing correct `loss_weight` remains inside that region. A sample with
zero or non-finite fixed mask mass fails closed rather than falling back to a
global background average.

The per-sample ranking energy is:

```text
E(action) = sum(mask * correct_loss_weight * unreduced_FM(action))
            / sum(mask * correct_loss_weight)
```

## Phase M: action-separation training

Phase M runs from step 0 through step 100:

```text
L_M = weighted_FM(correct) + lambda_cf * L_cf

L_cf = softplus((E_correct - E_wrong) / tau)
tau = 0.1
```

Only one wrong family is graph-bearing per optimizer step. The sequential
two-forward analytic-VJP implementation must match a direct pairwise oracle on
a small non-checkpoint fixture.

### Counterfactual calibration

`lambda_cf` is calibrated once on a deterministic calibration batch over the
common active native Q/K/V/O set:

```text
RMS(lambda_cf * grad_QKVO(L_cf))
    = 0.5 * RMS(grad_QKVO(weighted_FM))
```

Zero-gated geometry parameters are excluded from the calibration denominator.
Non-finite, zero, or disconnected gradients block launch. `lambda_cf` is then
frozen for Phase M and Phase T.

### Step25 health gate

Step25 does not eliminate a run for weak action separation. It requires:

- every approved Q/K/V/O family has finite, nonzero gradient history;
- channel gates receive finite gradients and begin updating;
- every frozen family has zero gradients;
- FM and ranking loss remain finite;
- attention, geometry residual, and block-output RMS remain below 10x their
  preflight references and show no monotonic explosion;
- routing retention is at least 90%;
- no NaN, Inf, OOM, optimizer drift, or scheduler drift occurs.

Failure stops immediately. A healthy run continues to step100.

### Step100 mechanism gate

The fixed held-out audit-20 requires:

- reverse wins at least 11/20;
- hard shift, using the more competitive of `+1/-1`, wins at least 11/20;
- swap wins at least 11/20;
- all three mean ranking margins are positive;
- routing retention is at least 90%;
- correct FM regression is below 2%;
- position and velocity do not regress by more than 2% relative to the fresh
  parent.

The report includes gate-enabled and gate-zero ablations. The promotion gate is
evaluated with learned gates enabled.

If any negative remains at or below 10/20, any mean margin is non-positive, or
a health condition fails, v8 stops. It does not add trajectory loss, increase
LR, widen the band, or continue to step250.

## Phase T: trajectory-alignment training

Phase T starts only from a passing step100 checkpoint and runs through step250.
The architecture, data pool, replay lineage, topology, trainable whitelist,
and counterfactual objective remain unchanged.

It adds the validated frozen v6 gripper probe losses:

```text
L_T = weighted_FM(correct)
      + lambda_cf * L_cf
      + lambda_pos * L_position
      + lambda_vel * L_velocity
```

The labels, frame-local probe, visibility masks, per-arm semantics, and
held-out probe contract are reused without modification. The losses do not
create a new trajectory network and do not use SAM3.

At the Phase T boundary, `lambda_pos` and `lambda_vel` are calibrated once over
native Q/K/V/O gradients:

```text
RMS(lambda_pos * grad_QKVO(L_position))
    = 0.25 * RMS(grad_QKVO(weighted_FM))

RMS(lambda_vel * grad_QKVO(L_velocity))
    = 0.15 * RMS(grad_QKVO(weighted_FM))
```

The scalars are frozen through step250. Phase T does not reset optimizer state,
warmup, or the counterfactual calibration.

## Optimizer and schedule

One AdamW optimizer uses two explicit groups:

| parameter family | initial LR | weight decay |
|---|---:|---:|
| native Q/K/V/O in blocks 8-13 | `1e-6` | `0.01` |
| six geometry channel gates | `5e-5` | `0` |

- warmup: 25 optimizer steps;
- schedule: cosine decay to 20% of each initial LR at step250;
- global gradient clipping: L2 norm `1.0`;
- micro-batch: one per rank;
- activation checkpointing: production Wan policy;
- production attention dtype: bfloat16 where supported;
- optimizer states, gates, SE(3), ranking reductions, and calibration norms:
  FP32.

The checkpoint stores complete optimizer groups and state, scheduler state,
all calibrated loss weights, phase identity, topology, replay cursor, source
closure, parent hash, dataset hash, and exact update count. Resume rejects any
mismatch and cannot resume Phase T from a Phase M checkpoint that failed its
gate.

## Step250 hard gate

The fixed held-out audit-20 requires:

- reverse wins at least 14/20;
- hard shift wins at least 14/20;
- swap wins at least 14/20;
- all three mean ranking margins are positive;
- position error improves by more than 5% over the fresh parent;
- velocity error improves by more than 5% over the fresh parent;
- routing retention is at least 90%;
- correct FM regression is at most 2%.

The report again includes gate-enabled and gate-zero ablations.

If any hard gate fails, training stops at step250. There is no default step500,
rank increase, wider band, V-only follow-up, or additional loss sweep.

If the hard gate passes, decode 4-8 fixed RGB sanity episodes. Black/broken
video, severe arm disappearance, temporal-role collapse, or obvious visual
quality failure blocks matched fast20.

## Production smoke and memory gate

Before Phase M, the selected immutable topology runs three production-shape
iterations:

```text
81 frames
480x640
cached latent and text context
complete correct and wrong action conditioning
six blocks of trainable native Q/K/V/O
six shared-projection SE(3) attention paths
correct forward + wrong forward
backward -> optimizer.step -> zero_grad
```

CUDA peak statistics are reset after initialization and warmup. Every rank
records peak allocated memory, peak reserved memory, and step time. Every rank
must remain below 22 GiB allocated and 22 GiB reserved.

Smoke additionally proves:

- exact Q/K/V/O and gate trainable whitelist;
- nonzero gradients for all required native projection families;
- no gradients outside the whitelist;
- optimizer state exists for every trainable tensor;
- correct/wrong sequential gradients match the direct oracle;
- T5 and VAE are absent from the training process;
- no unexplained memory growth occurs across three iterations.

If single-GPU smoke fails memory, the same architecture may move to a fixed
FSDP topology after a separate smoke. The architecture or batch semantics are
not silently reduced to make single-GPU execution pass.

## RGB and matched fast20

The first video comparison contains only:

- clean S1A125;
- clean-gated-step10;
- v8 step250;
- one earlier v8 checkpoint only if it has a documented action/quality
  trade-off distinct from step250.

Generation parameters are identical. Promotion requires:

- failure-aware trajectory improvement of at least 5%;
- paired wins of at least 12/20;
- detector coverage drop no greater than 5 percentage points;
- no detector-failure regression hidden by survivor-only averaging;
- zero black/broken videos;
- bimanual and temporal-leakage guardrails not worse than
  clean-gated-step10.

VLM and JEPA remain no-regression evaluators for later full development
evaluation. They do not enter v8 training.

## Scale-up boundary

Passing step250 and matched fast20 makes this exact architecture eligible for
clean-5k preparation and longer exposure. Scale-up is a separate approved run;
it does not alter Q/K/V/O scope or add new modules.

If v8 fails action separation at step100, or fails both action separation and
trajectory improvement at step250, the next route is not another small branch.
The project must reconsider a wider native-Wan fine-tune and larger
action-video data under a new design.

## Provenance and artifacts

Every run records:

- source commit and recursive source-closure hash;
- parent path and SHA256;
- Wan model-manifest and source hashes;
- clean-1785 source/cache and balanced-replay hashes;
- audit-20 identities and hash;
- per-step negative family and shift direction;
- exact trainable names, shapes, dtypes, and parameter count;
- optimizer groups and scheduler state;
- `lambda_cf`, `lambda_pos`, `lambda_vel`, their calibration batches, and raw
  gradient norms;
- topology, rank mapping, dtype, environment, and GPU ownership receipt;
- step25 health, step100 mechanism, step250 hard-gate, gate-ablation, RGB, and
  matched-evaluation reports.

Persistent outputs are written atomically under a dedicated v8 directory in
`/data/di/worldarena2_track1_20260815`. Existing v7/v7.1 artifacts remain
read-only provenance inputs and are never overwritten.

## Required tests

Implementation must provide focused tests for:

1. exact blocks `8-13` and prohibition of block scanning;
2. exact native Q/K/V/O plus six-gate trainable whitelist;
3. frozen parent, V outside the band, MLP, norms, cross-attention, VAE, and T5;
4. shared Q/K/V/O between native and geometry paths with no geometry LoRA or
   duplicate full projection copy;
5. zero-gate output equality with the clean parent;
6. gate-enabled versus forced-zero checkpoint audit;
7. anchored reverse, shift+1, shift-1, and swap complete-action transforms;
8. fixed ranking mask under changed conditioning support;
9. direct pairwise and sequential analytic-VJP gradient equality;
10. counterfactual calibration over native Q/K/V/O only;
11. v6 position/velocity calibration at the Phase T boundary;
12. audit/replay/dev-fast20 isolation and source-train partition provenance;
13. Phase M to Phase T gate and resume rejection after a failed step100;
14. exact optimizer LR, weight decay, scheduler, clipping, and checkpoint state;
15. step25/100/250 gate semantics and absence of an automatic step500;
16. production memory, gradient, optimizer-state, and T5/VAE-absence smoke;
17. checkpoint rejection after source, data, topology, phase, or replay drift;
18. zero persistent writes outside the authoritative `/data/di` root.

## Final decision tree

```text
clean-gated-step10
        |
fixed blocks 8-13
native Q/K/V/O trainable
+ shared deterministic SE(3) attention
        |
Phase M, steps 0-100
FM + full-action CF
        |
        +-- step100 action separation fails
        |      -> stop v8
        |
        +-- step100 passes
               |
          Phase T, steps 100-250
          + position + velocity
               |
               +-- step250 fails
               |      -> stop, report attribution
               |
               +-- step250 passes
                      -> RGB sanity
                      -> matched fast20
                      -> separately approved clean-5k scale-up
```

This design intentionally uses one architecture, one parent, one dataset, one
replay lineage, and one ordered loss transition. It is bold in where it trains
Wan and conservative in how many simultaneous hypotheses it asks the first run
to resolve.
