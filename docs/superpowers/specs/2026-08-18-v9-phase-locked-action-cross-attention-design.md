# Wan-Action v9 Phase-Locked Action Cross-Attention Design

## Decision

The approved experiment is `v9-phase-locked-action-cross-attention`.

v8 proved that a trainable native Wan attention band can learn action reversal
and arm identity, but it failed one-latent-step temporal displacement:

- reverse: 16/20 held-out wins;
- swap: 17/20 held-out wins;
- hard shift: 5/20 held-out wins with negative mean margin.

v9 changes one architectural fact: the action-to-video temporal relationship
is enforced by wiring. Each destination video latent time may read only the
corresponding preceding action transition. The model is not asked to infer
which interval belongs to which latent frame.

v9 starts from the immutable clean-gated-step10 parent. It does not warm-start
from v8 and does not reuse the v8 optimizer state.

## Fixed boundaries

- Parent SHA256:
  `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- Training data: provenance-bound clean-1785 minus the fixed audit20.
- Development evaluation: fixed audit20 and dev-fast20 remain disjoint from
  optimizer replay.
- Native Wan blocks 8-13 self-attention Q/K/V/O remain trainable.
- A new phase-locked action cross-attention module is installed in blocks
  8-13.
- The frozen support-gated raster parent remains active and unchanged.
- The v8 arm-grouped SE(3) residual branch is removed.
- FFN/MLP, norms, QK normalization, text cross-attention, other Wan blocks,
  the raster parent, probe, VAE, and T5 remain frozen.
- No explicit phase embedding, absolute-pose token, contact token,
  coordination token, Depth, V-JEPA, DMD, GAN, or geometry LoRA is allowed in
  the first v9 experiment.
- Formal artifacts are written only below
  `/data/di/worldarena2_track1_20260815`.
- GPU ownership is checked before launch. v9 does not stop or reuse another
  user's process.

## Temporal contract

The 81 RGB frames map to 21 video latent frames and 20 action transitions:

```text
latent 0                    <- no transition, exact zero cross-attention
latent 1                    <- transition 0: state 0 -> state 1
...
latent 20                   <- transition 19: state 19 -> state 20
```

For arm `k` and transition `i`:

```text
DeltaG[i,k] = inverse(G[i,k]) @ G[i+1,k]
xi[i,k]     = log_SE3(DeltaG[i,k])
```

The transition index is the destination slot contract. It is not encoded as
action content. In particular, a shift counterfactual changes only the motion
content placed in a destination slot. The destination slot, attention mask,
and any slot-local bookkeeping remain unchanged. The model cannot identify a
wrong shift from a source phase ID.

## Four modality tokens

Each arm and transition owns exactly four tokens.

### Translation token

Raw values:

```text
dx, dy, dz, translation_magnitude
```

The values come from the translational part of `log_SE3(DeltaG)`.

### Rotation token

Raw values:

```text
rx, ry, rz, rotation_angle
```

`rx,ry,rz` are the SO(3) log-map rotation vector. Quaternion input is
forbidden, avoiding the `q/-q` ambiguity.

### Image-motion token

Raw values:

```text
u_start, v_start, u_end, v_end, delta_u, delta_v
```

The EEF locations are derived from the existing kinematically identified arm
trajectory. They are normalized using train-only statistics. No dense raster
is tokenized here: the frozen raster parent remains responsible for spatial
support and dense image location.

### Gripper token

Raw values:

```text
opening_start, opening_end, delta_opening
```

Each modality uses its own small MLP into the common action width 256:

```text
raw feature -> Linear -> SiLU -> Linear(256)
            + learned modality-type embedding
```

The four tokenizer MLPs and type embeddings are shared between left and right
arms and across all six Wan blocks. Arm identity is not learned from an
embedding: it is enforced by the token bank and head mask.

## Transition presence and motion activity

`arm_present[i,k]` is true only when both endpoint states for transition `i`
are valid for arm `k`. A false value makes every cross-attention output from
that arm group exactly zero.

The interval-level ranking loss uses a correct-action motion-activity mask.
A transition is active when at least one correct-action quantity exceeds its
fixed physical threshold:

- translation magnitude: 1 mm;
- rotation angle: 0.5 degrees;
- EEF image displacement: 0.25 pixels on the 80x60 control grid;
- gripper opening change: 0.01.

Quiet transitions receive zero counterfactual weight but retain normal flow
matching. A sample with no active transition is FM-only for that step and is
reported as counterfactual-ineligible; it is never silently converted into a
global background ranking loss. Calibration and held-out audit samples must
contain at least one active transition.

## Strict phase-locked arm-grouped cross-attention

Each selected block owns an eight-head cross-attention module with inner width
768 and head dimension 96.

```text
visual hidden: 3072 -> Q: 768
action token:   256  -> K: 768
                       V: 768
cross output:   768  -> O: 3072
```

Head ownership is fixed:

```text
heads 0-3 -> left bank only
heads 4-7 -> right bank only
```

Every head within an arm group may attend all four modality tokens for that
arm and interval. Heads are not assigned one-to-one to modalities.

For a visual token at latent time `t > 0`:

```text
left heads  may read only action[t-1, left,  0:4]
right heads may read only action[t-1, right, 0:4]
```

No head may read action content from `t-2`, `t`, or another arm. Latent time
zero bypasses the module and receives an exact-zero cross-attention residual.

The formal implementation does not allocate a broad attention mask. It
reshapes visual tokens to `(batch,time,space,heads,head_dim)`, indexes the
single legal interval bank, and evaluates attention over four keys. This
makes illegal temporal reads structurally impossible and keeps activation
cost bounded.

The cross-attention output is projected to width 3072 and multiplied by one
FP32 zero-initialized channel gate per block. At initialization, v9 is exactly
equal to the clean parent. Cross-attention Q/K/V/O begin receiving gradients
after the gate's first update; the step25 health gate requires all cross
families to have nonzero finite gradient history after step1.

## Wan block integration

For blocks 8-13:

```text
raw visual hidden
    |-- native self-attention using trainable native Q/K/V/O
    |-- phase-locked action cross-attention
    |       Q from the same block input
    |       K/V from the strict local four-token arm banks
    |       zero-init gated output
    +-- sum at the validated self-attention output location
            -> original Wan modulation/residual contract
            -> original FFN
```

The action module does not duplicate native Wan projections and does not
modify the text cross-attention path. Activation-checkpoint condition leases
must preserve the exact action-token bank across recomputation and release it
after backward.

Approximate new control capacity per block is 5.1M parameters; six blocks add
about 30.7M parameters plus a shared sub-million-parameter tokenizer. Together
with the 226.6M native Q/K/V/O band, the formal runtime records the exact
trainable count, names, shapes, and dtypes.

## Counterfactual data contract

The existing correct and complete negative caches are converted into compact
transition features for:

- correct;
- reverse;
- shift+1;
- shift-1;
- swap.

Every derived sidecar stores the four raw modality arrays, transition
presence, and activity. It is bound to the source action cache, HDF5, URDF,
clean-1785 manifest, and train-only normalization receipt. Invalid or stale
sidecars are quarantined and rebuilt sample-by-sample.

Shift uses hold padding at the exposed boundary. It moves all four modality
contents together but does not move the destination slot or attach a source
phase identifier. Reverse and swap preserve legal initial arm anchors and
kinematic identity as in the existing complete-action contract.

## Phase-M objective

Phase M runs from step 0 through step 250.

```text
L_M = weighted_FM(correct) + lambda_cf * L_interval_CF
```

For each active transition `i`, the fixed ranking mask is derived from the
correct action support and valid latent mask at destination latent `i+1`:

```text
E_i(action) = weighted unreduced FM at latent i+1
              inside correct robot/action support
```

The shift/reverse/swap wrong action may not change the comparison mask or its
denominator.

```text
L_interval_CF = weighted_mean_i(
    softplus((E_i(correct) - E_i(wrong)) / tau),
    correct_motion_activity[i]
)

tau = 0.1
```

One wrong family is graph-bearing per optimizer step. A fixed ten-step cycle
provides exact 60/20/20 exposure:

```text
shift+1, reverse, shift-1, swap, shift+1,
shift-1, shift+1, reverse, shift-1, swap
```

The correct and wrong forwards share sample, noise, timestep, target, valid
mask, and ranking mask. Sequential analytic VJP is used so the two 5B graphs
are never resident simultaneously. A tiny direct-pairwise oracle must match
the sequential gradients.

`lambda_cf` is calibrated once over the common native Q/K/V/O and six new
cross-attention gates:

```text
RMS(lambda_cf * grad(L_interval_CF))
    = 0.5 * RMS(grad(weighted_FM))
```

The zero-gated cross Q/K/V/O and tokenizer parameters are excluded from the
step0 denominator; they are required to acquire nonzero gradients after the
gate moves.

## Optimizer and schedule

One AdamW optimizer has three explicit groups:

| family | LR | weight decay |
|---|---:|---:|
| native Wan Q/K/V/O, blocks 8-13 | `1e-6` | `0.01` |
| action tokenizer and cross Q/K/V/O | `1e-4` | `0.01` |
| six FP32 channel gates | `5e-5` | `0` |

- warmup: 25 optimizer steps;
- schedule: cosine decay to 20% of each group LR at step500;
- global gradient clipping: L2 norm 1.0;
- micro-batch: one per rank;
- production attention/model dtype: bfloat16 where supported;
- gates, reductions, calibration norms, and optimizer states: FP32.

## Checkpoints and gates

### Step25 health

Step25 requires:

- finite, nonzero native Q/K/V/O gradient history;
- finite, nonzero cross-attention Q/K/V/O, tokenizer, and gate gradient history
  after step1;
- zero gradients outside the exact whitelist;
- finite FM/ranking losses and optimizer state;
- no residual family above 10x its preflight reference;
- no NaN, Inf, OOM, scheduler drift, or topology drift.

Weak held-out separation alone does not stop a healthy step25 run.

### Step100 first audit

The fixed audit20 requires:

- shift+1 wins at least 12/20;
- shift-1 wins at least 12/20;
- reverse wins at least 12/20;
- swap wins at least 12/20;
- every mean margin is positive;
- routing retention at least 90%;
- correct FM regression at most 2%.

If either shift direction remains at 6/20 or below, v9 stops immediately. A
borderline result between 7/20 and 11/20 may continue only when both shift
margins and the last-half training trend are positive.

### Step250 mechanism hard gate

The fixed audit20 requires:

- shift+1, shift-1, reverse, and swap each at least 14/20;
- all four mean margins positive;
- position improvement over the fresh parent strictly above 5%;
- velocity improvement strictly above 5%;
- routing retention at least 90%;
- correct FM regression at most 2%;
- gate-enabled results no worse than the gate-zero ablation.

Failure ends v9 Phase M. No additional token, wider temporal window, more
blocks, or automatic continuation is allowed.

## Conditional Phase T

Phase T runs from step250 to step500 only after the step250 hard gate passes.
It keeps the same architecture, replay lineage, optimizer state, and
counterfactual objective, and adds the frozen v6 probe losses:

```text
L_T = L_M + lambda_position * L_position
          + lambda_velocity * L_velocity
```

At the boundary, Q/K/V/O-family gradient norms calibrate position to 0.25x FM
and velocity to 0.15x FM. The calibrated weights are frozen. Phase T does not
reset the optimizer or warmup.

Step500 requires the step250 action gates to remain satisfied, position and
velocity to improve further or stay within one estimated noise band, FM
regression at most 2%, and no visual-quality guardrail regression. Only a
passing step500 checkpoint may proceed to RGB sanity and matched fast20.

## Production smoke

Before Phase M, the selected immutable topology runs three full iterations:

```text
correct preview + wrong preview
correct graph backward
wrong graph backward
optimizer.step
zero_grad
```

The smoke uses 81 frames, 480x640, cached latent/text inputs, all six native
trainable blocks, all six phase-locked cross-attention modules, and complete
counterfactual conditions. After warm initialization, every rank must remain
below 22 GiB allocated and reserved.

Smoke also proves:

- exact zero equality at initialization;
- exact latent0 zero residual;
- no temporal or cross-arm reads outside the legal bank;
- absent-arm exact zero;
- all approved trainable families receive gradients by iteration three;
- no frozen parameter receives a gradient;
- T5 and VAE are absent from the process;
- optimizer state exists for every trainable tensor;
- no unexplained memory growth across three iterations.

If single-GPU smoke fails only memory, a separately approved fixed FSDP
topology may be tested. The model contract and batch semantics are not reduced
silently.

## Required tests

1. exact 81-to-21-to-20 transition mapping;
2. FP64 SE(3)/SO(3) log-map fixtures including small angles and near-pi cases;
3. exact four modality tokens and no fifth token;
4. train-only normalization and zero split leakage;
5. strict destination-slot behavior under shift counterfactuals;
6. no source phase ID or explicit phase embedding;
7. latent0 exact zero;
8. left/right head-bank isolation and overlapping spatial support allowance;
9. absent-arm exact zero;
10. each head attends four tokens, proving Q/K have nonzero gradients after
    gate activation;
11. exact blocks 8-13 and trainable whitelist;
12. zero-init equality with the clean parent;
13. checkpoint-replay condition lifecycle;
14. fixed per-interval ranking mask and active-window reduction;
15. exact 60/20/20 wrong-family cycle with balanced shift signs;
16. sequential VJP equality to a direct oracle;
17. step25/100/250/500 gate semantics;
18. Phase T rejection without a passing step250 receipt;
19. optimizer, scheduler, checkpoint, resume, source, data, and topology drift
    rejection;
20. three-iteration production memory/gradient/hot-path smoke.

## Final decision tree

```text
clean-gated-step10
        |
v9 strict four-token phase-locked cross-attention
+ native Q/K/V/O blocks 8-13
        |
Phase M 0-100
        |
        +-- shift direction <=6/20 -> stop
        |
        +-- first gate supports continuation
               |
          Phase M 100-250
               |
               +-- hard gate fails -> stop
               |
               +-- hard gate passes
                      |
                 Phase T 250-500
                      |
                      +-- final gate fails -> stop
                      |
                      +-- final gate passes
                             -> RGB sanity
                             -> matched fast20
                             -> separately approved scale-up
```

This experiment tests one hypothesis: strict interval wiring plus
content-based four-token selection can turn the v8 temporal-shift failure into
held-out action separation without sacrificing the reverse/swap gains.
