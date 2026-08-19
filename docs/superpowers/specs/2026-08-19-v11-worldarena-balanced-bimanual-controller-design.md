# v11 WorldArena-Balanced Bimanual World Controller Design

Date: 2026-08-19

Status: proposed for user review

## 1. Decision

The next experiment is `v11-worldarena-balanced-bimanual-world-controller`.
Its complete roadmap contains persistent left-arm, right-arm, object, and narrow
coordination streams around a frozen Wan visual prior. The implementation
authorized by this specification is deliberately narrower: only Stage A, the
closed-loop left/right arm controller, is implemented and trained first.

Object evolution, late coordination, V-JEPA supervision, depth supervision,
instruction routing, native Wan delta projections, and dataset expansion are
post-gate extensions. They must not enter the Stage-A source closure or
optimizer whitelist.

This boundary preserves causal attribution. Stage A answers exactly one
question:

> Can persistent, physically separate left/right controller states read the
> current frozen Wan visual state and independently correct the commanded arm
> regions without degrading the parent video distribution?

## 2. Evidence and motivation

The completed v10 run established:

- phase `+1` separation was `19/20`;
- phase `-1` separation was `20/20`;
- routing retention was `100%`;
- FM regression at step 150 was approximately `-0.026%`;
- reverse and swap wins improved from `9/20` to `12/20`;
- swap mean margin remained negative;
- the relation-enabled path did not beat the relation-forced-zero path;
- hidden EEF supervision converged without establishing final arm binding.

The next architecture therefore must not extend v10 relation Q/K or train more
native Wan attention. It needs physically separate arm state, visual feedback,
and local write authority.

## 3. Parent and immutable lineage

The parent is the existing `clean-gated-step10` checkpoint, not v10 step 50.

Reasons:

1. the fresh clean parent already had the strong phase `+1/-1` audit result;
2. v10 step 50 contains the failed relation path;
3. retaining that path pollutes attribution;
4. deleting it would make the claimed v10-step50 numerical equivalence false.

The Stage-A implementation freezes:

- the complete Wan backbone;
- all native Wan Q/K/V/O, FFN, normalization, and text cross-attention;
- the support-gated raster/flow parent adapter;
- T5 and VAE, which remain outside the training hot path;
- every v10 relation module, which must not exist in the v11 model;
- all object, coordination, depth, semantic, and native-delta modules.

Every new Stage-A output projection and residual gate is zero initialized. At
step 0, v11 must reproduce the clean parent within the declared BF16 numeric
tolerance.

The checkpoint and receipts bind at least:

- parent SHA256;
- source-closure SHA256;
- optimizer/audit/dev manifest SHA256 values;
- replay SHA256;
- cache and normalization SHA256 values;
- exact topology and CUDA visibility;
- initialization seed;
- exact trainable parameter inventory;
- loss-calibration receipt;
- samples seen and optimizer updates.

## 4. Data and leakage boundary

Stage A reuses the frozen v10 data split:

- source action-video rows: 2,100;
- optimizer rows: 2,060;
- audit rows: 20;
- dev rows: 20.

Optimizer, audit, and dev episode identities remain disjoint. The official test
set is unavailable to all training, cache, calibration, audit, and checkpoint
selection code.

No data-scale experiment is combined with Stage A. The existing deterministic
replay order is reused or deterministically extended to one full 2,060-sample
exposure without changing sample weights.

Training-time labels are limited to already validated cached inputs:

- correct action raster/support;
- per-arm action content and presence;
- RGB-observable EEF targets and validity masks;
- latent target/flow-matching inputs.

SAM3 object masks, V-JEPA features, object flow, depth teachers, contact labels,
and semantic object labels are not required by Stage A.

## 5. Stage-A architecture

### 5.1 Persistent arm state

Each arm owns a physically separate state tensor:

```text
C_left, C_right: (B, 21, 4, 384)
```

The four modality slots are:

1. translation;
2. rotation;
3. image motion;
4. gripper.

The time contract is:

```text
C[:, 0] = exact zero
C[:, t] = action interval t-1, for t=1...20
```

Stage A has no cross-time slot self-attention. A slot at latent time `t` may
only interact with visual tokens at the same latent time. Destination-time
identity is fixed; counterfactual action content cannot change the destination
slot.

Left and right tokenizers, visual-read modules, state updaters, visual-write
modules, EEF heads, and gates have no shared parameters.

### 5.2 Controller stages

Closed-loop stages execute after frozen Wan blocks:

```text
6, 16, 24
```

The late block-24 stage ensures that an arm correction cannot be completely
diluted by many subsequent frozen blocks.

Each stage operates functionally:

```text
(X, C_left, C_right)
    -> (X_next, C_left_next, C_right_next)
```

Slot state must not be stored in module attributes, global variables, or hook
caches. Activation-checkpoint inputs and outputs include all three tensors.
Controller dropout is zero.

### 5.3 Deterministic sparse gather

For each arm and latent time, gather at most 48 visual tokens from the correct
trajectory tube:

```text
current EEF support
union destination EEF support
union swept corridor
```

Selection is deterministic. Ties are resolved by flattened raster index.
Fewer than 48 tokens are padded and accompanied by an explicit validity mask.
Counterfactual action content never changes gather indices, write masks, or
loss masks.

The gather is local to the current latent time. There is no full-frame or
quadratic visual cross-attention.

### 5.4 Read, update, and write

For arm `k` at stage `s`:

```text
DeltaC_visual[k,s] = Read_k(C_k, Gather_k(X))
C_k_next = Update_k(C_k, DeltaC_visual[k,s])
DeltaX_k = Write_k(Gather_k(X), C_k_next)
X_next = X + M_left * gate_left * DeltaX_left
           + M_right * gate_right * DeltaX_right
```

The visual read uses arm slots as queries and local visual tokens as keys and
values. `DeltaC_visual` must become exact zero when visual values are forced to
zero.

The write uses local visual tokens as queries and the updated arm state as keys
and values. Each arm owns an independent output projection and FP32 zero gate.

Direct writeback outside that arm's fixed correct support is structurally zero.
Left and right supports may overlap; both residuals are added. There is no
winner-take-all routing or half-plane identity heuristic.

If `arm_present=false`, that arm's initial slots, updated slots, reads, writes,
and residuals are exact zero. The frozen Wan visual path remains active.

### 5.5 Visual-read-dependent EEF heads

An arm EEF head reads only `DeltaC_visual`, never the complete action-bearing
slot state. This prevents the head from directly decoding commanded EEF from
the action input.

The head predicts cached RGB-observable EEF heatmaps. Loss is applied only to
valid arm/time labels. A forced-zero visual-value audit must show that visual
read materially contributes to EEF accuracy.

### 5.6 Parameter budget

Stage A trains only:

- left and right action tokenizers/updaters;
- left and right sparse visual-read projections;
- left and right sparse visual-write projections;
- six FP32 stage/arm zero gates;
- left and right visual-read-dependent EEF heads.

The expected trainable range is 40M to 80M parameters. Preflight rejects a
count outside the final declared range and rejects any trainable frozen-Wan or
parent parameter.

## 6. Counterfactual contract

### 6.1 Wrong-left and wrong-right

For `wrong-left`, the following remain byte-identical to correct-left:

- arm identity;
- destination phase/time;
- arm presence;
- initial anchor;
- gather indices;
- support/write mask;
- supervision mask.

Only relative translation, relative rotation, image-motion content, and
gripper delta are replaced with right-arm motion content. Replacement motion
is re-anchored to the left initial pose. Right action remains correct.

`wrong-right` is symmetric.

### 6.2 Active-arm null

For a single-arm sample, the negative keeps:

- `arm_present=1`;
- the same initial anchor;
- the same support and gather indices;
- the same opening state;
- the same phase slot.

Relative motion and gripper delta are set to zero. It must not use
`arm_present=0`, which would leak the negative identity.

### 6.3 Per-arm energy

Binding energy is unreduced flow-matching error evaluated on the fixed correct
RGB-observable arm region. Left and right energies are separate, so one arm
cannot cancel the other.

Only eligible arm-events enter a denominator. Metrics must report eligible,
invalid, wins, paired win rate, mean margin, and median margin independently
for left and right.

### 6.4 Exact sequential-gradient execution

Peak memory must never retain correct and wrong full computation graphs
simultaneously. With dropout disabled and unchanged parameters, one optimizer
step executes:

1. correct no-grad forward to obtain `E_correct_ref`;
2. wrong grad forward and backward of half the pairwise objective using the
   detached correct reference;
3. release the wrong graph while retaining detached `E_wrong_ref`;
4. correct grad forward and backward of FM, EEF, and the other half of the
   pairwise objective using the detached wrong reference;
5. optimizer step and zero-grad.

The two half-objectives reproduce the two-sided gradient of the pairwise loss
at the same parameters. A test compares this sequential implementation with a
small simultaneous-graph oracle.

Negative type cycles deterministically across wrong-left, wrong-right, and
active-arm-null when eligible. Ineligible types are skipped by a recorded,
deterministic rule rather than silently changing the denominator.

## 7. Stage-A objective

Stage A uses only:

```text
L = L_FM + lambda_binding * L_per_arm_binding
         + lambda_eef * L_visual_read_EEF
```

No position, velocity, phase, locality, background-preservation, coordination,
object, V-JEPA, depth, contact, semantic, or smoothness loss is active.

Although the frozen parent is protected structurally by local zero writeback,
final background and photometric preservation are still measured as audits.

Lambda values are calibrated once on a fixed calibration batch using gradients
on the actual visual-write projections:

```text
FM gradient target      1.0
binding gradient target 0.5
EEF gradient target     0.2
```

Calibration is frozen and SHA-bound before smoke. Calibration must reject
non-finite gradients, zero eligible supervision, or a lambda outside its
reviewed safety bounds.

## 8. Single-GPU execution contract

Initial execution topology is exactly one opportunistic RTX 4090:

```text
CUDA_VISIBLE_DEVICES=6
world_size=1
microbatch=1
```

At the time of design, GPUs 0-5 are occupied by Lingbot and GPU7 hosts a model
service. The launcher must revalidate GPU6 ownership, compute PID, memory, and
utilization immediately before CUDA initialization. It must not stop or alter
other processes.

Precision:

```text
frozen Wan/parent: BF16
slot/read/write: BF16
residual gates: FP32
AdamW state: FP32
dropout: 0
```

The production smoke performs at least three complete
forward/backward/step/zero-grad cycles after warmup and peak reset.

Memory gates per iteration:

```text
recommended reserved <= 20.5 GiB
absolute allocated < 22 GiB
absolute reserved < 22 GiB
```

Any OOM, NaN/Inf, unexplained monotonic memory growth, or violation on any
iteration blocks training.

## 9. Checkpoints and gates

Selection is based on `samples_seen`, which equals optimizer updates in the
single-GPU microbatch-one run.

### 9.1 Step 0 invariants

Before training:

- v11 equals clean-gated-step10 within the declared tolerance;
- `slot[:,0]` is exact zero;
- left output has zero gradient with respect to right action;
- right output has zero gradient with respect to left action;
- direct writeback is exact zero outside each arm support;
- absent-arm slot and writeback are exact zero;
- overlap contains both independent residuals and their sum;
- checkpointed forward/recompute matches non-checkpointed execution;
- visual-value forced-zero makes `DeltaC_visual` exact zero;
- no frozen parameter is trainable.

### 9.2 Step 100 health gate

The run continues only if:

- tokenizer, updater, read, write, gate, and EEF gradients are finite/nonzero;
- every frozen Wan/parent gradient is absent;
- FM is finite;
- allocated and reserved memory remain below 22 GiB;
- neither arm slot collapses;
- controller residual RMS does not dominate the parent hidden RMS;
- phase `+1/-1` does not catastrophically regress;
- no condition or checkpoint state leaks across forwards.

This is an engineering health gate, not promotion.

### 9.3 Step 500 direction gate

Step 500 is a direction audit, not the final mechanism verdict.

Continue when:

- left paired binding win rate is greater than 50%;
- right paired binding win rate is greater than 50%;
- at least one arm mean margin is positive;
- the other arm does not show persistent severe negative drift;
- forced-zero locality ratios exceed 1.2 on both arms;
- phase `+1/-1` remains at least 18/20;
- routing retention is at least 90%;
- FM regression is at most 2%.

Early termination at step 500 requires all of the following simultaneously:

- left mean margin is non-positive;
- right mean margin is non-positive;
- both locality ratios are at most 1.05;
- controller-enabled versus forced-zero mean improvement is at most 1%.

Marginal results continue to the full-exposure gate instead of being killed by
one noisy audit20 measurement.

### 9.4 Step 2060 mechanism hard gate

After one exact optimizer-pool exposure, promotion requires:

- left paired binding win rate at least 60% and mean margin positive;
- right paired binding win rate at least 60% and mean margin positive;
- overall swap wins at least 14/20;
- left forced-zero locality ratio at least 1.5;
- right forced-zero locality ratio at least 1.5;
- bimanual/crossing subset paired win rate at least 50% and mean margin
  non-negative;
- phase `+1/-1` at least 18/20;
- routing retention at least 90%;
- FM regression at most 2%;
- controller-enabled correct-region energy at least 2% better than
  controller-forced-zero with at least 60% paired wins;
- visual-read-enabled EEF error at least 5% lower than visual-values-forced-zero
  with at least 60% paired wins;
- background and photometric proxy regressions remain inside the measured
  clean-parent noise band;
- black/broken output rate remains zero on the bounded decoded sanity set.

Failure does not authorize 4,120 samples automatically. A second exposure is
allowed only if all mechanism metrics are positive and the 500-to-2060 trend
continues improving without quality regression.

## 10. Post-gate roadmap, excluded from Stage A

Only after the step-2060 hard gate passes may a new reviewed phase add:

### Stage B: object and coordination

- two persistent object slots per time;
- first-frame object proposals and cached training labels;
- object mask/centroid/motion heads;
- a narrow coordination slot at stages 16 and 24;
- object-flow, contact, object-feature, and bimanual supervision.

No future GT mask is permitted as model input. SAM3, V-JEPA, RAFT, and depth
teachers remain offline cache producers or frozen evaluators.

### Stage C: metric balance

- position/velocity trajectory supervision;
- final local latent-flow correction;
- optional depth/normal audit and only then an independently reviewed scene
  adapter;
- official WorldArena, VLM, and JEPA development evaluation;
- full no-regression gates relative to the frozen parent.

Instruction Role Router, six-stage expansion, native arm-head delta V/O, and
clean-5k are separate ablations. None is implied by a Stage-A pass.

## 11. Official metric strategy

The complete roadmap assigns responsibilities rather than promising that every
metric will improve:

- frozen Wan and structural local writeback protect image, aesthetic,
  background, photometric, depth, and perspective baselines;
- arm streams target trajectory, action following, flow, dynamic degree, and
  smoothness;
- future object/coordination phases target interaction, subject consistency,
  JEPA, and semantic alignment;
- frozen T5/text cross-attention protects instruction following until a
  separately gated router is justified.

Internal proxies only select candidates. A final quality claim requires the
pinned WorldArena base, VLM, and JEPA development evaluators. Official test is
run only after checkpoint and inference parameters are frozen.

## 12. Required tests

Implementation must include focused CPU/Torch tests for:

- exact 21-time slot packing and latent0 zero;
- no cross-time slot attention;
- left/right parameter disjointness;
- cross-arm gradient zeros;
- deterministic gather and padded validity masks;
- counterfactual mask/anchor/phase invariance;
- SE(3) re-anchoring of wrong-arm motion;
- active-arm-null retaining presence and anchor;
- absent-arm exact-zero behavior;
- overlapping support additive behavior;
- support-outside direct-write exact zero;
- visual-read forced-zero behavior;
- EEF heads reading only visual residuals;
- functional activation-checkpoint replay;
- sequential pairwise gradient equality against a simultaneous oracle;
- exact optimizer whitelist and no frozen gradient;
- zero-init parent equivalence;
- calibration fail-closed behavior;
- checkpoint/resume lineage and samples-seen validation;
- gate boundary and denominator behavior;
- launcher GPU6 ownership and output-root guards.

Remote preflight must run the focused suite with zero unexpected skips before
production smoke.

## 13. Artifacts and output root

All persistent artifacts are written under:

```text
/data/di/worldarena2_track1_20260815/runs/
v11-worldarena-balanced-bimanual-world-controller/
```

Required artifacts include:

- immutable data/replay receipts;
- source closure and environment manifest;
- loss calibration;
- production smoke report;
- step-100, step-500, and step-2060 checkpoints;
- audit JSON with per-arm eligible denominators;
- training JSONL;
- gate decisions and failure reasons;
- bounded decoded sanity outputs only after mechanism eligibility.

No persistent output may be written outside the formal `/data/di` project
root.

## 14. Failure decisions

The following outcomes are pre-registered:

1. visual-read EEF improves, locality is strong, but final binding remains
   weak: write authority is insufficient; a new experiment may add independent
   head-bank delta V/O;
2. EEF performs equally when visual values are zero: the controller is using an
   action shortcut; stop this architecture rather than adding capacity;
3. left/right locality and binding both fail: persistent slots do not solve arm
   binding; stop before object/coordination work;
4. mechanism passes but video quality regresses: tighten local residual budget
   or parent preservation before object expansion;
5. mechanism and bounded RGB pass: proceed to a separately reviewed Stage B,
   not directly to official test.

## 15. Acceptance boundary

Stage A is successful only when the full-exposure hard gate proves all three:

1. left and right action semantics are separately identifiable;
2. each controller causally changes its own visual region more than the other
   arm/background;
3. the visual-read path materially improves the controller state over an
   action-only shortcut.

Until then, v11 is a mechanism experiment, not a leaderboard candidate.
