# Wan v9 Phase-Locked Action Cross-Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a bounded v9 experiment that hard-wires each video latent time to the preceding per-arm action interval through four-token local cross-attention, then accepts continuation only when held-out temporal-shift separation improves.

**Architecture:** Keep the immutable clean-gated-step10 raster parent. Train native Wan self-attention Q/K/V/O only in blocks 8-13 and add one independent eight-head cross-attention module per selected block. Each visual time `t>0` reads exactly four modality tokens from transition `t-1` for its own arm head group; latent0 and absent-arm outputs are exact zero. Phase M uses correct weighted flow matching plus an interval-local counterfactual ranking loss; Phase T is permitted only after the step250 mechanism gate.

**Tech Stack:** Python 3.11, PyTorch, Wan2.2 TI2V 5B, NumPy, pytest, HDF5/NPZ caches, AdamW, activation checkpointing, single-GPU CUDA smoke on GPU6.

**Spec:** `docs/superpowers/specs/2026-08-18-v9-phase-locked-action-cross-attention-design.md`

## Global Constraints

- Do not modify unrelated dirty files, especially `src/worldarena_baseline/skeleton.py`, `src/worldarena_baseline/cli.py`, `tests/test_skeleton.py`, or `tests/test_scripts.py`.
- Do not warm-start from v8. The only model parent is the pinned clean-gated-step10 SHA256 `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- The optimizer pool is clean-1785 minus the fixed audit20. Audit20 and dev-fast20 must never enter replay, normalization statistics, calibration, or optimizer batches.
- All persistent remote outputs must resolve below `/data/di/worldarena2_track1_20260815`.
- Formal GPU execution is restricted to GPU6 unless the user explicitly changes the topology. Never stop another user's process.
- T5 and VAE remain absent from the training process; training reads only cached video latents, cached text context, frozen raster conditions, and v9 transition sidecars.
- Implement with TDD: create a failing focused test, run it and record the expected failure, implement the smallest production change, rerun to green, then commit.
- Never claim completion from local skipped Torch tests. The remote formal Python must report zero skips for the v9 focused suite before CUDA smoke.
- Phase progression is fail-closed: no step250 after a failed step100 decision, and no Phase T after a failed step250 hard gate.

---

## Task 1: Compact Transition Feature and Cache Contract

**Files:**

- Create: `src/worldarena_baseline/wan_v9_transition.py`
- Create: `scripts/cache_wan_v9_transitions.py`
- Create: `tests/test_wan_v9_transition.py`

- [ ] **Step 1: Write failing geometry and 81-to-21 mapping tests**

Test these public contracts:

```python
from worldarena_baseline.wan_v9_transition import (
    LATENT_ENDPOINT_FRAMES,
    relative_transition_features,
    so3_log_map,
)

def test_endpoint_mapping_is_exact():
    assert LATENT_ENDPOINT_FRAMES == (0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
                                      40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80)

def test_so3_log_is_sign_unambiguous_and_stable():
    # identity, 1e-8 rad, ordinary rotation, and pi-epsilon fixtures
    ...

def test_relative_features_have_20_transitions_per_arm():
    result = relative_transition_features(states, present)
    assert result.translation.shape == (2, 20, 4)
    assert result.rotation.shape == (2, 20, 4)
```

Run:

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_transition.py -q
```

Expected: collection fails because `wan_v9_transition` does not exist.

- [ ] **Step 2: Implement FP64 SO(3)/SE(3) transition extraction**

Implement:

```python
LATENT_ENDPOINT_FRAMES = tuple([0] + [4 * index for index in range(1, 21)])

@dataclass(frozen=True)
class TransitionFeatures:
    translation: np.ndarray  # (2,20,4): dx,dy,dz,|d|
    rotation: np.ndarray     # (2,20,4): logR xyz,angle
    image: np.ndarray        # (2,20,6): u0,v0,u1,v1,du,dv
    gripper: np.ndarray      # (2,20,3): open0,open1,delta
    arm_present: np.ndarray  # (2,20), bool
    motion_active: np.ndarray  # (2,20), bool, correct action only
```

Requirements:

- compute `inverse(G_i) @ G_{i+1}` in FP64;
- use a stable SO(3) log-map with explicit small-angle and near-pi branches;
- obtain EEF UV from the arm-local rendered EEF heatmap at endpoint frames `0,4,...,80` using a deterministic weighted centroid;
- obtain opening from the arm-local opening map at those same endpoints;
- mark a transition present only when both SE(3) endpoints and both image endpoints are valid;
- derive activity in physical units using 1 mm, 0.5 degrees, 0.25 control-grid pixels, and opening delta 0.01;
- reject NaN, Inf, invalid rotations, ambiguous/missing heatmaps for a declared-present arm, and all shape drift.

- [ ] **Step 3: Write failing negative-content and no-phase-ID tests**

Cover `correct`, `reverse`, `shift+1`, `shift-1`, and `swap`:

```python
def test_shift_moves_all_four_contents_but_keeps_destination_slots():
    shifted = build_variant_features(correct, "shift+1")
    assert shifted.destination_slot.tolist() == list(range(20))
    assert np.array_equal(shifted.translation[:, 1:], correct.translation[:, :-1])
    assert not hasattr(shifted, "source_phase")

def test_shift_uses_hold_padding_and_swap_preserves_arm_banks():
    ...
```

Expected: fail because variant builders and schema are missing.

- [ ] **Step 4: Implement versioned atomic sidecars and normalization receipts**

The sidecar schema must include:

```text
contract = wan-v9-transition-cache/1
sample
variant
translation / rotation / image / gripper
arm_present / motion_active
source_hdf5_sha256
source_correct_action_sha256
source_counterfactual_sha256
urdf_sha256
clean1785_manifest_sha256
normalization_receipt_sha256
payload_sha256
```

Implement `write_transition_cache_atomic`, `validate_transition_cache`, and sample-scoped quarantine/rebuild. Fit per-modality mean/std only from optimizer-pool correct features; use an explicit minimum standard deviation and persist exact contributing sample IDs and manifest hashes. Negative variants reuse correct statistics and never contribute to them.

- [ ] **Step 5: Implement the cache CLI and leakage gates**

`scripts/cache_wan_v9_transitions.py` must:

- require exact clean-1785, audit20, dev-fast20, HDF5, URDF, correct-cache, and negative-cache paths;
- prove audit20/dev-fast20 disjointness before reading optimizer samples;
- exclude audit20 from the statistics and training sidecars;
- write only below `/data/di/worldarena2_track1_20260815` in non-dry mode;
- support deterministic `--dry-run` without creating output;
- validate complete five-variant coverage per optimizer sample and correct-only coverage for audit20.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_transition.py -q
python -m py_compile src/worldarena_baseline/wan_v9_transition.py scripts/cache_wan_v9_transitions.py
git diff --check -- src/worldarena_baseline/wan_v9_transition.py scripts/cache_wan_v9_transitions.py tests/test_wan_v9_transition.py
git add src/worldarena_baseline/wan_v9_transition.py scripts/cache_wan_v9_transitions.py tests/test_wan_v9_transition.py
git commit -m "feat: add v9 transition feature cache"
```

---

## Task 2: Four-Modality Shared Action Tokenizer

**Files:**

- Create: `src/worldarena_baseline/wan_v9_tokens.py`
- Create: `tests/test_wan_v9_tokens.py`

- [ ] **Step 1: Write failing token-count, sharing, and masking tests**

```python
from worldarena_baseline.wan_v9_tokens import V9ActionTokenizer

def test_exactly_four_tokens_per_arm_interval():
    tokens, present = tokenizer(features)
    assert tokens.shape == (batch, 20, 2, 4, 256)
    assert present.shape == (batch, 20, 2)

def test_tokenizer_has_no_phase_or_arm_embedding():
    names = dict(tokenizer.named_parameters())
    assert not any("phase" in name or "arm_embedding" in name for name in names)

def test_absent_arm_tokens_are_bitwise_zero():
    ...
```

Run and observe RED:

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_tokens.py -q
```

- [ ] **Step 2: Implement the shared four-token tokenizer**

Implement four modality-specific MLPs:

```python
translation_mlp: 4 -> hidden -> 256
rotation_mlp:    4 -> hidden -> 256
image_mlp:       6 -> hidden -> 256
gripper_mlp:     3 -> hidden -> 256
type_embedding:  (4,256)
```

Use Linear-SiLU-Linear. Weights are shared for both arms and all blocks. Validate the normalization receipt hash and normalize each raw modality before the MLP. Apply presence masking after adding the modality embedding so absent tokens remain exact zero.

- [ ] **Step 3: Add strict gradient and serialization tests**

Prove:

- all four MLPs receive finite gradients when their modality is present;
- type embeddings receive gradients;
- absent-arm examples do not contribute gradients from the absent bank;
- state dict contains no fifth token, phase ID, dense raster encoder, or per-block tokenizer copy;
- BF16 input produces stable output while normalization/reduction is FP32.

- [ ] **Step 4: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_tokens.py -q
python -m py_compile src/worldarena_baseline/wan_v9_tokens.py
git diff --check -- src/worldarena_baseline/wan_v9_tokens.py tests/test_wan_v9_tokens.py
git add src/worldarena_baseline/wan_v9_tokens.py tests/test_wan_v9_tokens.py
git commit -m "feat: add v9 four modality action tokenizer"
```

---

## Task 3: Strict Phase-Locked Arm-Grouped Cross-Attention

**Files:**

- Create: `src/worldarena_baseline/wan_v9_attention.py`
- Create: `tests/test_wan_v9_attention.py`

- [ ] **Step 1: Write failing structural indexing tests**

Use tiny dimensions but the production semantics:

```python
def test_latent_zero_and_absent_arm_are_exact_zero(): ...
def test_time_t_reads_only_transition_t_minus_one(): ...
def test_left_heads_cannot_read_right_bank(): ...
def test_overlapping_spatial_support_is_allowed(): ...
def test_every_head_can_select_among_four_modality_tokens(): ...
```

Use distinctive sentinel values in every `(transition, arm, modality)` cell. Change illegal cells by a large amount and assert bitwise-identical output; change the one legal bank and assert output changes.

Run and observe RED:

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_attention.py -q
```

- [ ] **Step 2: Implement local four-key attention without a global mask**

Implement:

```python
class PhaseLockedActionCrossAttention(nn.Module):
    def __init__(self, visual_width=3072, action_width=256,
                 inner_width=768, num_heads=8): ...

    def forward(self, visual, *, grid_sizes, action_tokens,
                arm_present, seq_lens) -> Tensor:
        # reshape visual to B,T,S,H,D
        # output[:,0] = exact zero
        # t>0 selects action_tokens[:,t-1]
        # H0:4 selects left bank; H4:8 selects right bank
        # local attention is only over four modality keys
```

Do not materialize `(video_tokens x all_action_tokens)` scores or a broad mask. Validate fixed 21 latent times for formal shape while permitting small test fixtures through an explicit test-only constructor argument.

- [ ] **Step 3: Add zero-gate and real Q/K selection tests**

Prove:

- channel gate is FP32 and zero initialized;
- wrapper output is exact zero at initialization;
- after a nonzero gate, query and key gradients are finite and nonzero;
- attention probabilities sum to one across exactly four tokens, not one token;
- output gate is channel-wise width 3072, not a scalar;
- invalid grid/sequence lengths fail closed.

- [ ] **Step 4: Add analytic/reference parity tests**

Build a slow explicit reference loop over batch/time/space/head and compare output and gradients to the structured implementation in FP64. Include both arms present, one arm absent, and unequal per-sample sequence lengths.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_attention.py -q
python -m py_compile src/worldarena_baseline/wan_v9_attention.py
git diff --check -- src/worldarena_baseline/wan_v9_attention.py tests/test_wan_v9_attention.py
git add src/worldarena_baseline/wan_v9_attention.py tests/test_wan_v9_attention.py
git commit -m "feat: add v9 strict phase locked attention"
```

---

## Task 4: Wan Blocks 8-13 Integration and Condition Lifecycle

**Files:**

- Create: `src/worldarena_baseline/wan_v9_model.py`
- Create: `tests/test_wan_v9_model.py`

- [ ] **Step 1: Write failing install, whitelist, and initialization tests**

Cover:

```python
V9_BLOCKS == (8, 9, 10, 11, 12, 13)
install rejects any other selection
native q/k/v/o trainable only in V9_BLOCKS
shared tokenizer appears once
each selected block owns one independent cross q/k/v/o and gate
frozen raster parent remains frozen and support-gated
v8 SE3 geometry modules are absent
zero-init v9 output equals clean parent bitwise
```

- [ ] **Step 2: Implement the integrated attention wrapper**

Create `NativePlusPhaseLockedAttention`:

```python
native = original_wan_self_attention(x, seq_lens, grid_sizes, freqs)
cross = local_cross_attention(
    x, grid_sizes=grid_sizes, action_tokens=bound.tokens,
    arm_present=bound.arm_present, seq_lens=seq_lens,
)
return native + cross
```

Native Q/K/V/O stay the original Wan modules. Cross Q/K/V/O are independent. The original output location, modulation, residual, and FFN contracts remain unchanged.

- [ ] **Step 3: Implement transactional condition binding**

Create a typed `V9ActionCondition` and a top-level `ParentPlusPhaseLockedActionWan`. Condition binding must:

- preflight all six wrappers before mutating any wrapper;
- bind one opaque lease token to the exact action-token bank for a forward graph;
- preserve the bank through activation-checkpoint recomputation;
- reject a second forward while an outstanding checkpoint graph owns the lease;
- release only the owning lease after replay or an exception;
- restore all original attention modules and `requires_grad` flags if installation fails partway.

- [ ] **Step 4: Add lifecycle regressions**

Tests must reproduce the prior v7 class of bugs:

```text
checkpointed forward -> rejected second forward -> backward first succeeds
checkpointed replay early-stop -> lease released
attention exception -> lease released
non-checkpoint sequential forwards -> both succeed
partial installation failure -> exact original modules and flags restored
```

- [ ] **Step 5: Implement exact trainable-family inventory**

`v9_trainable_parameter_names(model)` must return and verify exactly:

- native q/k/v/o in blocks 8-13;
- shared tokenizer MLPs and four type embeddings;
- six cross-attention q/k/v/o families;
- six FP32 channel gates.

Any missing, extra, duplicated, wrong-shape, or wrong-dtype trainable tensor raises before CUDA training.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_model.py tests/test_wan_v9_attention.py tests/test_wan_v9_tokens.py -q
python -m py_compile src/worldarena_baseline/wan_v9_model.py
git diff --check -- src/worldarena_baseline/wan_v9_model.py tests/test_wan_v9_model.py
git add src/worldarena_baseline/wan_v9_model.py tests/test_wan_v9_model.py
git commit -m "feat: integrate v9 phase locked attention into Wan"
```

---

## Task 5: Interval-Local Counterfactual Objective and Sequential VJP

**Files:**

- Create: `src/worldarena_baseline/wan_v9_objective.py`
- Create: `tests/test_wan_v9_objective.py`

- [ ] **Step 1: Write failing interval energy tests**

Implement tests for the intended API:

```python
energy = interval_robot_fm_energy(
    prediction, target,
    correct_support=correct_support,
    loss_weight=correct_loss_weight,
    valid_mask=valid_mask,
    motion_active=correct_motion_active,
)
assert energy.shape == (batch, 20)
```

Prove transition `i` reads only latent `i+1`; latent0 never enters ranking; quiet intervals contribute zero weight; wrong conditions cannot alter mask, activity, or denominator; an all-quiet sample is FM-only and reports `cf_eligible=False`.

- [ ] **Step 2: Implement the exact negative schedule and interval ranking**

```python
NEGATIVE_CYCLE = (
    "shift+1", "reverse", "shift-1", "swap", "shift+1",
    "shift-1", "shift+1", "reverse", "shift-1", "swap",
)

def negative_family_for_step(step: int) -> str:
    return NEGATIVE_CYCLE[step % 10]
```

Use `softplus((E_correct-E_wrong)/0.1)` and a normalized active-interval mean. Add tests for exact 60/20/20 family frequency over 50 steps and balanced shift signs.

- [ ] **Step 3: Write a failing direct-vs-sequential gradient oracle**

On a tiny differentiable model compare:

1. direct correct+wrong graphs retained together;
2. sequential correct forward/VJP release, wrong forward/VJP release.

Assert FP64 loss and every parameter gradient match within a stated tolerance. Record peak live-graph count in the test harness and assert sequential never owns two model graphs.

- [ ] **Step 4: Implement sequential analytic VJP and calibration**

Expose:

```python
sequential_pairwise_vjp(...)
calibrate_lambda_cf(..., target_ratio=0.5)
```

Calibration uses only native Q/K/V/O and six gates. It excludes zero-gated cross projections/tokenizer at step0, rejects zero/nonfinite norms, emits a provenance receipt, and freezes lambda for resume. The production helper must never call `retain_graph=True` on the 5B graphs.

- [ ] **Step 5: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_objective.py -q
python -m py_compile src/worldarena_baseline/wan_v9_objective.py
git diff --check -- src/worldarena_baseline/wan_v9_objective.py tests/test_wan_v9_objective.py
git add src/worldarena_baseline/wan_v9_objective.py tests/test_wan_v9_objective.py
git commit -m "feat: add v9 interval counterfactual objective"
```

---

## Task 6: Optimizer, Checkpoint, Audit, and Phase Gates

**Files:**

- Create: `src/worldarena_baseline/wan_v9_training.py`
- Create: `src/worldarena_baseline/wan_v9_audit.py`
- Create: `tests/test_wan_v9_training.py`
- Create: `tests/test_wan_v9_audit.py`

- [ ] **Step 1: Write failing optimizer and schedule tests**

Require three exact AdamW groups:

```text
native_qkvo:      lr=1e-6, wd=.01
action_cross:     lr=1e-4, wd=.01
channel_gates:    lr=5e-5, wd=0
```

Test 25-step warmup, cosine decay to exactly 20% at step500, global clip norm 1.0, no duplicate parameters, exact parameter names in serialized groups, and optimizer state for every trainable tensor after a step.

- [ ] **Step 2: Implement optimizer and scheduler contracts**

Expose `build_v9_optimizer`, `v9_lr_multiplier`, `set_v9_learning_rates`, and `validate_v9_optimizer_state`. Scheduler state and calibrated lambda must be resumed exactly; no warmup reset at the Phase-M/Phase-T boundary.

- [ ] **Step 3: Write failing gate tests**

Cover exact boundary cases for:

- step25 health and post-step1 gradient history;
- step100 shift/reverse/swap 12/20 threshold, shift hard stop at 6/20, and borderline trend rule;
- step250 four-family 14/20, positive margins, position/velocity >5%, routing >=90%, FM <=2%, and enabled-vs-zero gate comparison;
- step500 retention and quality guardrails;
- Phase T rejection without a passing step250 receipt.

- [ ] **Step 4: Implement audit aggregation and decisions**

Audit output keeps per-episode correct/wrong energies, paired win, margin, eligibility, task, arm-presence mix, position/velocity probes, routing retention, FM regression, and gate-zero ablation. Missing/ineligible samples are explicit; they are never dropped from coverage counts.

- [ ] **Step 5: Implement checkpoint and replay lineage**

Checkpoint schema binds:

```text
parent SHA, source closure SHA, clean1785/audit20/devfast20 hashes,
transition stats and cache receipts, replay SHA, topology/GPU mapping,
step, phase, negative-cycle index, lambda calibration,
model whitelist, optimizer/scheduler state, gradient history, gate receipts
```

Approved checkpoints: `10,25,50,100,150,200,250,300,400,500`. Resume rejects any lineage/config/state drift. A step above 250 requires a passing step250 receipt in both the checkpoint and current validation input.

- [ ] **Step 6: Run focused tests and commit**

```bash
PYTHONPATH=src uv run pytest tests/test_wan_v9_training.py tests/test_wan_v9_audit.py -q
python -m py_compile src/worldarena_baseline/wan_v9_training.py src/worldarena_baseline/wan_v9_audit.py
git diff --check -- src/worldarena_baseline/wan_v9_training.py src/worldarena_baseline/wan_v9_audit.py tests/test_wan_v9_training.py tests/test_wan_v9_audit.py
git add src/worldarena_baseline/wan_v9_training.py src/worldarena_baseline/wan_v9_audit.py tests/test_wan_v9_training.py tests/test_wan_v9_audit.py
git commit -m "feat: add v9 training and audit contracts"
```

---

## Task 7: Production Trainer, GPU6 Launcher, and Source Closure

**Files:**

- Create: `src/worldarena_baseline/wan_v9_sync_closure.py`
- Create: `scripts/validate_wan_v9_sync_closure.py`
- Create: `scripts/train_wan_v9_phase_locked.py`
- Create: `scripts/run_wan_v9_phase_locked_gpu6.sh`
- Create: `tests/test_wan_v9_scripts.py`
- Create: `tests/test_wan_v9_runtime.py`

- [ ] **Step 1: Write failing CLI and fail-closed path tests**

Tests must prove:

- dry-run starts no process and creates no artifact;
- non-dry outputs outside the exact `/data/di/worldarena2_track1_20260815` root are rejected;
- launcher sets only `CUDA_VISIBLE_DEVICES=6`, `nproc_per_node=1`, and world size one;
- GPU6 ownership is rechecked before preflight, smoke, train, and audit;
- another compute PID, >1 GiB memory, or nonzero utilization blocks launch;
- no T5/VAE import or model load exists in the trainer hot path;
- stale source/data/cache/preflight/checkpoint receipts are rejected before CUDA initialization.

- [ ] **Step 2: Implement recursive source closure validation**

Use a committed exact entrypoint set and recursively resolve every local static import from the cache CLI, trainer, launcher, model, objective, and audit. Reject missing, symlinked, untracked, staged, modified, ambiguous, and dynamic local dependencies. Retain only the already-reviewed exact legacy `skeleton.py` content-hash exception if it is still required and byte-identical.

- [ ] **Step 3: Implement trainer phases**

The trainer exposes separate guarded commands:

```text
preflight
smoke
train --stop-step 25|100|250|500
audit --step 25|100|250|500
```

Preflight validates all lineage and whitelist contracts before setting CUDA. Smoke runs three full production iterations after resetting CUDA peaks and emits per-iteration allocated/reserved memory, step time, gradient family history, residual norms, and graph-release evidence. Train uses the fixed replay, exact negative cycle, sequential VJP, and atomic checkpoints. Audit never mutates model/optimizer state.

- [ ] **Step 4: Implement the fail-closed GPU6 launcher**

The launcher sequence is:

```text
validate source closure
validate/build transition cache on CPU
preflight
GPU6 ownership check
three-iteration smoke
train to 25 -> audit health
train to 100 -> audit decision
if continue: train to 250 -> hard audit
if pass: train Phase T to 500 -> final audit
```

It must stop on any failed receipt and never silently continue. Every invocation records command argv, environment allowlist, PID identity, GPU UUID, source closure, and artifact hashes below a dedicated v9 run root.

- [ ] **Step 5: Add fake-backbone and minimal distributed runtime tests**

Run the trainer with a tiny hook-compatible Wan fixture through preflight, three-step smoke, checkpoint, resume, audit, and Phase-T rejection. Assert no frozen gradient, all families gain gradients by iteration three, leases are released, and sequential CF never retains both graphs.

- [ ] **Step 6: Run all local v9 tests and commit**

```bash
PYTHONPATH=src uv run pytest \
  tests/test_wan_v9_transition.py tests/test_wan_v9_tokens.py \
  tests/test_wan_v9_attention.py tests/test_wan_v9_model.py \
  tests/test_wan_v9_objective.py tests/test_wan_v9_training.py \
  tests/test_wan_v9_audit.py tests/test_wan_v9_scripts.py \
  tests/test_wan_v9_runtime.py -q
python -m py_compile src/worldarena_baseline/wan_v9_*.py scripts/*wan_v9*.py
bash -n scripts/run_wan_v9_phase_locked_gpu6.sh
git diff --check -- src/worldarena_baseline/wan_v9_*.py scripts/*wan_v9* tests/test_wan_v9_*.py
git add src/worldarena_baseline/wan_v9_sync_closure.py scripts/validate_wan_v9_sync_closure.py scripts/train_wan_v9_phase_locked.py scripts/run_wan_v9_phase_locked_gpu6.sh tests/test_wan_v9_scripts.py tests/test_wan_v9_runtime.py
git commit -m "feat: add guarded v9 production runner"
```

---

## Task 8: Remote Verification, Bounded Training, and Detailed Report

**Files:**

- Create: `reports/2026-08-18-v9-phase-locked-action-cross-attention.md`

- [ ] **Step 1: Sync only the reviewed v9 closure**

Generate the local closure receipt, transfer exactly those files to `/home/huazhi/nlh/baseline`, then generate the remote closure receipt and require digest equality. Do not rsync the full dirty worktree and do not copy any artifact from other users' directories.

- [ ] **Step 2: Run the remote formal zero-skip suite**

```bash
cd /home/huazhi/nlh/baseline
PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest \
  tests/test_wan_v9_transition.py tests/test_wan_v9_tokens.py \
  tests/test_wan_v9_attention.py tests/test_wan_v9_model.py \
  tests/test_wan_v9_objective.py tests/test_wan_v9_training.py \
  tests/test_wan_v9_audit.py tests/test_wan_v9_scripts.py \
  tests/test_wan_v9_runtime.py -q
```

Require zero failures and zero Torch skips. Save stdout/stderr and environment hashes below the v9 report root.

- [ ] **Step 3: Build and validate compact caches on CPU**

Run normalization, correct/audit sidecars, and four optimizer negative variants. Validate exact expected sample counts, zero leakage, five-variant completeness, source hashes, disk reserve, and sample-scoped rebuild behavior before GPU initialization.

- [ ] **Step 4: Run GPU6 production smoke**

Recheck GPU6 ownership immediately before launch. Require three complete correct/wrong forward, sequential backward, optimizer-step, zero-grad iterations with every allocated and reserved peak below 22 GiB. Save per-family gradient and residual telemetry. If smoke fails, stop and diagnose; do not weaken shape, frame count, resolution, or architecture.

- [ ] **Step 5: Run bounded Phase M gates**

Run to step25 and audit health. If healthy, resume to step100 and run the fixed audit20. Apply the spec decision exactly. Continue to step250 only when the step100 receipt allows it. Do not manually reinterpret a failed shift hard stop.

- [ ] **Step 6: Run conditional Phase T only after the hard gate**

If and only if step250 passes, calibrate v6 position/velocity weights at the boundary and resume to step500 without optimizer reset. Otherwise terminate v9 cleanly and release GPU6.

- [ ] **Step 7: Produce the detailed experiment report**

The report must contain:

- exact architecture and parameter inventory;
- parent/source/data/cache/replay/checkpoint hashes;
- optimizer groups, calibrated weights, scheduler, topology, and GPU UUID;
- smoke memory and timing per iteration;
- training curves for FM, ranking, four negative margins, gradients, gates, and residuals;
- audit20 per-episode table and aggregate wins/margins/coverage;
- position/velocity/routing/FM comparisons to parent;
- all pass/fail decisions with exact reasons;
- whether Phase T/RGB/fast20 ran and why;
- artifact paths and next recommendation constrained by the decision tree.

- [ ] **Step 8: Verify final artifacts**

```bash
git diff --check
test -s reports/2026-08-18-v9-phase-locked-action-cross-attention.md
```

Read back every checkpoint and receipt with the production validators. Confirm GPU6 is released after terminal success or failure. Do not claim leaderboard improvement unless matched RGB evaluation actually ran.

