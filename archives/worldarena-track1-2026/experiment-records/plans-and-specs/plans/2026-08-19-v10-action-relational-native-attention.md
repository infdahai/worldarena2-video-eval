# Wan v10 Action-Relational Native Attention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the bounded v10 experiment that injects factorized, arm-specific SE(3) relations into Wan native self-attention, using the complete leakage-free full-action clean replay and direct temporal/trajectory supervision.

**Architecture:** Blocks 6-17 keep Wan's native attention path while LEFT/RIGHT heads receive a rank-16 factorized relation through augmented Q/K and zero-padded V; GLOBAL heads remain native. The run starts from the frozen clean-gated-step10 raster parent, trains only native Q/K/V/O plus relation encoders/gates and two training-only EEF heads, and advances through step50, step150, step300, and step500 gates with a staged semantics/timing/trajectory curriculum.

**Tech Stack:** Python 3.11, PyTorch, Wan2.2 TI2V-5B, fused scaled-dot-product attention, NumPy, HDF5, pytest, Bash, single RTX 4090 GPU6.

**Spec:** `docs/superpowers/specs/2026-08-19-v10-action-relational-native-attention-design.md`

## Global Constraints

- Persistent artifacts may be written only below `/data/di/worldarena2_track1_20260815`.
- Remote source is `/home/huazhi/nlh/baseline`; Wan source is read-only at `/home/huazhi/nlh/Wan2.2`.
- Initial topology is exactly physical GPU6 with `CUDA_VISIBLE_DEVICES=6`, `WORLD_SIZE=1`, and local rank 0.
- The production contract is 81 RGB frames at 480x640 and 21 latent frames; no reduced-shape smoke may satisfy the production gate.
- T5 and VAE are offline cache producers only and must never be constructed in the training hot path.
- Parent is the immutable clean-gated-step10 checkpoint and remains frozen.
- Trainable native attention is exactly Q/K/V/O in blocks 6-17; FFN, normalization, text cross-attention, and every other Wan parameter stay frozen.
- Relation rank is 16; heads 0-7 are LEFT, 8-15 RIGHT, and 16-23 GLOBAL.
- The implementation must use fused attention with augmented head width 144 and native scale `1/sqrt(128)`; explicit quadratic pairwise action bias is forbidden.
- Relation gates are signed FP32 scalars, exactly zero initialized; GLOBAL relation factors and absent-arm relation output are exactly zero.
- Counterfactuals change only relational action state; the frozen parent raster/support always uses the correct action.
- Semantics is steps 1-150, timing is 151-300, and trajectory is 301-500; position and velocity are forbidden before a passing gated step300 checkpoint.
- Formal base learning rates are `1e-6` native QKVO, `1e-4` relation encoders, `5e-5` gates, and `1e-4` hidden heads; warmup is 50 steps and cosine reaches 20% at step500.
- Audit20 must be 20/20 finite and disjoint from optimizer and dev-fast20 before CUDA initialization.
- Production smoke is three complete correct+wrong optimizer iterations and every rank must remain below 22 GiB allocated and reserved.
- No Depth, V-JEPA training loss, DMD, GAN, PRoPE, coordination branch, gripper bias, or extra token side-attention is introduced.
- Existing unrelated dirty files are preserved and never staged.

---

## File Map

- `src/worldarena_baseline/wan_v10_attention.py`: factorized relation state encoder and fused augmented native attention wrapper.
- `src/worldarena_baseline/wan_v10_model.py`: install/rollback lifecycle, frozen parent wrapper, deterministic initialization, trainable whitelist, and gate-zero ablation.
- `src/worldarena_baseline/wan_v10_data.py`: full-action clean/audit20 selection, leakage receipt, relation cache schema, and validators.
- `src/worldarena_baseline/wan_v10_objective.py`: robot-region counterfactual ranking, phase ranking, hidden EEF, position, and velocity losses plus calibration.
- `src/worldarena_baseline/wan_v10_probe.py`: finite-only/failure-aware aggregation and gate-zero comparisons.
- `src/worldarena_baseline/wan_v10_training.py`: optimizer, schedule, checkpoint, resume, replay, lineage, and phase gates.
- `src/worldarena_baseline/wan_v10_sync_closure.py`: recursive source closure and exact source receipt.
- `scripts/probe_wan_v10_augmented_attention.py`: actual Wan fused-kernel feasibility probe.
- `scripts/build_wan_v10_data.py`: atomic full-action clean/audit20 manifests and zero-leakage receipt.
- `scripts/cache_wan_v10_inputs.py`: resumable offline cache production and completeness receipt.
- `scripts/build_wan_v10_replay.py`: deterministic sample/noise/timestep/dropout/counterfactual replay.
- `scripts/train_wan_v10_relational.py`: preflight, smoke, train, audit, and bounded resume entry point.
- `scripts/run_wan_v10_gpu6.sh`: fail-closed phase launcher using physical GPU6 only.
- `scripts/validate_wan_v10_sync_closure.py`: source closure CLI.
- `tests/test_wan_v10_attention.py`, `tests/test_wan_v10_model.py`, `tests/test_wan_v10_data.py`, `tests/test_wan_v10_objective.py`, `tests/test_wan_v10_probe.py`, `tests/test_wan_v10_training.py`, `tests/test_wan_v10_scripts.py`: focused contracts.

---

### Task 1: Factorized Fused-Attention Feasibility Gate

**Files:**
- Create: `src/worldarena_baseline/wan_v10_attention.py`
- Create: `scripts/probe_wan_v10_augmented_attention.py`
- Create: `tests/test_wan_v10_attention.py`
- Create: `tests/test_wan_v10_attention_static.py`

**Interfaces:**
- Consumes: Wan self-attention modules exposing `q`, `k`, `v`, `o`, Q/K normalization, and the production fused attention callable.
- Produces: `RelationCondition`, `ActionRelationEncoder`, `FactorizedRelationalSelfAttention`, `assert_augmented_attention_supported(...)`, and CLI receipt contract `wan-v10-attention-feasibility/1`.

- [ ] **Step 1: Write RED tests for exact shape, routing, and zero-gate equivalence**

```python
def test_relation_attention_routes_arm_heads_and_zero_gate_is_native():
    base, attention_fn = make_fake_wan_attention()
    wrapper = FactorizedRelationalSelfAttention(base, attention_fn=attention_fn)
    condition = make_condition(left_present=True, right_present=False)
    native = base_reference(base, condition.visual_tokens)
    actual = wrapper(condition.visual_tokens, condition=condition)
    torch.testing.assert_close(actual, native, rtol=0, atol=1e-6)
    assert wrapper.last_augmented_shape[-1] == 144
    assert torch.count_nonzero(wrapper.last_right_relation) == 0
    assert torch.count_nonzero(wrapper.last_global_relation) == 0
```

Also assert signed nonzero gates change LEFT only, support overlap is allowed, shifted content retains destination time, and no intermediate tensor has two dimensions both equal to visual-token count.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_attention.py tests/test_wan_v10_attention_static.py`

Expected: collection fails because `worldarena_baseline.wan_v10_attention` does not exist.

- [ ] **Step 3: Implement compact relation state and augmented fused attention**

```python
@dataclass(frozen=True)
class RelationCondition:
    anchored_se3: Tensor       # [B, 21, 2, 6]
    velocity: Tensor           # [B, 21, 2, 6]
    uv: Tensor                 # [B, 21, 2, 2]
    gripper: Tensor            # [B, 21, 2, 2]
    arm_present: Tensor        # [B, 21, 2]
    motion_active: Tensor      # [B, 21, 2]
    support: Tensor            # [B, 21, 2, 30, 40]
    destination_time: Tensor   # [B, 21]

class FactorizedRelationalSelfAttention(nn.Module):
    relation_rank = 16
    num_heads = 24
    head_dim = 128

    def forward(self, x, seq_lens, grid_sizes, freqs, *, relation_condition):
        q, k, v = self._native_qkv(x, grid_sizes, freqs)
        phi, psi = self.relation_encoder(relation_condition, grid_sizes)
        q_aug = torch.cat((q, self._gated_phi(phi)), dim=-1)
        k_aug = torch.cat((k, psi), dim=-1)
        v_aug = F.pad(v, (0, self.relation_rank))
        attended = self.attention_fn(q_aug, k_aug, v_aug, seq_lens, scale=128 ** -0.5)
        return self.base.o(attended[..., :128].flatten(2))
```

Validate all dtypes/shapes before allocation, construct GLOBAL/absent-arm factors with `zeros_like`, keep gates FP32, and reject attention callables that do not accept an explicit scale.

- [ ] **Step 4: Implement the real fused-kernel feasibility probe**

The CLI loads only the production Wan attention callable and synthetic production-size Q/K/V, executes width-128 and width-144 fused calls with explicit scale, and writes an atomic JSON receipt containing kernel identity, dtype, shapes, output finiteness, peak allocated/reserved, and `quadratic_fallback=false`. It exits nonzero if width144 is unsupported or memory indicates an unfused `25200 x 25200` allocation.

- [ ] **Step 5: Run GREEN tests and static quadratic-allocation scan**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_attention.py tests/test_wan_v10_attention_static.py`

Expected: PASS; static test rejects `einsum`/`matmul` constructions producing `[N,N]`, an `attn_bias` parameter, or any fallback call outside the supplied fused callable.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/worldarena_baseline/wan_v10_attention.py scripts/probe_wan_v10_augmented_attention.py tests/test_wan_v10_attention.py tests/test_wan_v10_attention_static.py
git commit -m "feat: add v10 fused relational attention"
```

### Task 2: Deterministic Model Integration and Trainable Whitelist

**Files:**
- Create: `src/worldarena_baseline/wan_v10_model.py`
- Create: `tests/test_wan_v10_model.py`
- Create: `tests/test_wan_v10_model_static.py`

**Interfaces:**
- Consumes: `FactorizedRelationalSelfAttention`, v8/v9 frozen-parent hook lifecycle, production backbone blocks.
- Produces: `V10_BLOCKS`, `install_v10_relational_band(...)`, `ParentPlusRelationalWan`, `v10_trainable_parameter_names(...)`, `relation_gates_enabled(...)`, and `v10_initialized_state_sha256(...)`.

- [ ] **Step 1: Write RED lifecycle and determinism tests**

```python
def test_installation_is_exact_deterministic_and_transactional():
    first = install_fresh(seed=20260819)
    second = install_fresh(seed=20260819)
    assert first.state_sha256 == second.state_sha256
    assert tuple(first.wrappers) == tuple(range(6, 18))
    assert v10_trainable_parameter_names(first.model) == expected_inventory(first.model)
```

Add tests for rollback after a late block failure, frozen parent/backbone exclusions, sequential/checkpointed forward cleanup, exact gate-zero equality, and rejection of aliases or an extra trainable parameter.

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_model.py tests/test_wan_v10_model_static.py`

Expected: FAIL because model integration is absent.

- [ ] **Step 3: Implement installation, hidden heads, and deterministic state hash**

Set Python/NumPy/Torch CPU/CUDA seeds before constructing any v10 module. Wrap exactly blocks 6-17, install training-only `EEFHeatmapHead` at blocks 11 and 17, preserve original `requires_grad` flags for rollback, and re-enable only these four families:

```python
ALLOWED_FAMILIES = (
    "native_qkvo",
    "relation_encoders",
    "relation_gates",
    "hidden_eef_heads",
)
```

The state hash serializes sorted trainable names, dtypes, shapes, and contiguous CPU bytes; it must not include storage addresses or enumeration-order-dependent values.

- [ ] **Step 4: Run GREEN tests and compile checks**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_model.py tests/test_wan_v10_model_static.py`

Run: `python -m py_compile src/worldarena_baseline/wan_v10_attention.py src/worldarena_baseline/wan_v10_model.py`

Expected: all PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add src/worldarena_baseline/wan_v10_model.py tests/test_wan_v10_model.py tests/test_wan_v10_model_static.py
git commit -m "feat: integrate v10 native attention band"
```

### Task 3: Leakage-Free Full-Action Clean and Observable Audit20 Contract

**Files:**
- Create: `src/worldarena_baseline/wan_v10_data.py`
- Create: `scripts/build_wan_v10_data.py`
- Create: `scripts/cache_wan_v10_inputs.py`
- Create: `tests/test_wan_v10_data.py`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Consumes: source-pinned active full-action training pool, dev-fast20 receipt, v6 RGB observability labels, v9 relation transition cache validators.
- Produces: `V10Split`, `build_v10_split(...)`, `validate_v10_data_receipt(...)`, `V10CacheEntry`, `validate_v10_cache_entry(...)`, full-action-clean/audit20 JSONL, normalization JSON, and cache receipt.

- [ ] **Step 1: Write RED selection and leakage tests**

```python
def test_split_is_exact_deterministic_balanced_and_zero_leakage():
    split = build_v10_split(source_rows(), dev_rows(), seed=20260819)
    assert len(split.optimizer) == len(source_rows()) - 20 - len(dev_rows())
    assert len(split.audit) == 20
    assert ids(split.optimizer).isdisjoint(ids(split.audit) | ids(dev_rows()))
    assert set(split.source_counts) == {
        "single_dominant", "bimanual_heavy", "mixed", "quiet"
    }
    assert build_v10_split(reversed(source_rows()), dev_rows(), seed=20260819) == split
```

Add rejection tests for official-test access, duplicate identities, an audit row without eight finite arm-time labels, fewer than 60% supervised optimizer rows, source/hash drift, statistics containing audit/dev rows, path escape, and partial cache sidecars.

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_data.py tests/test_scripts.py -k v10`

Expected: FAIL because v10 data contracts and CLIs do not exist.

- [ ] **Step 3: Implement deterministic selection and atomic receipt**

Use normalized identity keys and content hashes, reject `robot_only` and quarantined paths, sort before seeded selection, and remove audit20 and dev-fast20 before retaining every remaining optimizer identity. The replay, not the source manifest, realizes the 40/35/20/5 target mix. The audit selector requires multi-task coverage, 20/20 finite probe observability, and explicit single/bimanual/crossing/sequential-role coverage. Write JSONL and receipts via temporary file, `fsync`, and `os.replace`.

- [ ] **Step 4: Implement cache schema and offline producer**

```python
@dataclass(frozen=True)
class V10CacheEntry:
    latent: Tensor                 # [16, 21, 30, 40]
    text_context: Tensor
    parent_raster: Tensor          # correct action only
    parent_support: Tensor
    relation_correct: RelationCondition
    relation_reverse: RelationCondition
    relation_swap: RelationCondition
    trajectory_xy: Tensor          # [21, 2, 2]
    trajectory_velocity: Tensor    # [21, 2, 2]
    trajectory_valid: Tensor       # [21, 2]
```

The cache producer may load T5/VAE only in its offline phases. The trainer-facing loader must contain no T5/VAE import or construction. Shift targets are derived at load time with destination slot and mask unchanged.

- [ ] **Step 5: Run GREEN tests and a 2-row local fixture**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_data.py tests/test_scripts.py -k v10`

Expected: PASS, including corruption/recovery tests that rebuild only the invalid sample and leave another valid row byte-identical.

- [ ] **Step 6: Commit Task 3**

```bash
git add src/worldarena_baseline/wan_v10_data.py scripts/build_wan_v10_data.py scripts/cache_wan_v10_inputs.py tests/test_wan_v10_data.py tests/test_scripts.py
git commit -m "feat: add v10 full-action data contract"
```

### Task 4: Direct Temporal and Trajectory Objectives

**Files:**
- Create: `src/worldarena_baseline/wan_v10_objective.py`
- Create: `src/worldarena_baseline/wan_v10_probe.py`
- Create: `tests/test_wan_v10_objective.py`
- Create: `tests/test_wan_v10_probe.py`

**Interfaces:**
- Consumes: predicted FM tensors, fixed robot/support weights, hidden heatmaps, frozen trajectory probe, `V10CacheEntry` labels.
- Produces: `robot_region_energy(...)`, `counterfactual_ranking_loss(...)`, `phase_ranking_loss(...)`, `hidden_eef_loss(...)`, `trajectory_losses(...)`, `calibrate_v10_lambdas(...)`, `aggregate_v10_probe(...)`.

- [ ] **Step 1: Write RED mathematical contract tests**

```python
def test_phase_loss_prefers_destination_target_without_boundary_padding():
    result = phase_ranking_loss(pred, target, valid, margin=0.0)
    assert result.valid_pairs == expected_real_pairs
    assert torch.isfinite(result.loss)

def test_probe_reports_invalid_without_inf_or_silent_exclusion():
    report = aggregate_v10_probe([finite_episode(), invalid_episode("missing_right")])
    assert report.total == 2 and report.finite == 1 and report.invalid == 1
    assert math.isfinite(report.finite_mean)
    assert math.isfinite(report.failure_aware_score)
```

Also test correct-vs-wrong sign, robot-region-only energy, correct raster fixed across counterfactual forwards, high-sigma weighting, arm/time validity masks, and zero gradients outside valid regions.

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_objective.py tests/test_wan_v10_probe.py`

Expected: FAIL because objective/probe modules are missing.

- [ ] **Step 3: Implement losses with unreduced finite masks**

```python
schedule = v10_loss_schedule(step)
loss = fm_correct + schedule.cf * lambdas.cf * cf_loss
loss += schedule.hidden * lambdas.hidden * hidden_loss
loss += schedule.phase * lambdas.phase * phase_loss
loss += schedule.position * lambdas.position * position_loss
loss += schedule.velocity * lambdas.velocity * velocity_loss
```

Use `softplus((E_correct-E_wrong)/tau)` for CF; use separate `t-1` and `t+1` valid pairs for phase. Hidden EEF is head-only through step25 and ramps its gradient into native attention through step75. Phase ramps at 151-200. Position/velocity call the frozen validated v6 probe, ramp at 301-350, and are impossible to request before trajectory stage.

- [ ] **Step 4: Implement deterministic FP32 gradient calibration**

Calibrate on the fixed batch/RNG state against the combined native-QKVO FP32 gradient norm. Target ratios are CF 0.25, phase 0.45, hidden 0.20, position 0.30, velocity 0.20. Reject zero/nonfinite source or target norms and lambda outside `[1e-4,100]`; return a canonical receipt with per-family raw norms, curriculum version, and hashes.

- [ ] **Step 5: Run GREEN tests**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_objective.py tests/test_wan_v10_probe.py`

Expected: PASS with no aggregate Inf/NaN even when an episode is invalid.

- [ ] **Step 6: Commit Task 4**

```bash
git add src/worldarena_baseline/wan_v10_objective.py src/worldarena_baseline/wan_v10_probe.py tests/test_wan_v10_objective.py tests/test_wan_v10_probe.py
git commit -m "feat: add v10 temporal trajectory objectives"
```

### Task 5: Optimizer, Replay, Checkpoint, and Runtime Gates

**Files:**
- Create: `src/worldarena_baseline/wan_v10_training.py`
- Create: `tests/test_wan_v10_training.py`
- Create: `scripts/build_wan_v10_replay.py`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Consumes: exact v10 model whitelist, full-action-clean receipt, calibration receipt, deterministic replay rows, audit reports.
- Produces: `build_v10_optimizer(...)`, `set_v10_learning_rates(...)`, `build_v10_checkpoint(...)`, `validate_v10_checkpoint(...)`, `evaluate_v10_gate(...)`, and replay contract `wan-v10-replay/1`.

- [ ] **Step 1: Write RED optimizer and checkpoint tests**

```python
def test_optimizer_has_four_disjoint_exact_groups():
    optimizer = build_v10_optimizer(model)
    assert [g["name"] for g in optimizer.param_groups] == [
        "native_qkvo", "relation_encoders", "relation_gates", "hidden_eef_heads"
    ]
    assert [g["lr"] for g in optimizer.param_groups] == [1e-6, 1e-4, 5e-5, 1e-4]
    assert optimizer.param_groups[2]["weight_decay"] == 0.0

def test_raw_step300_cannot_start_trajectory_phase():
    with pytest.raises(ValueError, match="gated step300"):
        validate_v10_checkpoint(raw_step300, expected_phase="trajectory")
```

Add tests for warmup/cosine values, full AdamW state, parameter aliases, replay tamper, two identical preflights, topology/GPU mapping, cumulative samples/strata, head-only hidden gradients, forbidden position/velocity before step300, and incomplete lineage.

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_training.py tests/test_scripts.py -k v10`

Expected: FAIL because v10 training/replay contracts are absent.

- [ ] **Step 3: Implement optimizer, scheduler, and exact replay**

Replay rows record global step, optimizer identity, sample identity, noise seed, timestep seed, dropout decisions, and deterministic wrong family (`reverse`, `swap`, alternating). The scheduler is linear warmup through step50 and cosine to 0.2 at step500. Gradient clipping is global norm 1.0 after all loss backwards and before optimizer step.

- [ ] **Step 4: Implement checkpoint lineage and gate receipts**

Checkpoint contract `wan-v10-relational-checkpoint/1` records every hash and state required by the spec. `evaluate_v10_gate` implements exact step50/150/300/500 thresholds, including relation-enabled versus gate-zero combined separation. Only `step-000300-gated.pt` may be the trajectory-stage parent.

- [ ] **Step 5: Run GREEN tests and serialized resume continuation**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_training.py tests/test_scripts.py -k v10`

Expected: PASS; a save/load/next-step fixture must match uninterrupted model, optimizer, scheduler, replay index, and loss values.

- [ ] **Step 6: Commit Task 5**

```bash
git add src/worldarena_baseline/wan_v10_training.py scripts/build_wan_v10_replay.py tests/test_wan_v10_training.py tests/test_scripts.py
git commit -m "feat: add v10 training contracts"
```

### Task 6: Guarded Trainer, Launcher, and Recursive Source Closure

**Files:**
- Create: `scripts/train_wan_v10_relational.py`
- Create: `scripts/run_wan_v10_gpu6.sh`
- Create: `src/worldarena_baseline/wan_v10_sync_closure.py`
- Create: `scripts/validate_wan_v10_sync_closure.py`
- Create: `tests/test_wan_v10_scripts.py`
- Create: `tests/test_wan_v10_runtime.py`

**Interfaces:**
- Consumes: Tasks 1-5 contracts and immutable parent/data/cache/replay/probe receipts.
- Produces: modes `preflight`, `smoke`, `train`, `audit`, `rgb8`; recursive closure receipt; phase-safe GPU6 launcher.

- [ ] **Step 1: Write RED launcher-order and hot-path tests**

```python
def test_gpu6_launcher_runs_gates_before_torch_and_never_loads_t5_vae():
    text = Path("scripts/run_wan_v10_gpu6.sh").read_text()
    assert text.index("validate_wan_v10_sync_closure.py") < text.index("train_wan_v10_relational.py")
    assert "CUDA_VISIBLE_DEVICES=6" in text
    assert "T5" not in trainer_ast_imports() and "WanVAE" not in trainer_ast_imports()
```

Add tests for formal-root guards, current closure revalidation before CUDA, GPU6 ownership/memory/utilization checks, raw checkpoint rejection, gated checkpoint selection, audit20 finite gate, no `torchrun --nproc_per_node` greater than 1, and dry-run `starts_processes=false`.

- [ ] **Step 2: Run tests and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_scripts.py tests/test_wan_v10_runtime.py`

Expected: FAIL because trainer/launcher/closure files do not exist.

- [ ] **Step 3: Implement pre-CUDA validation and trainer modes**

The trainer validates source closure, formal paths, parent/data/cache/replay/audit/probe/calibration hashes, topology, and deterministic initial-state equality before calling `torch.cuda.set_device`. Smoke resets peaks after warmup and runs three full correct+wrong/backward/step/zero-grad iterations. Train uses sequential correct/wrong forwards, activation checkpointing, and atomic checkpoints.

- [ ] **Step 4: Implement phase-safe launcher**

```text
feasibility
-> data
-> cache
-> preflight-A
-> preflight-B byte comparison
-> smoke
-> train50 -> audit50
-> train150 -> audit150 -> gate150
-> train300 -> audit300 -> gate300
-> train500 -> audit500 -> gate500
-> rgb8 only after gate500
```

Every non-dry phase takes a filesystem lock, writes PID identity, rechecks physical GPU6 immediately before CUDA, and refuses existing artifacts unless their complete contract validates. A failed gate stops without launching the next phase.

- [ ] **Step 5: Implement recursive exact closure**

Recursively resolve local static imports from the trainer, launcher helpers, cache, attention, model, objective, probe, and training roots. Require the source-controlled manifest to equal the discovered closure exactly, require tracked/index-clean/worktree-clean bytes except already reviewed source-pinned legacy exceptions, and bind actual byte SHA256 values into every run receipt.

- [ ] **Step 6: Run GREEN tests, shell syntax, and dry-run**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v10_scripts.py tests/test_wan_v10_runtime.py`

Run: `bash -n scripts/run_wan_v10_gpu6.sh`

Run: `DRY_RUN=1 bash scripts/run_wan_v10_gpu6.sh all`

Expected: tests PASS; dry-run lists every phase in order, reports physical GPU6 only, and reports `starts_processes=false`.

- [ ] **Step 7: Commit Task 6**

```bash
git add scripts/train_wan_v10_relational.py scripts/run_wan_v10_gpu6.sh src/worldarena_baseline/wan_v10_sync_closure.py scripts/validate_wan_v10_sync_closure.py tests/test_wan_v10_scripts.py tests/test_wan_v10_runtime.py
git commit -m "feat: add guarded v10 gpu6 runner"
```

### Task 7: Remote Verification, Data Preparation, Smoke, and Bounded Training

**Files:**
- Create remotely under formal root: `runs/v10-action-relational-native-attention/receipts/*.json`
- Create remotely under formal root: `cache/v10-action-relational-native-attention/**`
- Create remotely under formal root: `runs/v10-action-relational-native-attention/checkpoints/*.pt`
- Create: `reports/2026-08-19-v10-action-relational-native-attention.md`

**Interfaces:**
- Consumes: committed clean recursive closure from Task 6 and explicit user authorization for scoped sync/execution.
- Produces: remote zero-skip Torch verification, feasibility receipt, full-action-clean/cache receipts, smoke receipt, gated checkpoints/audits, RGB8 results if eligible, and final report.

- [ ] **Step 1: Verify local closure and sync only its exact files**

Run locally:

```bash
PYTHONPATH=src python scripts/validate_wan_v10_sync_closure.py --write-receipt /private/tmp/v10-source-closure.json
rsync -a --files-from=/private/tmp/v10-source-files.txt ./ huazhi@183.147.142.110:/home/huazhi/nlh/baseline/
```

Expected: remote file hashes exactly match the local closure receipt; no unrelated dirty/untracked file is copied.

- [ ] **Step 2: Run remote zero-skip focused tests**

Run with the existing formal Conda/Torch interpreter discovered from the server environment manifest:

```bash
PYTHONPATH=/home/huazhi/nlh/baseline/src:/home/huazhi/nlh/Wan2.2 \
  <formal-python> -m pytest -q \
  tests/test_wan_v10_attention.py tests/test_wan_v10_attention_static.py \
  tests/test_wan_v10_model.py tests/test_wan_v10_model_static.py \
  tests/test_wan_v10_data.py tests/test_wan_v10_objective.py \
  tests/test_wan_v10_probe.py tests/test_wan_v10_training.py \
  tests/test_wan_v10_scripts.py tests/test_wan_v10_runtime.py
```

Expected: all selected tests PASS and `0 skipped`; any skip blocks execution.

- [ ] **Step 3: Run the actual augmented-head feasibility gate on GPU6**

First confirm no foreign compute PID and GPU6 memory/utilization satisfy the ownership contract. Then run `probe_wan_v10_augmented_attention.py` with the formal Wan callable and production token count. Expected: width144 fused path PASS, explicit scale PASS, finite output, no quadratic fallback, receipt below formal root. Failure retires v10 exactly as designed.

- [ ] **Step 4: Build full-action-clean/audit20 and complete offline cache**

Run the data CLI, validate zero leakage, then run resumable cache production. It may use only currently available GPUs without disturbing another process; cache work and training must not overlap on GPU6. Expected for the current pinned source: 2,100 active full episodes, exact 2,060 optimizer rows after disjoint dev20/audit20 removal, exact 20/20 finite audit rows, at least 60% supervised rows, all sidecar hashes valid, and at least 50 GiB disk reserve after completion. Any different count must be explained and content-bound by the receipt rather than silently padded with `robot_only` or quarantined rows.

- [ ] **Step 5: Run two preflights and compare bytes**

Run fresh preflight A and B from the same seed/replay. Expected: identical v10 initialized-state SHA256 and identical calibration receipt bytes. Any difference blocks smoke.

- [ ] **Step 6: Run production GPU6 smoke**

Run exactly three complete iterations at production geometry. Expected: allocated and reserved peaks each below 22 GiB; finite nonzero gradients in all four trainable families; zero-gate native equivalence; no memory growth, OOM, NaN/Inf, fallback, or foreign-GPU access.

- [ ] **Step 7: Run bounded phase gates without skipping**

Run step50 and audit. If it passes, run step150 semantics audit, then step300 timing audit with enabled-versus-zero relation ablation. Only a passing `step-000300-gated.pt` may start trajectory. Run step500 and audit only after that. Decode exactly RGB8 only after the step500 hard gate.

- [ ] **Step 8: Write the detailed experiment report**

The report records exact source/config/data/cache/replay/checkpoint hashes, topology, wall time, memory, samples/strata, LR/calibration, FM curves, all finite/invalid audit counts and reasons, correct wins/margins for reverse/swap/phase±1, position/velocity/routing, gate-zero ablation, RGB8 guardrails, every stopped gate, and the next decision. It must explicitly preserve negative results and must not claim official test performance.

- [ ] **Step 9: Commit the report only after verifying its cited artifacts**

```bash
git add reports/2026-08-19-v10-action-relational-native-attention.md
git commit -m "docs: report v10 relational attention experiment"
```

---

## Final Verification Checklist

- [ ] `git diff --check` passes and unrelated dirty files remain untouched.
- [ ] The focused local suite passes; the remote formal Torch suite passes with zero skips.
- [ ] The actual Wan fused-kernel width144 feasibility receipt passes before full-action-clean cache work.
- [ ] Data receipt proves optimizer/audit20/dev-fast20 pairwise disjointness and official test unavailability.
- [ ] Training hot path imports/constructs neither T5 nor VAE.
- [ ] Two preflights produce byte-identical initialized state and calibration.
- [ ] GPU6 smoke completes three correct+wrong optimizer iterations below 22 GiB allocated and reserved.
- [ ] Only gated checkpoints cross phase boundaries.
- [ ] Probe aggregates are finite-only plus failure-aware and never silently drop invalid episodes.
- [ ] RGB decoding is exactly eight samples and occurs only after step500 passes.
- [ ] Every persistent artifact resolves below `/data/di/worldarena2_track1_20260815`.
