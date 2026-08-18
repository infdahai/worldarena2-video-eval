# Wan-Action v8 Direct Action Band Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run the bounded v8 experiment that trains native Wan Q/K/V/O in blocks 8-13 first for complete-action separation and then, only after a held-out gate, for gripper trajectory alignment.

**Architecture:** A frozen clean-gated-step10 parent injects complete raster action conditioning. Six consecutive Wan self-attention blocks share trainable native Q/K/V/O between the original attention path and a deterministic arm-grouped SE(3) path controlled by six zero-initialized channel gates. Phase M uses weighted FM plus full-action counterfactual ranking through step100; a passing checkpoint continues in the same lineage with the frozen v6 position/velocity probe through step250.

**Tech Stack:** Python 3.11, PyTorch/FSDP, Wan2.2-TI2V-5B, NumPy/HDF5, pytest, Bash, cached bfloat16 latents and text contexts.

**Spec:** `docs/superpowers/specs/2026-08-18-v8-partial-wan-action-finetune-design.md`

## Global Constraints

- Preserve all unrelated dirty files in the shared `feat/oscar-track1-baseline` workspace.
- Persistent artifacts must stay under `/data/di/worldarena2_track1_20260815`.
- Remote source is `/home/huazhi/nlh/baseline`; Wan source is `/home/huazhi/nlh/Wan2.2`.
- Parent SHA256 is `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- Use clean-1785 minus held-out audit20; dev-fast20 and official test are forbidden training inputs.
- Blocks 8-13 native Q/K/V/O and six geometry channel gates are the only trainable tensors.
- Geometry Q/K/V/O LoRA, block scanning, MLP/norm/cross-attention training, VAE/T5 loading, and automatic step500 are forbidden.
- Phase M is steps 0-100; Phase T is permitted only after the step100 gate and ends at step250.
- Production smoke requires three complete correct/wrong optimizer iterations with every rank below 22 GiB allocated and reserved.
- The execution topology is immutable after smoke; GPU processes owned by other users are never stopped or reused.

---

## File Structure

- `src/worldarena_baseline/wan_v8_data.py`: v8 audit selection, replay construction, correct/negative cache schema, and completeness validation.
- `src/worldarena_baseline/wan_v8_counterfactual.py`: physically anchored complete-action reverse, shift, and swap builders plus fixed ranking-mask helpers.
- `src/worldarena_baseline/wan_v8_attention.py`: shared-projection native/SE(3) attention wrapper and gate ablation.
- `src/worldarena_baseline/wan_v8_model.py`: exact blocks 8-13 installation, condition lifecycle, and trainable whitelist.
- `src/worldarena_baseline/wan_v8_objective.py`: Phase M/Phase T losses, sequential VJP, and gradient-ratio calibration.
- `src/worldarena_baseline/wan_v8_training.py`: optimizer, scheduler, checkpoint, resume, phase transition, and gate contracts.
- `src/worldarena_baseline/wan_v8_audit.py`: held-out audit20 metrics, gate-zero ablation, and step100/250 decisions.
- `src/worldarena_baseline/wan_v8_sync_closure.py`: exact recursive v8 runtime-source closure and digest validation.
- `scripts/prepare_wan_v8_data.py`: CPU-only completion of 1,785 SE(3) and complete-action negative caches plus replay/audit receipts.
- `scripts/train_wan_v8_direct_action_band.py`: preflight, smoke, Phase M, Phase T, audit, and checkpoint entrypoint.
- `scripts/run_wan_v8_direct_action_band.sh`: guarded single-GPU-first launcher with fixed-topology FSDP fallback.
- `tests/test_wan_v8_*.py`: focused pure, Torch, trainer, launcher, and lineage tests.

### Task 1: Complete and Freeze the v8 Data Contract

**Files:**
- Create: `src/worldarena_baseline/wan_v8_data.py`
- Create: `scripts/prepare_wan_v8_data.py`
- Create: `tests/test_wan_v8_data.py`
- Create: `src/worldarena_baseline/wan_v8_sync_closure.py`

**Interfaces:**
- Consumes: clean-1785 cached manifest, v6 probe split, dev-fast20 manifest, role strata, HDF5/FK sources, existing correct action/SE(3) sidecars.
- Produces: `select_v8_audit20(...)`, `build_v8_replay(...)`, `validate_v8_cache(...)`, a 20-row held-out audit manifest, a 250-step topology-bound replay, 1,785 valid SE(3) sidecars, and 1,785 complete-negative sidecars.

- [ ] **Step 1: Write failing data-lineage tests**

```python
def test_audit20_comes_from_probe_heldout_and_is_not_trainable():
    receipt = build_v8_data_receipt(rows, probe_split, dev_rows, seed=20260818)
    assert len(receipt.audit_samples) == 20
    heldout = {row["sample"] for row in probe_split["heldout_rows"]}
    assert set(receipt.audit_samples) <= heldout
    assert set(receipt.audit_samples).isdisjoint(receipt.optimizer_samples)
    assert set(receipt.optimizer_samples).isdisjoint(dev_samples)


def test_cache_requires_all_correct_and_four_negative_families(tmp_path):
    with pytest.raises(ValueError, match="SE3 coverage 1000/1785"):
        validate_v8_cache(tmp_path, expected_samples=sample_ids)
```

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `PYTHONPATH=src uv run pytest tests/test_wan_v8_data.py -q`

Expected: collection fails because `worldarena_baseline.wan_v8_data` does not exist.

- [ ] **Step 3: Implement immutable audit and balanced replay construction**

```python
V8_AUDIT_SEED = 20260818
V8_TARGET = {
    "single_dominant": 0.45,
    "bimanual_heavy": 0.30,
    "mixed": 0.15,
    "quiet": 0.10,
}

def select_v8_audit20(*, cached_rows, heldout_samples, dev_samples) -> tuple[str, ...]:
    eligible = sorted(
        set(row["sample"] for row in cached_rows)
        & set(heldout_samples)
        - set(dev_samples),
        key=lambda sample: hashlib.sha256(
            f"{V8_AUDIT_SEED}:{sample}".encode()
        ).digest(),
    )
    selected = task_arm_stratified_take(eligible, count=20)
    if len(selected) != 20:
        raise ValueError("v8 audit20 cannot satisfy held-out coverage")
    return tuple(selected)
```

Use `build_balanced_history` from `wan_balanced_sampling.py` after removing all audit identities. Bind sample, noise, timestep, negative family, shift direction, rank, and optimizer step into every replay record.

- [ ] **Step 4: Implement resumable CPU cache completion**

The preparation script must:

1. validate the 1,785-row cached manifest and parent/source receipts;
2. reuse a sidecar only after full schema/hash validation;
3. generate the missing 785 correct SE(3) sidecars from raw HDF5/FK;
4. render `reverse`, `shift_plus`, `shift_minus`, and anchored `swap` complete-action negatives for all 1,785 samples;
5. publish each sample sidecar atomically and quarantine only an invalid sample pair;
6. emit one canonical cache receipt and never load CUDA.

The negative sidecar schema is:

```python
V8_NEGATIVE_KEYS = {
    f"{family}_{field}"
    for family in ("reverse", "shift_plus", "shift_minus", "swap")
    for field in ("raster", "support", "se3", "arm_present")
}
```

- [ ] **Step 5: Run pure tests and a 2-sample preparation fixture**

Run: `PYTHONPATH=src uv run pytest tests/test_wan_v8_data.py -q`

Expected: all tests pass; a second preparation run performs zero rewrites and produces the same receipt SHA256.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/worldarena_baseline/wan_v8_data.py scripts/prepare_wan_v8_data.py tests/test_wan_v8_data.py src/worldarena_baseline/wan_v8_sync_closure.py
git commit -m "feat: add v8 data and replay contract"
```

### Task 2: Build Physically Complete Counterfactual Actions

**Files:**
- Create: `src/worldarena_baseline/wan_v8_counterfactual.py`
- Create: `tests/test_wan_v8_counterfactual.py`

**Interfaces:**
- Consumes: correct raw per-arm motion, raster channels, opening, support, SE(3), and arm presence.
- Produces: `CompleteAction`, `build_complete_counterfactual(...)`, `negative_for_step(...)`, and `fixed_ranking_energy_mask(...)`.

- [ ] **Step 1: Write failing physical-contract tests**

```python
@pytest.mark.parametrize("family,direction", [
    ("reverse", 0), ("shift", 1), ("shift", -1), ("swap", 0)
])
def test_complete_negative_changes_every_action_family(family, direction):
    wrong = build_complete_counterfactual(correct, family, shift_direction=direction)
    assert not torch.equal(wrong.raster, correct.raster)
    assert not torch.equal(wrong.conditioning_support, correct.conditioning_support)
    assert not torch.equal(wrong.se3, correct.se3)
    assert torch.equal(wrong.ranking_energy_mask, correct.ranking_energy_mask)


def test_swap_preserves_each_arm_anchor():
    swapped = build_complete_counterfactual(correct, "swap", shift_direction=0)
    torch.testing.assert_close(swapped.se3[:, :, 0], correct.se3[:, :, 0])
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src uv run pytest tests/test_wan_v8_counterfactual.py -q`

Expected: missing module/function failures.

- [ ] **Step 3: Implement typed conditions and deterministic negative cycle**

```python
@dataclass(frozen=True)
class CompleteAction:
    raster: Tensor
    conditioning_support: Tensor
    se3: Tensor
    arm_present: Tensor
    ranking_energy_mask: Tensor
    loss_weight: Tensor

def negative_for_step(step: int) -> tuple[str, int]:
    cycle = (("reverse", 0), ("shift", 1), ("swap", 0),
             ("reverse", 0), ("shift", -1), ("swap", 0))
    if step <= 0:
        raise ValueError("optimizer step must be positive")
    return cycle[(step - 1) % len(cycle)]
```

Reject direct absolute-pose swapping. Reverse and swap operate on relative motion and re-anchor to each arm's original frame-0 pose. Shift uses hold padding. Recompute time-derived raster flow instead of reversing flow bytes.

- [ ] **Step 4: Implement the fixed energy mask and energy reduction**

```python
def fixed_ranking_energy_mask(correct_support: Tensor, valid: Tensor) -> Tensor:
    union = correct_support.float().amax(dim=1, keepdim=True)
    mask = union * valid.float()
    if not torch.isfinite(mask).all() or bool((mask.flatten(1).sum(1) <= 0).any()):
        raise ValueError("ranking energy mask has no finite mass")
    return mask
```

Correct and wrong energies must receive the same mask and correct `loss_weight` object.

- [ ] **Step 5: Run tests and commit**

Run: `PYTHONPATH=src uv run pytest tests/test_wan_v8_counterfactual.py -q`

```bash
git add src/worldarena_baseline/wan_v8_counterfactual.py tests/test_wan_v8_counterfactual.py
git commit -m "feat: add complete v8 counterfactual actions"
```

### Task 3: Implement Shared-Projection Direct Action Attention

**Files:**
- Create: `src/worldarena_baseline/wan_v8_attention.py`
- Create: `src/worldarena_baseline/wan_v8_model.py`
- Create: `tests/test_wan_v8_attention.py`
- Create: `tests/test_wan_v8_model.py`

**Interfaces:**
- Consumes: production Wan self-attention modules, existing `ArmGroupedSE3Geometry`, and model-bound SE(3) condition.
- Produces: `DirectActionBandAttention`, `install_v8_action_band(...)`, `ParentPlusV8Wan`, `v8_trainable_parameter_names(...)`, and `geometry_gates_enabled(...)`.

- [ ] **Step 1: Write failing equality, sharing, and whitelist tests**

```python
def test_v8_has_no_duplicate_projection_or_geometry_lora(fake_attention):
    wrapper = DirectActionBandAttention(fake_attention, attention_fn=attention, rope_apply_fn=rope)
    assert wrapper.base.q is fake_attention.q
    assert wrapper.base.k is fake_attention.k
    assert wrapper.base.v is fake_attention.v
    assert wrapper.base.o is fake_attention.o
    assert not any("lora" in name for name, _ in wrapper.named_parameters())


def test_zero_gate_is_parent_equal_and_qkvo_are_trainable(model, inputs):
    parent = model.backbone(*inputs)
    install_v8_action_band(model.backbone, blocks=(8, 9, 10, 11, 12, 13), **runtime)
    actual = model(*inputs, action=condition)
    assert torch.equal(actual, parent)
    assert v8_trainable_parameter_names(model) == expected_qkvo_and_gate_names(model)
```

- [ ] **Step 2: Confirm RED in the remote Torch environment**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_attention.py tests/test_wan_v8_model.py -q`

Expected: missing v8 modules.

- [ ] **Step 3: Implement one shared native/geometry wrapper**

```python
class DirectActionBandAttention(nn.Module):
    def __init__(self, base, *, attention_fn, rope_apply_fn):
        super().__init__()
        self.base = base
        self.geometry = ArmGroupedSE3Geometry(attention_fn=attention_fn, num_heads=24, head_dim=128)
        self.geometry.requires_grad_(False)
        self.channel_gate = nn.Parameter(torch.zeros(3072, dtype=torch.float32))

    def enable_v8_training(self):
        self.base.requires_grad_(False)
        for projection in (self.base.q, self.base.k, self.base.v, self.base.o):
            projection.requires_grad_(True)
        self.channel_gate.requires_grad_(True)
```

Compute raw Q/K/V once. Native and geometry attention use the same tensors and shared O projection. Apply the channel gate after geometry O and add it at the existing pre-modulation residual point.

- [ ] **Step 4: Implement exact blocks 8-13 installation and condition leases**

Reuse the proven token-scoped checkpoint condition lifecycle from `wan_v7_model.py`, but do not broaden v7 APIs. Installation must roll back all modules and original `requires_grad` flags on any late failure.

- [ ] **Step 5: Add gate-zero audit context and checkpoint replay tests**

```python
with geometry_gates_enabled(model, enabled=False):
    zero_geometry = model(**batch)
learned_geometry = model(**batch)
assert all(wrapper.channel_gate.requires_grad for wrapper in model.geometry_wrappers.values())
```

Prove a rejected second forward cannot clear a first checkpoint graph's condition before backward.

- [ ] **Step 6: Run focused tests and commit**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_attention.py tests/test_wan_v8_model.py -q`

```bash
git add src/worldarena_baseline/wan_v8_attention.py src/worldarena_baseline/wan_v8_model.py tests/test_wan_v8_attention.py tests/test_wan_v8_model.py
git commit -m "feat: add v8 direct action attention band"
```

### Task 4: Implement Phase M and Phase T Objectives

**Files:**
- Create: `src/worldarena_baseline/wan_v8_objective.py`
- Create: `tests/test_wan_v8_objective.py`
- Modify: `src/worldarena_baseline/wan_gripper_trajectory_loss.py`

**Interfaces:**
- Consumes: correct/wrong FM predictions, fixed mask, native Q/K/V/O parameters, frozen v6 probe, target position/velocity, sigma calibration.
- Produces: `V8Phase`, `calibrate_cf_lambda(...)`, `calibrate_trajectory_lambdas(...)`, `phase_m_backward(...)`, and `phase_t_backward(...)`.

- [ ] **Step 1: Write failing direct-versus-sequential gradient tests**

```python
def test_sequential_vjp_matches_direct_pairwise_gradient(tiny_model):
    direct = direct_pairwise_gradients(tiny_model, correct, wrong, tau=0.1)
    sequential = sequential_pairwise_gradients(tiny_model, correct, wrong, tau=0.1)
    for expected, actual in zip(direct, sequential):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_objective.py -q`

- [ ] **Step 3: Implement Q/K/V/O-only calibration**

```python
def calibrated_ratio(*, reference_norm: Tensor, objective_norm: Tensor, ratio: float) -> float:
    values = (reference_norm, objective_norm)
    if any(v.numel() != 1 or not torch.isfinite(v) or v <= 0 for v in values):
        raise ValueError("calibration norms must be positive finite scalars")
    return float((ratio * reference_norm / objective_norm).detach())
```

Phase M freezes `lambda_cf` at ratio `0.5`. Phase T calibrates `lambda_pos` at `0.25` and `lambda_vel` at `0.15` without resetting `lambda_cf`.

Production FSDP calibration must use separate ordinary `backward()` passes and
read the accumulated Q/K/V/O gradients after each pass; it must not call
cross-graph `torch.autograd.grad` on sharded parameters. The direct pairwise
oracle remains limited to the tiny non-checkpoint unit fixture.

- [ ] **Step 4: Reuse the frozen v6 probe without its correction head**

Use `predicted_clean_latent`, `freeze_probe_for_trajectory_loss`, `probe_trajectory_terms`, and `sigma_weights`. Add a generic gradient-norm helper over explicit Q/K/V/O parameters rather than reusing the v6 B-only helper.

- [ ] **Step 5: Implement phase-specific backward contracts**

`phase_m_backward` performs wrong then correct graph-bearing forwards and applies the analytic wrong-energy coefficient. `phase_t_backward` adds position and velocity gradients from the correct forward only. Both return detached scalar telemetry and leave no checkpoint lease alive.

- [ ] **Step 6: Run tests and commit**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_objective.py tests/test_wan_gripper_trajectory_loss.py -q`

```bash
git add src/worldarena_baseline/wan_v8_objective.py src/worldarena_baseline/wan_gripper_trajectory_loss.py tests/test_wan_v8_objective.py
git commit -m "feat: add v8 phased action objectives"
```

### Task 5: Implement Optimizer, Checkpoint, and Phase Contracts

**Files:**
- Create: `src/worldarena_baseline/wan_v8_training.py`
- Create: `tests/test_wan_v8_training.py`

**Interfaces:**
- Consumes: v8 model, data/replay receipt, phase metrics, parent/source/cache hashes.
- Produces: `build_v8_optimizer(...)`, `set_v8_learning_rates(...)`, `build_v8_checkpoint(...)`, `validate_v8_checkpoint(...)`, `step100_gate(...)`, and `step250_gate(...)`.

- [ ] **Step 1: Write failing optimizer and gate tests**

```python
def test_optimizer_has_exact_native_and_gate_groups(model):
    optimizer = build_v8_optimizer(model)
    assert [g["lr"] for g in optimizer.param_groups] == [1e-6, 5e-5]
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.01, 0.0]


def test_failed_step100_cannot_resume_as_phase_t(valid_checkpoint):
    valid_checkpoint["gates"]["step100"]["swap_wins"] = 10
    with pytest.raises(ValueError, match="Phase T requires passing step100"):
        validate_v8_checkpoint(valid_checkpoint, expected_phase="trajectory")
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_training.py -q`

- [ ] **Step 3: Implement exact optimizer and schedule**

Use AdamW, 25-step linear warmup, cosine decay to 20% at step250, and global L2 clip `1.0`. Serialize actual group parameter names and reject empty/missing state, duplicate parameters, unknown parameters, or LR/scheduler drift.

- [ ] **Step 4: Implement immutable checkpoint lineage**

Checkpoint validation must independently re-hash parent, source closure, clean-1785 receipt, negative cache receipt, audit20, replay, probe, sigma calibration, topology, and phase transition evidence. It must reject v7/v7.1 parents and any step above 100 in Phase M or below 100 in Phase T.

- [ ] **Step 5: Implement the exact step100 and step250 decisions**

```python
def step100_gate(m):
    return all((m[n]["wins"] >= 11 and m[n]["mean_margin"] > 0)
               for n in ("reverse", "shift", "swap")) \
        and m["routing_retention"] >= 0.90 \
        and m["fm_regression"] < 0.02 \
        and m["position_regression"] <= 0.02 \
        and m["velocity_regression"] <= 0.02
```

Step250 raises the win thresholds to 14 and requires position/velocity improvement strictly above 5%.

- [ ] **Step 6: Run tests and commit**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_training.py -q`

```bash
git add src/worldarena_baseline/wan_v8_training.py tests/test_wan_v8_training.py
git commit -m "feat: add v8 training and phase contracts"
```

### Task 6: Implement Held-Out Audit and Gate-Zero Attribution

**Files:**
- Create: `src/worldarena_baseline/wan_v8_audit.py`
- Create: `tests/test_wan_v8_audit.py`

**Interfaces:**
- Consumes: fixed audit20, checkpoint, model, frozen probe, correct and all wrong action caches.
- Produces: `run_v8_audit(...)`, per-negative paired rows, gate-enabled/gate-zero reports, and canonical step100/250 decision payloads.

- [ ] **Step 1: Write failing audit tests**

```python
def test_shift_win_uses_harder_of_plus_and_minus(row):
    row["shift_plus_energy"] = 0.9
    row["shift_minus_energy"] = 0.7
    assert hard_shift_energy(row) == 0.7


def test_gate_ablation_uses_identical_inputs_and_checkpoint(model, audit_batch):
    report = run_v8_audit(model, audit_batch)
    assert report["enabled"]["input_sha256"] == report["gate_zero"]["input_sha256"]
    assert report["enabled"]["checkpoint_sha256"] == report["gate_zero"]["checkpoint_sha256"]
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_audit.py -q`

- [ ] **Step 3: Implement failure-aware paired metrics**

For reverse, hard shift, and swap report wins/20, mean energy margin, position separation, velocity separation, and valid counts. Report correct FM, routing retention, geometry gate norms, and parent-relative position/velocity. Never drop invalid rows silently; use an explicit failure-aware value and count.

- [ ] **Step 4: Implement gate-zero attribution**

Run the same checkpoint/input twice under a context manager that only changes gate multiplication. Record whether SE(3) improves or harms each metric without using the ablation to select a different checkpoint.

- [ ] **Step 5: Run tests and commit**

Run: `PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_audit.py -q`

```bash
git add src/worldarena_baseline/wan_v8_audit.py tests/test_wan_v8_audit.py
git commit -m "feat: add v8 heldout mechanism audit"
```

### Task 7: Implement the Guarded Trainer and Launcher

**Files:**
- Create: `scripts/train_wan_v8_direct_action_band.py`
- Create: `scripts/run_wan_v8_direct_action_band.sh`
- Create: `tests/test_wan_v8_trainer.py`
- Create: `tests/test_wan_v8_scripts.py`
- Create: `scripts/validate_wan_v8_sync_closure.py`
- Modify: `src/worldarena_baseline/wan_v8_sync_closure.py`

**Interfaces:**
- Consumes: Tasks 1-6 contracts and production Wan runtime.
- Produces: `preflight`, `smoke`, `phase-m`, `audit100`, `phase-t`, `audit250`, and `rgb-ready` guarded modes.

- [ ] **Step 1: Write failing launcher and no-hot-path tests**

```python
def test_launcher_orders_all_gates_before_training(script_text):
    assert script_text.index("preflight") < script_text.index("smoke")
    assert script_text.index("smoke") < script_text.index("phase-m")
    assert script_text.index("audit100") < script_text.index("phase-t")


def test_training_runtime_never_loads_t5_or_vae(monkeypatch):
    monkeypatch.setattr(runtime, "load_t5", forbidden)
    monkeypatch.setattr(runtime, "load_vae", forbidden)
    trainer = build_trainer(cached_batch)
    trainer.run_one_step()
```

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src uv run pytest tests/test_wan_v8_scripts.py -q`

- [ ] **Step 3: Implement fail-closed preflight**

Before `torch.cuda.set_device` or distributed initialization, validate source closure, parent/model/data/probe hashes, cache coverage, audit/replay isolation, topology, trainable names, phase, and writable output root.

- [ ] **Step 4: Implement production smoke**

Run three correct+wrong `forward -> backward -> optimizer.step -> zero_grad` iterations. Reset peak memory after initialization/warmup. Gather every rank's allocated/reserved peak, step time, required-gradient history, frozen-gradient absence, and optimizer state. Reject any rank at or above 22 GiB.

- [ ] **Step 5: Implement phase execution and atomic checkpoints**

The launcher performs:

```text
prepare/validate data
preflight
single-GPU smoke on GPU6 if owner-free
phase-m to 25
health gate
phase-m to 100
audit100
phase-t to 250 only when audit100 passes
audit250
emit rgb-ready receipt only when audit250 passes
```

If single-GPU smoke fails only the memory gate, stop and require a separately validated fixed FSDP topology. Do not silently alter model scope or batch semantics.

- [ ] **Step 6: Run focused trainer tests and dry-run**

Run:

```bash
PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest tests/test_wan_v8_trainer.py tests/test_wan_v8_scripts.py -q
DRY_RUN=1 bash scripts/run_wan_v8_direct_action_band.sh
```

Expected: tests pass; dry-run reports zero started processes and exact ordered commands.

- [ ] **Step 7: Commit Task 7**

```bash
git add scripts/train_wan_v8_direct_action_band.py scripts/run_wan_v8_direct_action_band.sh scripts/validate_wan_v8_sync_closure.py tests/test_wan_v8_trainer.py tests/test_wan_v8_scripts.py src/worldarena_baseline/wan_v8_sync_closure.py
git commit -m "feat: add guarded v8 direct action run"
```

### Task 8: Remote Verification, Data Completion, Smoke, and Bounded Run

**Files:**
- Create: `reports/2026-08-18-v8-direct-action-band.md`
- Modify only if required by verified failures: files owned by Tasks 1-7 and their focused tests.

**Interfaces:**
- Consumes: clean local commits and reviewed source closure.
- Produces: remote zero-skip test evidence, complete data receipt, smoke receipt, step25/100/250 artifacts as allowed, and a detailed final report.

- [ ] **Step 1: Validate the committed recursive source closure locally**

Run: `PYTHONPATH=src uv run python scripts/validate_wan_v8_sync_closure.py --output /private/tmp/v8-source-closure.json`

Expected: all v8 transitive runtime dependencies are tracked, clean or explicitly source-pinned, and included in the closure digest.

- [ ] **Step 2: Sync only the reviewed closure to the remote source root**

Use the closure manifest rather than a broad working-tree copy. Re-run the validator remotely and require the same digest before tests.

- [ ] **Step 3: Run the complete remote v8 test selection**

Run:

```bash
PYTHONPATH=src /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest \
  tests/test_wan_v8_data.py \
  tests/test_wan_v8_counterfactual.py \
  tests/test_wan_v8_attention.py \
  tests/test_wan_v8_model.py \
  tests/test_wan_v8_objective.py \
  tests/test_wan_v8_training.py \
  tests/test_wan_v8_audit.py \
  tests/test_wan_v8_trainer.py \
  tests/test_wan_v8_scripts.py -q
```

Expected: zero failures and zero unexpected skips.

- [ ] **Step 4: Complete CPU-only v8 data artifacts**

Run `scripts/prepare_wan_v8_data.py` under the formal Python. Require:

- correct SE(3): 1,785/1,785;
- each complete negative family: 1,785/1,785;
- audit20: exactly 20 held-out rows;
- optimizer/audit/dev overlap: zero;
- replay: exact 250-step topology-bound records;
- a second validator pass with zero rewrites;
- at least 150 GiB disk free after preparation.

- [ ] **Step 5: Run guarded preflight and production smoke**

Do not claim readiness from unit tests. Run the actual correct/wrong production-shape optimizer loop and verify every memory/gradient/optimizer/T5/VAE gate from the smoke receipt.

- [ ] **Step 6: Run Phase M and obey gates**

Train to step25, evaluate health, then continue to step100 only if healthy. Run held-out audit100 with gates enabled and forced zero. If step100 fails, stop and write the report; do not create a Phase T optimizer update.

- [ ] **Step 7: Run Phase T only after a passing step100**

Calibrate position/velocity weights, continue the same optimizer lineage to step250, and run the hard audit. Decode RGB only if step250 passes every gate.

- [ ] **Step 8: Write and verify the detailed report**

The report must contain data coverage, leakage evidence, exact trainable count, calibration norms/weights, smoke memory per rank, full training curves, enabled/zero-gate audits, phase decisions, checkpoint hashes, GPU release state, failures encountered, and the next decision. Verify report SHA256 and copy it under `/data/di/worldarena2_track1_20260815/reports`.

- [ ] **Step 9: Commit the report without touching unrelated files**

```bash
git add reports/2026-08-18-v8-direct-action-band.md
git commit -m "docs: record v8 direct action band run"
```

## Completion Criteria

- The raw dataset is not expanded or downloaded for Stage A.
- All 1,785 correct and negative derived conditions are complete and provenance-bound.
- audit20 is held out from probe fitting, optimizer replay, and dev-fast20.
- Remote focused tests pass with no unexpected Torch skips.
- Production smoke passes the 22 GiB hard gate on the chosen fixed topology.
- Phase T cannot run without a passing step100 receipt.
- RGB cannot run without a passing step250 receipt.
- All outputs remain under the authoritative `/data/di` root.
- The final report preserves negative results as first-class evidence.
