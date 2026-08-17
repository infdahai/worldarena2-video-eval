# Wan-Action v7 SE(3) Mechanism Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a bounded three-block arm-grouped SE(3) attention probe that changes no pretrained Wan parameters and determines whether analytic relative rigid-motion attention creates correct action counterfactual separation.

**Architecture:** Generate an immutable SE(3) condition from original end-effector trajectories, fork frozen Wan Q/K/V after Q/K normalization and before 3D RoPE, and run a second arm-grouped geometric attention call in blocks 8/16/24. Only a zero-initialized `(24,128)` channel gate per block is trainable; the original attention, frozen Wan O, and clean parent Adapter remain unchanged.

**Tech Stack:** Python 3.11, PyTorch, Wan2.2-TI2V-5B, FlashAttention/SDPA, NumPy, HDF5, pytest, torch.distributed with seven RTX 4090 ranks.

**Spec:** `docs/superpowers/specs/2026-08-18-v7-se3-mechanism-probe-design.md`

## Global Constraints

- Persistent outputs are restricted to `/data/di/worldarena2_track1_20260815`.
- Remote source is `/home/huazhi/nlh/baseline`; Wan source is `/home/huazhi/nlh/Wan2.2`.
- GPU0-6 are the only legal training devices; GPU7 must never be selected.
- Parent checkpoint SHA256 is `105fb760fd371885ba362d26ef2352c260755e47cd036f46711181edc3b30ca2`.
- Stage A uses clean-1000, a maximum of 50 optimizer steps, and checkpoints 10/25/50.
- Dev-fast20, discovery rows, and official test rows are illegal training rows.
- Wan, the support-gated parent Adapter, Wan Q/K/V/O, and the gripper probe remain frozen.
- The only trainable tensors are three gates of shape `(24,128)`, for exactly 9,216 scalar parameters.
- Stage A uses existing weighted flow matching only; no gripper bias, trajectory loss, QKV LoRA, or Stage B code is enabled.
- T5 and VAE are forbidden from the training hot path.
- Stage B six-block work requires a separate explicit user approval after Stage A and RGB gates pass.

---

## File Structure

- `src/worldarena_baseline/wan_se3_condition.py`: authoritative SE(3) construction, temporal sampling, anchoring, normalization, cache schema, and validation.
- `src/worldarena_baseline/wan_se3_attention.py`: pure tensor group-action transforms, geometry attention branch, gate parameters, and Wan self-attention wrapper.
- `src/worldarena_baseline/wan_v7_model.py`: install/remove wrappers in blocks 8/16/24, inject the frozen parent Adapter, set/clear per-forward conditions, and enforce the trainable whitelist.
- `src/worldarena_baseline/wan_v7_training.py`: immutable data/replay/optimizer/checkpoint/audit contracts.
- `scripts/cache_wan_v7_se3_conditions.py`: build provenance-bound SE(3) sidecars from clean-1000 source rows.
- `scripts/build_wan_v7_replay.py`: derive the matched 50-step v7 replay from the immutable v6 replay.
- `scripts/train_wan_se3_probe_v7_fsdp.py`: phase-1 audit, production smoke, bounded training, and discovery audit.
- `scripts/run_wan_se3_probe_v7.sh`: fail-closed dry-run and seven-rank phase launcher.
- `tests/test_wan_se3_condition.py`: geometry/cache contracts.
- `tests/test_wan_se3_attention.py`: operator and raw-QKV fork contracts.
- `tests/test_wan_v7_model.py`: complete wrapper, parent equality, and whitelist contracts.
- `tests/test_wan_v7_training.py`: replay, optimizer, checkpoint, and promotion gates.
- `tests/test_wan_v7_scripts.py`: CLI, path, hot-path, GPU, and launcher contracts.

### Task 1: Authoritative SE(3) condition and cache

**Files:**
- Create: `src/worldarena_baseline/wan_se3_condition.py`
- Create: `scripts/cache_wan_v7_se3_conditions.py`
- Create: `tests/test_wan_se3_condition.py`

**Interfaces:**
- Consumes: raw `(T,7)` end poses in `xyz + quaternion_wxyz`, `EpisodeTimeline`, and clean-1000 manifest rows.
- Produces: `build_se3_condition(left_endpose, right_endpose, timeline, left_present=None, right_present=None) -> SE3Condition`; `validate_se3_cache(path, expected_source_sha256) -> dict[str, np.ndarray]`; atomic `.npz` sidecars containing `arm_transform` and `arm_present`.

- [ ] **Step 1: Write failing tests for anchor, interpolation, invariance, presence, and schema**

```python
def test_common_global_transform_does_not_change_condition():
    base = build_fixture_trajectories()
    shifted = left_multiply_all(base, fixture_global_transform())
    actual = build_se3_condition(*base, timeline=timeline())
    transformed = build_se3_condition(*shifted, timeline=timeline())
    np.testing.assert_allclose(actual.arm_transform, transformed.arm_transform, atol=1e-5)

def test_missing_left_uses_right_anchor_but_left_stays_absent():
    result = build_se3_condition(
        left_endpose=None,
        right_endpose=right_fixture(),
        timeline=timeline(),
    )
    assert result.anchor_arm == "right"
    assert not result.arm_present[0].any()
    assert result.arm_present[1].all()
    np.testing.assert_array_equal(result.arm_transform[0], np.eye(4)[None])

def test_cache_rejects_standardized_11d_pose_and_wrong_source_hash(tmp_path):
    path = write_v7_cache(tmp_path, arm_transform=np.zeros((2, 21, 11)))
    with pytest.raises(ValueError, match="2,21,4,4"):
        validate_se3_cache(path, expected_source_sha256="a" * 64)
```

- [ ] **Step 2: Run the tests and record RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_se3_condition.py`

Expected: collection fails because `worldarena_baseline.wan_se3_condition` does not exist.

- [ ] **Step 3: Implement the condition dataclass and group construction**

```python
@dataclass(frozen=True)
class SE3Condition:
    arm_transform: np.ndarray  # (2, 21, 4, 4), float32
    arm_present: np.ndarray    # (2, 21), bool
    anchor_arm: str
    motion_scale: float

def build_se3_condition(..., epsilon: float = 1e-6) -> SE3Condition:
    # Validate raw poses, sample translation + SLERP, select left/right/identity
    # anchor, compute inverse(anchor) @ pose, normalize a shared translation
    # scale, invert normalized frames, and store identity for absent arms.
```

Use the existing validated quaternion and SLERP helpers from
`action_condition.py`; do not average rotation matrices or quaternion values.
Reject non-finite values, invalid homogeneous rows, determinants outside
`1 +/- 1e-4`, and non-orthogonal rotations outside `1e-4`.

- [ ] **Step 4: Implement atomic cache publication and shared validation**

The sidecar must contain exact keys:

```python
{
    "schema": "wan-action-v7-se3-condition/1",
    "arm_transform": float32_array_2x21x4x4,
    "arm_present": bool_array_2x21,
    "anchor_arm": scalar_unicode,
    "motion_scale": scalar_float64,
    "source_episode_sha256": scalar_lowercase_sha256,
    "source_manifest_sha256": scalar_lowercase_sha256,
    "temporal_contract": "81-to-21-causal-v3",
}
```

Write to a same-directory partial file, `fsync`, validate the partial with the
same public validator, then `os.replace`. Reuse an existing sidecar only after
full validation; quarantine only the corrupt sample pair before rebuilding.

- [ ] **Step 5: Run focused tests and compile checks**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_wan_se3_condition.py
python -m py_compile src/worldarena_baseline/wan_se3_condition.py scripts/cache_wan_v7_se3_conditions.py
git diff --check
```

Expected: all tests pass and no compile/diff errors.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/worldarena_baseline/wan_se3_condition.py scripts/cache_wan_v7_se3_conditions.py tests/test_wan_se3_condition.py
git commit -m "feat: add v7 se3 condition contract"
```

### Task 2: Pure arm-grouped geometry operator

**Files:**
- Create: `src/worldarena_baseline/wan_se3_attention.py`
- Create: `tests/test_wan_se3_attention.py`

**Interfaces:**
- Consumes: Q/K/V `(B,L,24,128)`, grid sizes `(B,3)`, transforms `(B,2,21,4,4)`, presence `(B,2,21)`, sequence lengths, and an injected attention callable.
- Produces: `apply_group_action(value, matrix, transpose=False) -> Tensor`; `ArmGroupedSE3Geometry.forward(q,k,v,...) -> Tensor`; `SE3AugmentedSelfAttention.forward(x,seq_lens,grid_sizes,freqs) -> Tensor`.

- [ ] **Step 1: Write the materialized-Kronecker and head-routing tests**

```python
def test_broadcast_transform_matches_materialized_kron():
    value = torch.randn(2, 5, 4, 8, dtype=torch.float64)
    matrix = fixture_se3(batch=2, tokens=5, heads=4)
    actual = apply_group_action(value, matrix)
    expected = apply_materialized_kron_reference(value, matrix)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)

def test_absent_arm_is_exact_zero_before_output_projection():
    result = geometry(qkv(), transforms(), presence=torch.tensor([[[0]*21,[1]*21]]))
    assert torch.count_nonzero(result[:, :, :12]) == 0
    assert torch.count_nonzero(result[:, :, 12:]) > 0
```

Also cover transpose/inverse orientation, 21x15x20 token indexing, text-token
padding, wrong head width, non-finite transforms, and swapped assignment.

- [ ] **Step 2: Run the tests and record RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_se3_attention.py`

Expected: collection fails because the operator module is absent.

- [ ] **Step 3: Implement the non-materialized group action**

```python
def apply_group_action(value: Tensor, matrix: Tensor, *, transpose: bool = False) -> Tensor:
    if value.shape[-1] % 4:
        raise ValueError("head_dim must be divisible by four")
    blocks = value.float().reshape(*value.shape[:-1], value.shape[-1] // 4, 4)
    transform = matrix.float().transpose(-1, -2) if transpose else matrix.float()
    output = torch.einsum("...rc,...nc->...nr", transform, blocks)
    return output.reshape_as(value).to(value.dtype)
```

Lock the row/column convention against the explicit Kronecker reference rather
than relying on the example alone. Compute matrix inverses once per cached
condition, not once per token.

- [ ] **Step 4: Implement fixed 12/12 head ownership and gated output**

`ArmGroupedSE3Geometry` must expand latent-frame transforms over spatial tokens
without allocating per-token 128x128 matrices. Apply `D^T` to Q, `D^-1` to K/V,
call the injected production attention function, apply `D` to its output,
zero absent heads, and return `(B,L,24,128)` before Wan O.

- [ ] **Step 5: Implement the raw-QKV self-attention wrapper**

`SE3AugmentedSelfAttention` wraps one frozen Wan self-attention module. It
reproduces its Q/K/V projection and Q/K normalization exactly once, calls the
original RoPE path unchanged, runs the geometry path on pre-RoPE tensors, then
computes:

```python
geometry_heads = geometry(...) * gate[None, None]
geometry_output = base.o(geometry_heads.flatten(2))
return original_output + geometry_output
```

The gate is float32 `(24,128)` and zero initialized. The wrapper receives
`rope_apply_fn` and `attention_fn` explicitly from the loaded Wan runtime; it
must not import or vendor a second attention implementation.

- [ ] **Step 6: Verify exact original-path and zero-gate equality**

Use a deterministic fake Wan attention module and injected deterministic
attention callable. Assert raw Q/K/V are forked before RoPE, original output is
bitwise equal with the wrapper disabled, zero-gate complete output is bitwise
equal, and only the gate receives gradients.

- [ ] **Step 7: Run focused tests and commit Task 2**

```bash
PYTHONPATH=src pytest -q tests/test_wan_se3_attention.py
python -m py_compile src/worldarena_baseline/wan_se3_attention.py
git diff --check
git add src/worldarena_baseline/wan_se3_attention.py tests/test_wan_se3_attention.py
git commit -m "feat: add arm grouped se3 attention"
```

### Task 3: Complete clean-parent Wan integration

**Files:**
- Create: `src/worldarena_baseline/wan_v7_model.py`
- Create: `tests/test_wan_v7_model.py`

**Interfaces:**
- Consumes: loaded frozen Wan backbone, frozen clean parent Adapter, injected Wan `rope_apply` and `attention`, and v7 SE(3) tensors.
- Produces: `install_v7_attention(backbone, block_indices, rope_apply_fn, attention_fn) -> dict[int, SE3AugmentedSelfAttention]`; `ParentPlusSE3Wan`; `v7_trainable_parameter_names(model) -> set[str]`.

- [ ] **Step 1: Write failing integration tests**

Tests must prove:

```python
assert torch.equal(parent_output, v7_zero_gate_output)
assert selected_blocks == (8, 16, 24)
assert trainable_names == {
    "geometry_wrappers.8.gate",
    "geometry_wrappers.16.gate",
    "geometry_wrappers.24.gate",
}
assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9216
```

Also assert condition state is cleared in `finally` after success and exception,
the same condition survives activation-checkpoint recomputation, null action
zeros both head groups, and existing parent raster hooks still fire once.

- [ ] **Step 2: Run the tests and record RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_model.py`

Expected: collection fails because `wan_v7_model` is absent.

- [ ] **Step 3: Implement wrapper installation without modifying Wan source**

Replace only `backbone.blocks[index].self_attn` after the parent checkpoint and
Wan weights are loaded. Preserve the original module as `wrapper.base`; freeze
it before replacement. Reject already wrapped modules, missing blocks, wrong
width/head count, duplicated indices, or any index other than 8/16/24 in Stage A.

- [ ] **Step 4: Implement the complete model wrapper**

`ParentPlusSE3Wan.forward` accepts the existing raster/support/action-present
arguments plus:

```python
se3_arm_transform: Tensor  # (B,2,21,4,4)
se3_arm_present: Tensor    # (B,2,21)
```

It computes frozen parent residuals under `torch.no_grad()`, registers the
existing block pre-hooks, binds the SE(3) condition to all three attention
wrappers, executes Wan, and clears hooks/conditions in one `finally` block.

- [ ] **Step 5: Run integration tests and commit Task 3**

```bash
PYTHONPATH=src pytest -q tests/test_wan_v7_model.py tests/test_wan_se3_attention.py
python -m py_compile src/worldarena_baseline/wan_v7_model.py
git diff --check
git add src/worldarena_baseline/wan_v7_model.py tests/test_wan_v7_model.py
git commit -m "feat: integrate v7 geometry attention with Wan"
```

### Task 4: Replay, optimizer, checkpoint, and promotion contracts

**Files:**
- Create: `src/worldarena_baseline/wan_v7_training.py`
- Create: `scripts/build_wan_v7_replay.py`
- Create: `tests/test_wan_v7_training.py`

**Interfaces:**
- Consumes: immutable v6 replay, clean-1000 manifest hash, parent hash, source hash, and three-gate state dict.
- Produces: `v7_training_contract()`, `build_v7_replay_from_v6(...)`, `v7_optimizer_group(model, calibrated_lr)`, `build_v7_checkpoint(...)`, `validate_v7_checkpoint(...)`, and `v7_discovery_gate(metrics)`.

- [ ] **Step 1: Write failing deterministic replay and checkpoint tests**

```python
def test_v7_replay_is_exact_first_50_steps_of_v6():
    actual = build_v7_replay_from_v6(v6_payload())
    assert actual["records"] == v6_payload()["records"][:50]
    assert actual["max_steps"] == 50

def test_checkpoint_rejects_any_non_gate_trainable_state():
    payload = valid_checkpoint()
    payload["model"]["backbone.blocks.8.self_attn.q.weight"] = torch.ones(1)
    with pytest.raises(ValueError, match="gate-only"):
        validate_v7_checkpoint(payload, expected=expected_contract())
```

Cover exact world size/rank map, step 10/25/50, replay SHA, parent SHA, source
hashes, cache hash, nonempty optimizer state, gate shapes/dtypes, and resume
step consistency.

- [ ] **Step 2: Run the tests and record RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_training.py`

Expected: collection fails because the training-contract module is absent.

- [ ] **Step 3: Implement immutable Stage A contract and replay derivation**

The contract returns exact values:

```python
{
    "contract": "wan-action-v7-se3-mechanism/1",
    "world_size": 7,
    "rank_mapping": [0,1,2,3,4,5,6],
    "dataset_rows": 1000,
    "max_steps": 50,
    "checkpoint_steps": [10,25,50],
    "injection_points": [8,16,24],
    "head_groups": {"left": [0,12], "right": [12,24]},
    "trainable_parameters": 9216,
    "loss": "weighted_flow_matching_only",
}
```

Reject a v6 replay whose dataset hash, world size, rank map, or first 50
records differs from the clean-1000 source contract.

- [ ] **Step 4: Implement calibrated single optimizer group and gates**

The optimizer group name is `se3_channel_gates`; it contains exactly three
unique parameters. Store the calibrated learning rate in the preflight receipt
and checkpoint rather than leaving it as a launcher default.

The step-25 gate passes only when at least one counterfactual family shows
paired separation and position or velocity direction is positive. The step-50
gate requires aggregate wins >=6/8, positive position and velocity improvement,
routing retention >=0.90, and FM regression <=0.02.

- [ ] **Step 5: Run tests and commit Task 4**

```bash
PYTHONPATH=src pytest -q tests/test_wan_v7_training.py
python -m py_compile src/worldarena_baseline/wan_v7_training.py scripts/build_wan_v7_replay.py
git diff --check
git add src/worldarena_baseline/wan_v7_training.py scripts/build_wan_v7_replay.py tests/test_wan_v7_training.py
git commit -m "feat: add v7 training and replay contracts"
```

### Task 5: Seven-rank trainer, mechanism audit, and launcher

**Files:**
- Create: `scripts/train_wan_se3_probe_v7_fsdp.py`
- Create: `scripts/run_wan_se3_probe_v7.sh`
- Create: `tests/test_wan_v7_scripts.py`

**Interfaces:**
- Consumes: Tasks 1-4, cached Wan latents/text/raster, v7 SE(3) sidecars, clean parent, frozen probe, v7 replay, and zero-leakage receipt.
- Produces: preflight gradient audit JSON, production smoke JSON, checkpoints 10/25/50, discovery audit JSON, and a guarded launcher with phases `dry-run`, `preflight`, `smoke`, `train10`, `train25`, `audit25`, `train50`, and `audit50`.

- [ ] **Step 1: Write failing CLI and static hot-path tests**

```python
def test_v7_launcher_excludes_gpu7_and_cannot_skip_audit25():
    result = run_bash("scripts/run_wan_se3_probe_v7.sh", "dry-run")
    assert result.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6" in result.stdout
    assert "audit25" in result.stdout
    assert "train50 requires audit25 pass" in result.stdout
    assert "GPU7" not in result.stdout

def test_v7_trainer_has_no_hot_path_encoder_or_stage_b_symbols():
    source = Path("scripts/train_wan_se3_probe_v7_fsdp.py").read_text()
    assert "WanActionCachedDataset" in source
    assert "T5EncoderModel" not in source
    assert "WanVAE" not in source
    assert "LoRA" not in source
    assert "gripper_bias" not in source
```

- [ ] **Step 2: Run tests and record RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_scripts.py`

Expected: failures because trainer and launcher do not exist.

- [ ] **Step 3: Implement preflight and production smoke**

Preflight validates all hashes and leakage receipts, loads the cached-only
dataset, installs v7 wrappers after loading Wan/parent weights, validates the
9,216-parameter whitelist, and records correct/reverse/shift/swap geometry
feature and gate-gradient statistics.

Smoke executes three complete production-shape
`forward -> backward -> optimizer.step -> zero_grad` iterations. Reset CUDA
peak counters after initialization/warmup. Record allocated/reserved peak and
step time for every rank. Any rank >=22 GiB, non-finite loss/gradient, zero
gate gradient, original-parameter gradient, or GPU topology mismatch fails.

- [ ] **Step 4: Implement bounded training and exact resume**

Use the derived replay record for each global step/rank. Save atomically only
at 10, 25, and 50. Checkpoints include source/config/cache/replay/parent hashes,
optimizer state, completed step, rank map, and the three gate tensors. Resume
loads through `validate_v7_checkpoint` before restoring optimizer state.

- [ ] **Step 5: Implement matched counterfactual audit**

Audit correct/reverse/shift/swap with the same sample, noise, and timestep on
the fixed eight discovery rows. Reuse the frozen v6 gripper probe for position
and velocity, and existing v6 helpers for FM/routing metrics. Write raw
per-episode metrics and the deterministic gate decision. Audit does not decode
RGB.

- [ ] **Step 6: Implement launcher gates and `/data/di` path guards**

`train50` must first validate `audit-step-000025.json` with the shared gate
function. No shell-only `test -f` may approve a checkpoint or audit. All
writable paths must resolve below the formal root before creation. Dry-run
performs no download, GPU initialization, worker launch, or file creation.

- [ ] **Step 7: Run local focused regression and commit Task 5**

```bash
PYTHONPATH=src pytest -q \
  tests/test_wan_se3_condition.py \
  tests/test_wan_se3_attention.py \
  tests/test_wan_v7_model.py \
  tests/test_wan_v7_training.py \
  tests/test_wan_v7_scripts.py
python -m py_compile scripts/train_wan_se3_probe_v7_fsdp.py
bash -n scripts/run_wan_se3_probe_v7.sh
git diff --check
git add scripts/train_wan_se3_probe_v7_fsdp.py scripts/run_wan_se3_probe_v7.sh tests/test_wan_v7_scripts.py
git commit -m "feat: add bounded v7 mechanism run"
```

### Task 6: Remote verification and bounded execution

**Files:**
- Modify only if evidence requires a reviewed repair: files created in Tasks 1-5.
- Create remotely: `/data/di/worldarena2_track1_20260815/cache/v7-se3-clean1000/`
- Create remotely: `/data/di/worldarena2_track1_20260815/runs/v7-se3-mechanism-probe/`
- Create remotely: `/data/di/worldarena2_track1_20260815/runs/v7-se3-mechanism-smoke/`

**Interfaces:**
- Consumes: reviewed local commits and existing formal assets.
- Produces: formal test evidence, 1,000 validated SE(3) sidecars, preflight/smoke receipts, bounded checkpoints, and step-25/50 decisions.

- [ ] **Step 1: Sync only reviewed v7 scoped files**

Use an explicit file list. Do not sync the entire dirty repository and do not
read or copy `/data/fjy` or `/home/fjy`.

- [ ] **Step 2: Run formal-Python focused tests with zero skips**

```bash
cd /home/huazhi/nlh/baseline
PYTHONPATH=/home/huazhi/nlh/baseline/src:/home/huazhi/nlh/Wan2.2 \
  /data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest -q \
  tests/test_wan_se3_condition.py \
  tests/test_wan_se3_attention.py \
  tests/test_wan_v7_model.py \
  tests/test_wan_v7_training.py \
  tests/test_wan_v7_scripts.py
```

Expected: all pass, zero PyTorch skips.

- [ ] **Step 3: Run dry-run and cache clean-1000 conditions**

Verify disk reserve before cache publication. Build exactly 1,000 sidecars,
then independently validate every sidecar and its source/manifest hashes.
Do not load Wan, VAE, or T5 for this phase.

- [ ] **Step 4: Run preflight and seven-rank production smoke**

Check GPU0-6 process ownership immediately before launch and leave GPU7
untouched. Preflight must show distinct correct/counterfactual geometry features
and nonzero finite gate gradients. Smoke must pass all per-rank 22 GiB and
whitelist gates.

- [ ] **Step 5: Train to step 10 and step 25, then stop for audit**

Do not launch step 50 in the same command. Validate checkpoint payloads and
hashes, run the fixed discovery audit, and report the raw paired metrics and
gate decision.

- [ ] **Step 6: Continue to step 50 only when the shared step-25 gate passes**

If step 25 fails, record the negative result and stop without RGB generation.
If it passes, train to step 50, run the stricter audit, and decode 4-8 RGB
episodes only after the step-50 gate passes.

- [ ] **Step 7: Produce the v7 report and request Stage B approval**

The report records lineage, data exposure, mathematical tests, gradients,
memory, checkpoint hashes, counterfactual raw rows, aggregate gates, and RGB
anti-exploitation evidence. It must distinguish mechanism compatibility from
official trajectory improvement. A six-block Stage B plan begins only after
the user reviews this report and explicitly approves promotion.
