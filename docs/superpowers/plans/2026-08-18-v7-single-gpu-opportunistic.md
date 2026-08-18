# V7 Single-GPU Opportunistic Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the bounded v7 SE(3) mechanism probe on GPU6 as an isolated one-rank lineage.

**Architecture:** Add explicit single-GPU replay/checkpoint contracts beside the existing seven-rank contract. A dedicated launcher accepts only physical GPU6, performs source/cache validation and a three-step production smoke before bounded training.

**Tech Stack:** Python, PyTorch/FSDP-compatible distributed runtime, Bash, existing Wan v7 contracts.

**Spec:** `docs/superpowers/specs/2026-08-18-v7-single-gpu-opportunistic-design.md`

## Global Constraints

- Never alter or resume seven-rank v7 artifacts.
- Use only GPU6; never allocate GPU0-5 or GPU7.
- Preserve 81 frames, 480x640, frozen parent, and gate-only training.
- All outputs must be below `/data/di/worldarena2_track1_20260815/runs/v7-se3-single-gpu/`.
- Fail closed for OOM, non-finite values, zero gate gradient, original parameter gradient, or peak >=22 GiB.

---

### Task 1: Single-GPU lineage contract

**Files:**
- Modify: `src/worldarena_baseline/wan_v7_training.py`
- Create: `tests/test_wan_v7_single_gpu.py`

**Interfaces:**
- Consumes: trusted v6 replay and existing v7 gate state.
- Produces: `build_v7_single_gpu_replay(v6_replay) -> dict` and strict single-GPU checkpoint validation.

- [ ] **Step 1: Write failing topology-isolation tests**

```python
def test_single_gpu_replay_uses_only_physical_rank_six():
    replay = build_v7_single_gpu_replay(v6_replay)
    assert replay["world_size"] == 1
    assert replay["rank_mapping"] == [6]
    assert all(len(step) == 1 for step in replay["records"])

def test_single_gpu_checkpoint_rejects_seven_rank_replay():
    with pytest.raises(ValueError, match="single-gpu"):
        build_v7_single_gpu_checkpoint(replay=seven_rank_replay, ...)
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_single_gpu.py`

Expected: import failures for missing single-GPU builders.

- [ ] **Step 3: Implement deterministic rank-6 replay and isolated checkpoint contract**

```python
def build_v7_single_gpu_replay(v6_replay: Mapping[str, Any]) -> dict[str, Any]:
    canonical = _validate_v6_prefix(v6_replay)
    return _with_hash({
        "contract": "wan-action-v7-se3-single-gpu-replay/1",
        "world_size": 1,
        "rank_mapping": [6],
        "records": [[step[6]] for step in canonical["records"][:50]],
    })
```

Validate all input/output contracts exactly and use a separate checkpoint
contract string.

- [ ] **Step 4: Run GREEN and commit**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_single_gpu.py`

Commit: `feat: add v7 single gpu lineage contract`

### Task 2: Dedicated GPU6 runner and smoke

**Files:**
- Create: `scripts/run_wan_se3_probe_v7_single_gpu.sh`
- Modify: `scripts/train_wan_se3_probe_v7_fsdp.py`
- Modify: `src/worldarena_baseline/wan_v7_sync_closure.py`
- Modify: `source_inputs/wan-v7-stagea-sync-closure.json`
- Modify: `tests/test_wan_v7_scripts.py`

**Interfaces:**
- Consumes: single-GPU replay, cache, parent, and source closure.
- Produces: preflight/smoke/checkpoints/audits under the dedicated run root.

- [ ] **Step 1: Write failing launcher tests**

```python
def test_single_gpu_launcher_uses_only_gpu6():
    result = run_bash("scripts/run_wan_se3_probe_v7_single_gpu.sh", "dry-run")
    assert result.returncode == 0
    assert "CUDA_VISIBLE_DEVICES=6" in result.stdout
    assert "nproc_per_node=1" in result.stdout
    assert "v7-se3-single-gpu" in result.stdout
```

- [ ] **Step 2: Run RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_scripts.py -k single_gpu`

Expected: missing launcher.

- [ ] **Step 3: Implement one-rank mode and guarded launcher**

Add a `--topology single-gpu` mode that validates the isolated replay/checkpoint
contract before device initialization. The launcher must export exactly
`CUDA_VISIBLE_DEVICES=6`, use `torch.distributed.run --nproc_per_node=1`, run
the existing three-step smoke, and reject cache/closure/receipt mismatches.

- [ ] **Step 4: Run GREEN and commit**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v7_scripts.py -k single_gpu`

Commit: `feat: add v7 gpu6 opportunistic runner`

### Task 3: Remote bounded execution

**Files:**
- No local source modifications unless verification fails.

- [ ] **Step 1: Sync the validated source closure and focused tests only**

Run remote closure validation and the focused PyTorch suite with zero skips.

- [ ] **Step 2: Run GPU6 production smoke**

Run the launcher smoke phase. Confirm every recorded peak is below 22 GiB and
gate gradients are finite/nonzero.

- [ ] **Step 3: Run train10 then train25/audit25**

Never launch step50 in the same command. Validate all generated artifact
contracts and report the discovery gate.
