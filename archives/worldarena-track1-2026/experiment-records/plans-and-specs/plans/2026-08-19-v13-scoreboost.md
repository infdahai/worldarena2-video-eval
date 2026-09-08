# v13-ScoreBoost Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run one bounded v13 experiment that improves robot/object integrity and relative geometry while preserving the `clean-gated-step10` trajectory incumbent.

**Architecture:** Keep the complete action-conditioned parent frozen, inject rank-16 LoRA only into Wan blocks 18–25 self-attention Q/V/O, and attach one training-only relative-depth head to block 25. SAM3 masks and Depth Anything V2 Small labels are computed offline for 21 latent-aligned frames; the trainer reads only cached latent, text, action condition, masks, and depth.

**Tech Stack:** Python 3, PyTorch, Wan2.2-TI2V-5B, HDF5/NumPy caches, SAM3, Depth Anything V2 Small, pytest, shell launchers, WorldArena commit `7b3feee108427bee3380064bb5154970ed7468b5`.

**Spec:** `docs/superpowers/specs/2026-08-19-v13-scoreboost-design.md`

## Global Constraints

- Parent checkpoint is exactly `clean-gated-step10`; every native Wan and parent parameter remains frozen.
- Trainable parameters are exactly rank/alpha `16/16` LoRA weights for `self_attn.q/v/o` in blocks `18..25` plus the training-only depth head.
- LoRA LR is `5e-6`; depth-head LR is `1e-4`; no LR sweep.
- Training data is exactly 150 unique, task-balanced episodes from clean optimizer-2060 with zero overlap against audit20, dev-fast20, dev-clean50, RGB8, and official test when available.
- Offline labels use exactly the 21 RGB frames aligned to Wan latent slots and produce arrays on the `21x30x40` latent grid.
- SAM3 and Depth Anything V2 Small never load in the trainer or inference process.
- Objective is exactly `L_fm + 0.5*L_mask + 0.1*L_temporal + 0.1*L_depth`.
- Training is single-process on physical GPU6 only, maximum 150 exposures, checkpoints at 100 and 150, warmup 10, micro-batch 1.
- Production smoke runs three full optimizer iterations and requires allocated and reserved CUDA peaks below 22 GiB.
- All persistent artifacts are descendants of `/data/di/worldarena2_track1_20260815`; GPU0–5 and GPU7 are not touched.
- No phase, SE(3), PRA, top-k, GRU/TCN, JEPA training loss, GAN, DMD, counterfactual, trajectory, or new action branch is permitted.
- Existing dirty files outside the v13 scope are preserved.

---

## File Map

| File | Responsibility |
|---|---|
| `src/worldarena_baseline/wan_v13_data.py` | deterministic split, zero-leakage receipt, cache schema, dataset |
| `src/worldarena_baseline/wan_v13_labels.py` | 21-frame alignment, SAM3 mask validation, depth normalization/cache publication |
| `source_inputs/wan_v13_object_prompts.json` | reviewed task-to-manipulated-object prompt map |
| `src/worldarena_baseline/wan_v13_model.py` | Q/V/O LoRA installation, block-25 hook, depth head, whitelist |
| `src/worldarena_baseline/wan_v13_objective.py` | FM, masked clean-latent, object temporal, and depth losses |
| `src/worldarena_baseline/wan_v13_training.py` | optimizer, schedule, lineage, checkpoint and resume contracts |
| `src/worldarena_baseline/wan_v13_inference.py` | load parent plus LoRA only; explicitly omit the depth head |
| `src/worldarena_baseline/wan_v13_gate.py` | RGB8 and fast20 fail-closed decision logic |
| `src/worldarena_baseline/wan_v13_sync_closure.py` | recursive runtime-source receipt |
| `scripts/build_wan_v13_data.py` | build train150/RGB8 manifests and immutable receipts |
| `scripts/download_wan_v13_depth_teacher.py` | guarded DAV2-Small download and SHA receipt |
| `scripts/cache_wan_v13_supervision.py` | CPU/GPU offline SAM3 and depth cache production |
| `scripts/train_wan_v13_scoreboost.py` | prepare/preflight/smoke/train/status entry point |
| `scripts/run_wan_v13_gpu6.sh` | physical-GPU6 ownership gate and bounded phase runner |
| `scripts/generate_wan_v13_video.py` | matched parent/v13 video generation worker |
| `scripts/evaluate_wan_v13_gate.py` | RGB8/fast20 paired gate and machine-readable result |
| `scripts/validate_wan_v13_sync_closure.py` | local and remote source-closure verification |

---

### Task 1: Freeze the 150-row optimizer split and held-out RGB8

**Files:**
- Create: `src/worldarena_baseline/wan_v13_data.py`
- Create: `scripts/build_wan_v13_data.py`
- Create: `tests/test_wan_v13_data.py`

**Interfaces:**
- Consumes: optimizer-2060 rows and the existing `assert_zero_dataset_leakage()` identity rules.
- Produces: `select_v13_manifests` returning `V13DataSplit`, `write_v13_data_artifacts` returning `V13DataPaths`, and `validate_v13_data_receipt` returning `dict[str, object]`.

- [ ] **Step 1: Write RED tests for deterministic balancing and zero leakage**

```python
def test_selects_exact_train150_and_heldout_rgb8_without_leakage():
    split = select_v13_manifests(
        optimizer_rows=optimizer_rows,
        strata_by_sample=strata,
        evaluation_rows={
            "audit20": audit20,
            "dev-fast20": fast20,
            "dev-clean50": clean50,
        },
        seed=20260819,
    )
    assert len(split.train) == 150
    assert len(split.rgb8) == 8
    assert len({row["sample"] for row in split.train}) == 150
    assert {row["sample"] for row in split.train}.isdisjoint(
        row["sample"] for row in split.rgb8
    )
    assert [row["category"] for row in split.rgb8].count("single") == 4
    assert [row["category"] for row in split.rgb8].count("bimanual_crossing") == 4
```

Add rejection tests for duplicate source identities, missing task strata, any evaluation collision, official-test collision when that manifest exists, noncanonical row order, and a caller-supplied foreign manifest.

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_data.py`

Expected: collection or assertion failure because `wan_v13_data` does not exist.

- [ ] **Step 3: Implement the exact data contracts**

```python
@dataclass(frozen=True)
class V13DataSplit:
    train: Sequence[dict[str, object]]
    rgb8: Sequence[dict[str, object]]
    task_counts: dict[str, int]
    exclusion_sha256: dict[str, str]
```

Implement `select_v13_manifests` with keyword-only optimizer rows, strata metadata, evaluation rows and seed `20260819`. Use stable SHA256 ranking within each task. Allocate 150 rows by largest-remainder task balancing, never by source order. Choose RGB8 before train150 from probe-observable candidates, exactly four single-arm and four bimanual/crossing, then exclude them from training. Serialize JSONL and a canonical receipt with source/exclusion/output hashes and exact sample arrays; validation must rebuild the selection and byte-compare it.

- [ ] **Step 4: Add the CLI and persistent-root guard**

The CLI must require all input manifests explicitly and only write beneath:

```text
/data/di/worldarena2_track1_20260815/runs/v13-scoreboost/data
```

It writes `train150.jsonl`, `rgb8.jsonl`, and `data-receipt.json` using partial files, `fsync`, and `os.replace`.

- [ ] **Step 5: Run GREEN verification**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_data.py tests/test_dataset_leakage.py`

Expected: all pass.

- [ ] **Step 6: Commit only this task's files**

```bash
git add src/worldarena_baseline/wan_v13_data.py scripts/build_wan_v13_data.py tests/test_wan_v13_data.py
git commit -m "feat: freeze v13 scoreboost data split"
```

---

### Task 2: Produce strict offline SAM3 and relative-depth labels

**Files:**
- Create: `src/worldarena_baseline/wan_v13_labels.py`
- Create: `source_inputs/wan_v13_object_prompts.json`
- Create: `scripts/download_wan_v13_depth_teacher.py`
- Create: `scripts/cache_wan_v13_supervision.py`
- Create: `tests/test_wan_v13_labels.py`
- Create: `tests/test_wan_v13_cache_scripts.py`

**Interfaces:**
- Consumes: Task 1 manifests; HDF5 RGB frames; pinned SAM3 checkpoint; downloaded DAV2-Small checkpoint.
- Produces: one `supervision/<sample>.npz` plus sidecar per accepted sample and `validate_v13_supervision_cache`.

- [ ] **Step 1: Write RED tests for alignment, mask validity, and robust depth**

```python
def test_latent_frame_indices_are_exactly_causal_21_slots():
    assert latent_rgb_indices(81) == (0, 4, 8, 12, 16, 20, 24, 28, 32, 36,
                                           40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80)

def test_depth_is_finite_robustly_normalized_and_scale_invariant():
    a = normalize_relative_inverse_depth(raw_depth)
    b = normalize_relative_inverse_depth(raw_depth * 3.0 + 7.0)
    torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)

def test_invalid_object_track_is_rejected_not_published_as_empty():
    with pytest.raises(ValueError, match="object track"):
        build_supervision_payload(robot_masks, broken_object_masks, depth)
```

Cover exact `(21,30,40)` float16 payload shapes, boolean validity arrays, finite values, mask overlap allowed, implausible area jumps rejected, insufficient temporal continuity rejected, prompt-map absence rejected, wrong teacher/source hash rejected, partial-pair recovery, and symlink/path escape rejection.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_labels.py tests/test_wan_v13_cache_scripts.py`

- [ ] **Step 3: Add the source-controlled object prompt map**

Use this schema and require every train150/RGB8 task to resolve to exactly one entry:

```json
{
  "contract": "wan-v13-object-prompts/1",
  "tasks": {
    "task_name": {
      "prompts": ["cup"],
      "min_valid_frames": 12,
      "max_area_ratio_jump": 4.0
    }
  }
}
```

Populate entries from the actual selected tasks during implementation; do not use a generic `object` fallback.

- [ ] **Step 4: Implement pure label construction and validation**

```python
@dataclass(frozen=True)
class V13Supervision:
    robot_mask: np.ndarray       # float16 (21,30,40)
    object_mask: np.ndarray      # float16 (21,30,40)
    object_valid: np.ndarray     # bool (21,)
    inverse_depth: np.ndarray    # float16 (21,30,40)
    frame_indices: np.ndarray    # int16 (21,)
```

Implement `normalize_relative_inverse_depth`, `build_supervision_payload`, and `validate_v13_supervision_payload` around this dataclass. Depth normalization uses finite pixels only: inverse depth, median subtraction, division by `max(MAD, 1e-6)`, clipping to `[-10,10]`. SAM3 masks are conservative max-pooled/area-preserving resized masks. Robot/object masks may overlap.

- [ ] **Step 5: Implement the guarded teacher downloader**

Download only the Depth Anything V2 Small checkpoint into:

```text
/data/di/worldarena2_track1_20260815/models/depth-anything-v2-small
```

Support resumable partial download, optional bandwidth cap, minimum 50 GiB free-space guard, explicit expected filename, SHA256 receipt, and no cache outside `/data/di`. Authentication tokens must come from the environment and must not be logged.

- [ ] **Step 6: Implement offline cache generation**

The cache script loads SAM3 and DAV2 sequentially, not concurrently, to keep peak memory bounded. It processes only the exact 21 HDF5 RGB indices, writes one sample atomically, validates it after publication, and deterministically refills rejected object tracks from the remaining clean pool until train150 and RGB8 are complete. The final receipt binds model hashes, HDF5 identity, prompt-map hash, data receipt, payload hashes, rejected samples and refill reasons.

- [ ] **Step 7: Run GREEN verification and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_wan_v13_labels.py tests/test_wan_v13_cache_scripts.py
python -m py_compile src/worldarena_baseline/wan_v13_labels.py scripts/download_wan_v13_depth_teacher.py scripts/cache_wan_v13_supervision.py
git diff --check
```

Then commit only Task 2 files with `feat: add v13 offline supervision cache`.

---

### Task 3: Implement the v13 LoRA model and training-only depth head

**Files:**
- Create: `src/worldarena_baseline/wan_v13_model.py`
- Create: `tests/test_wan_v13_model.py`
- Modify: `src/worldarena_baseline/wan_lora.py`
- Modify: `tests/test_wan_lora.py`

**Interfaces:**
- Consumes: Wan backbone exposing 30 blocks and the frozen action-conditioned parent.
- Produces: `install_v13_scoreboost` returning `V13ModelReport`, `V13DepthHead`, `capture_block25_hidden`, and exact trainable-name helpers.

- [ ] **Step 1: Write RED tests for the exact installation contract**

```python
def test_v13_installs_only_qvo_lora_in_blocks_18_through_25():
    report = install_v13_scoreboost(backbone, hidden_dim=3072)
    assert report.module_names == tuple(
        f"blocks.{block}.{target}"
        for block in range(18, 26)
        for target in ("self_attn.q", "self_attn.v", "self_attn.o")
    )
    assert all(not p.requires_grad for n, p in backbone.named_parameters()
               if ".lora_" not in n)
```

Add tests for rank/alpha 16/16, no K LoRA, zero-init up matrices preserving parent output bit-for-bit at installation, hook cleanup after success/exception, block-25 tensor reshaping, depth output `(B,21,30,40)`, absence of teacher modules, and exact whitelist rejection.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_model.py tests/test_wan_lora.py`

- [ ] **Step 3: Extend the reusable LoRA helper without changing old defaults**

Keep `DEFAULT_TARGETS=(q,v)` unchanged. Use the existing `targets=` argument with v13's explicit tuple `("self_attn.q", "self_attn.v", "self_attn.o")`. Add only any state/inspection helper that is strictly required; old v1–v12 tests must remain unchanged.

- [ ] **Step 4: Implement model composition and hook lifecycle**

```python
class V13DepthHead(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 256), nn.SiLU(), nn.Linear(256, 1)
        )

    def forward(self, hidden: Tensor) -> Tensor:
        if hidden.ndim != 5 or hidden.shape[-1] != self.norm.normalized_shape[0]:
            raise ValueError("block-25 hidden must have shape (B,T,15,20,C)")
        low_resolution = self.proj(self.norm(hidden.float())).squeeze(-1)
        batch, frames, height, width = low_resolution.shape
        return F.interpolate(
            low_resolution.reshape(batch * frames, 1, height, width),
            size=(30, 40), mode="bilinear", align_corners=False,
        ).reshape(batch, frames, 30, 40)

@dataclass(frozen=True)
class V13ModelReport:
    module_names: Sequence[str]
    lora_parameter_count: int
    depth_parameter_count: int
    trainable_names: Sequence[str]
```

Register a scoped block-25 forward hook for training, consume and clear its tensor in the same forward transaction, and never install it in inference. Any failed install rolls back all replaced projections and `requires_grad` flags.

- [ ] **Step 5: Run GREEN and old-LoRA regressions**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_model.py tests/test_wan_lora.py tests/test_wan_stage2.py`

- [ ] **Step 6: Commit**

Commit only the four files with `feat: add v13 late-block scoreboost model`.

---

### Task 4: Implement the four-loss objective and cached dataset

**Files:**
- Create: `src/worldarena_baseline/wan_v13_objective.py`
- Modify: `src/worldarena_baseline/wan_v13_data.py`
- Create: `tests/test_wan_v13_objective.py`
- Modify: `tests/test_wan_v13_data.py`

**Interfaces:**
- Consumes: predicted flow, noisy/clean latent, sigma, Task 2 labels, Task 3 predicted depth.
- Produces: `v13_scoreboost_objective` returning `V13Losses` and `WanV13CachedDataset` batches.

- [ ] **Step 1: Write RED tests for exact math and normalization**

```python
def test_v13_total_is_the_frozen_four_term_contract():
    losses = v13_scoreboost_objective(**fixture)
    expected = losses.fm + .5 * losses.mask + .1 * losses.temporal + .1 * losses.depth
    torch.testing.assert_close(losses.total, expected)

def test_mask_weights_are_normalized_per_sample_over_valid_tokens():
    weights = normalized_region_weights(robot, obj, valid)
    assert torch.allclose(
        (weights * valid).sum((1,2,3,4)) / valid.sum((1,2,3,4)),
        torch.ones(batch),
    )
```

Cover `1+2R+4O`, overlap weight 7, valid-latent normalization, no cross-sample normalization, temporal adjacent-pair union masks, invalid pair exclusion, all-invalid rejection, robust scale/shift depth alignment, finite gradients into LoRA/depth head, and no gradient into cached labels or frozen parent.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_objective.py tests/test_wan_v13_data.py`

- [ ] **Step 3: Implement the objective**

```python
@dataclass(frozen=True)
class V13Losses:
    total: Tensor
    fm: Tensor
    mask: Tensor
    temporal: Tensor
    depth: Tensor
```

Implement `v13_scoreboost_objective` with keyword-only prediction, noisy latent, clean latent, sigma, robot mask, object mask, object validity, target depth, predicted depth and valid-latent arguments. Reconstruct `z0_hat` with the existing tested `predicted_clean_latent()` helper. Use SmoothL1 with explicit unreduced tensors before masks and denominators. Depth loss first solves one detached least-squares scale and shift per sample between prediction and target, then applies SmoothL1 on finite valid pixels.

- [ ] **Step 4: Extend the dataset as a new class, not a v3 mutation**

`WanV13CachedDataset` wraps/reuses v3 latent/context/action loading but requires the Task 2 sidecar/payload hash and adds:

```python
{
    "robot_mask": FloatTensor[21,30,40],
    "object_mask": FloatTensor[21,30,40],
    "object_valid": BoolTensor[21],
    "inverse_depth": FloatTensor[21,30,40],
}
```

Missing/mixed schema, provenance, prompt-map hash, teacher hash, or payload semantics fail during dataset construction, before CUDA.

- [ ] **Step 5: Run GREEN verification and commit**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_objective.py tests/test_wan_v13_data.py tests/test_wan_action_loss.py`

Commit with `feat: add v13 scoreboost objective`.

---

### Task 5: Add the bounded trainer, checkpoint lineage, and GPU6 launcher

**Files:**
- Create: `src/worldarena_baseline/wan_v13_training.py`
- Create: `scripts/train_wan_v13_scoreboost.py`
- Create: `scripts/run_wan_v13_gpu6.sh`
- Create: `tests/test_wan_v13_training.py`
- Create: `tests/test_wan_v13_scripts.py`

**Interfaces:**
- Consumes: Tasks 1–4 artifacts, clean-gated parent, existing cached Wan inputs.
- Produces: preflight receipt, smoke receipt, `exposure-0100.pt`, `exposure-0150.pt`, telemetry JSONL.

- [ ] **Step 1: Write RED tests for optimizer and lineage**

```python
def test_optimizer_has_only_reviewed_groups_and_lrs():
    optimizer = build_v13_optimizer(model, expected_trainables)
    assert [(g["name"], g["lr"]) for g in optimizer.param_groups] == [
        ("late_qvo_lora", 5e-6),
        ("depth_head", 1e-4),
    ]

def test_trainer_rejects_teacher_modules_in_hot_path():
    with pytest.raises(RuntimeError, match="teacher.*hot path"):
        validate_v13_runtime_modules(model_with_sam3)
```

Add tests for exact exposure count, 10-step warmup, checkpoint steps, parent/source/data/cache/model hashes, atomic saves, strict resume, optimizer-state completeness, trainable whitelist, frozen-gradient rejection, nonfinite loss/grad rejection, world-size 1, `CUDA_VISIBLE_DEVICES=6`, and output-root escape.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_training.py tests/test_wan_v13_scripts.py`

- [ ] **Step 3: Implement optimizer, scheduler, lineage, checkpoint**

```python
def build_v13_optimizer(model: nn.Module, expected_names: Iterable[str]) -> AdamW:
    actual = {name: parameter for name, parameter in model.named_parameters()
              if parameter.requires_grad}
    if set(actual) != set(expected_names):
        raise ValueError("v13 trainable inventory differs from whitelist")
    lora = [parameter for name, parameter in actual.items() if ".lora_" in name]
    depth = [parameter for name, parameter in actual.items()
             if name.startswith("depth_head.")]
    if not lora or not depth or len(lora) + len(depth) != len(actual):
        raise ValueError("v13 optimizer families are incomplete")
    return AdamW([
        {"params": lora, "lr": 5e-6, "name": "late_qvo_lora"},
        {"params": depth, "lr": 1e-4, "name": "depth_head"},
    ], betas=(0.9, 0.95), weight_decay=0.01)

@dataclass(frozen=True)
class V13Lineage:
    parent_sha256: str
    source_closure_sha256: str
    data_receipt_sha256: str
    supervision_receipt_sha256: str
    prompt_map_sha256: str
    sam3_sha256: str
    depth_teacher_sha256: str
    config_sha256: str
```

Checkpoint contract includes completed exposure, model report, only LoRA/depth-head state, complete AdamW state, scheduler state, RNG state, lineage, peak-memory telemetry and loss history. Resume requires exact equality for every immutable field.

- [ ] **Step 4: Implement trainer modes**

The entry point supports only:

```text
source-receipt | prepare | preflight | smoke | train100 | train150 | status
```

`prepare` is CPU-only and validates every cache row. `preflight` loads model on GPU6 but does not update. `smoke` runs exactly three full iterations and atomically publishes per-iteration allocated/reserved/step-time and all gradient families. `train100` starts fresh; `train150` resumes only the exact exposure100 checkpoint.

- [ ] **Step 5: Implement physical-GPU6 ownership checks**

Before each GPU phase:

```bash
uuid=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader,nounits)
```

Reject any compute PID for that UUID, memory `>=1024 MiB`, utilization `>=10%`, any visible-device value other than `6`, or any world size other than 1. Do not stop or signal another process.

- [ ] **Step 6: Run GREEN plus production-shaped fake-model integration**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_wan_v13_training.py tests/test_wan_v13_scripts.py
python -m py_compile src/worldarena_baseline/wan_v13_training.py scripts/train_wan_v13_scoreboost.py
bash -n scripts/run_wan_v13_gpu6.sh
scripts/run_wan_v13_gpu6.sh dry-run
git diff --check
```

- [ ] **Step 7: Commit**

Commit with `feat: add bounded v13 scoreboost training`.

---

### Task 6: Add inference, paired RGB8/fast20 gating, and incumbent profiling

**Files:**
- Create: `src/worldarena_baseline/wan_v13_inference.py`
- Create: `src/worldarena_baseline/wan_v13_gate.py`
- Create: `scripts/generate_wan_v13_video.py`
- Create: `scripts/evaluate_wan_v13_gate.py`
- Create: `tests/test_wan_v13_inference.py`
- Create: `tests/test_wan_v13_gate.py`
- Create: `tests/test_wan_v13_eval_scripts.py`

**Interfaces:**
- Consumes: parent/v13 checkpoint, RGB8/fast20 manifests, existing atomic queue and official/proxy evaluators.
- Produces: matched video sets, per-episode metrics, blinded review sheet, and one fail-closed decision JSON.

- [ ] **Step 1: Write RED tests for inference stripping and gate decisions**

```python
def test_inference_loads_lora_but_never_depth_head():
    model = load_v13_for_inference(parent, checkpoint)
    assert not any(isinstance(m, V13DepthHead) for m in model.modules())

def test_rgb8_requires_integrity_improvement_and_no_regression():
    result = evaluate_rgb8_gate(parent_rows, candidate_rows)
    assert result.passed
    assert result.next_phase == "fast20"
```

Add boundary tests for exactly 2% failure-aware trajectory tolerance, equal coverage accepted/lower rejected, any black/broken rejected, disappearance and penetration/deformation nonincrease, at least one strict improvement, episode identity equality, missing episode rejection, and fast20 `>=55%` paired trajectory wins.

- [ ] **Step 2: Confirm RED**

Run: `PYTHONPATH=src pytest -q tests/test_wan_v13_inference.py tests/test_wan_v13_gate.py tests/test_wan_v13_eval_scripts.py`

- [ ] **Step 3: Implement LoRA-only inference loading**

Validate parent SHA and v13 lineage, install the same blocks/targets, load exact LoRA tensors, and ignore depth-head tensors only after proving they are the complete reviewed depth-head key set. The inference state must contain no trainable parameter and no depth/SAM3 teacher.

- [ ] **Step 4: Implement matched video generation**

Use `AtomicJobQueue`; each job binds episode, model identity, checkpoint SHA, seed, prompt, `81x480x640`, 50 denoise steps, action scale, CFG, and output hash. Parent and candidate receive byte-identical condition inputs. Existing completed videos are reused only when their sidecars validate exactly.

- [ ] **Step 5: Implement incumbent dev-clean50 profile schedule**

Before v13 evaluation, enqueue missing `clean-gated-step10` dev-clean50 videos and invoke the already-vendored official base, VLM, JEPA and trajectory runners. Store the full profile under:

```text
/data/di/worldarena2_track1_20260815/runs/v13-scoreboost/eval/incumbent-dev-clean50
```

Label it development-only; do not touch or materialize official test-1000.

- [ ] **Step 6: Implement RGB8 and fast20 gate output**

The evaluator writes failure-aware all-episode metrics, detected-only diagnostics, paired wins/ties/losses, coverage, black/broken counts, object track gaps, penetration/deformation counts, blinded review rows, input hashes and a single `passed/next_phase/reasons` decision. Missing human adjudication makes RGB8 pending, never passing.

- [ ] **Step 7: Run GREEN and commit**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_wan_v13_inference.py tests/test_wan_v13_gate.py tests/test_wan_v13_eval_scripts.py tests/test_atomic_job_queue.py tests/test_official_track1_eval.py
git diff --check
```

Commit with `feat: add v13 matched evaluation gates`.

---

### Task 7: Bind the runtime closure and execute the bounded remote experiment

**Files:**
- Create: `src/worldarena_baseline/wan_v13_sync_closure.py`
- Create: `scripts/validate_wan_v13_sync_closure.py`
- Create: `tests/test_wan_v13_sync_closure.py`
- Create: `reports/2026-08-19-v13-scoreboost-experiment-report.md`
- Modify: `reports/2026-08-19-wan-action-v1-v11-architecture-experiment-lineage.md`

**Interfaces:**
- Consumes: all Task 1–6 runtime entry points and their recursive local imports.
- Produces: reviewed source receipt, remote test/smoke/train/eval evidence, and final experiment decision.

- [ ] **Step 1: Write RED closure tests**

Require recursive static local-import discovery from every v13 entry point, exact manifest equality, no symlink/untracked/staged/modified runtime member, actual-byte SHA256, and rejection of unresolved dynamic local imports.

- [ ] **Step 2: Implement and verify the closure**

Run:

```bash
PYTHONPATH=src pytest -q tests/test_wan_v13_sync_closure.py
PYTHONPATH=src python scripts/validate_wan_v13_sync_closure.py --source-root . --write-receipt /tmp/v13-source-receipt.json
git diff --check
```

- [ ] **Step 3: Run the complete local focused suite**

Run:

```bash
PYTHONPATH=src pytest -q \
  tests/test_wan_v13_data.py \
  tests/test_wan_v13_labels.py \
  tests/test_wan_v13_cache_scripts.py \
  tests/test_wan_v13_model.py \
  tests/test_wan_v13_objective.py \
  tests/test_wan_v13_training.py \
  tests/test_wan_v13_scripts.py \
  tests/test_wan_v13_inference.py \
  tests/test_wan_v13_gate.py \
  tests/test_wan_v13_eval_scripts.py \
  tests/test_wan_v13_sync_closure.py
```

Expected: zero failures and zero unexpected skips.

- [ ] **Step 4: Sync only the validated v13 closure**

After the user-authorized remote boundary, copy the exact receipt-listed files to `/home/huazhi/nlh/baseline`, then rebuild the receipt remotely and require byte equality. Do not copy the whole dirty worktree.

- [ ] **Step 5: Run remote Torch tests before any GPU work**

Use:

```bash
/data/di/worldarena2_track1_20260815/venv_reuse/bin/python -m pytest -q \
  tests/test_wan_v13_data.py tests/test_wan_v13_labels.py \
  tests/test_wan_v13_cache_scripts.py tests/test_wan_v13_model.py \
  tests/test_wan_v13_objective.py tests/test_wan_v13_training.py \
  tests/test_wan_v13_scripts.py tests/test_wan_v13_inference.py \
  tests/test_wan_v13_gate.py tests/test_wan_v13_eval_scripts.py \
  tests/test_wan_v13_sync_closure.py
```

Expected: zero failures and zero Torch-dependent skips.

- [ ] **Step 6: Build data, download/pin DAV2-Small, and create offline labels**

Run in order: data split, guarded teacher download, cache production, full cache validation. Record elapsed time, rejected/refilled samples, disk before/after, model hashes and final counts. No trainer starts until exactly 150 train and 8 RGB gate caches validate.

- [ ] **Step 7: Run GPU6 preflight and smoke**

Run `source-receipt`, `prepare`, `preflight`, then `smoke`. Require all three iterations under 22 GiB allocated and reserved, nonzero finite Q/V/O LoRA and depth-head gradients, and no forbidden module in memory. If any condition fails, stop without formal training.

- [ ] **Step 8: Train to 100 and then 150 exposures**

Run `train100`; validate its checkpoint and health telemetry. Unless a hard health failure exists, resume exact lineage with `train150`. Never relaunch from scratch over an existing valid checkpoint.

- [ ] **Step 9: Execute evaluation decision tree**

Complete the incumbent dev-clean50 profile, then matched RGB8. If RGB8 fails, terminate v13. If it passes, run matched fast20. Only after fast20 passes, perform the sequential action-scale then text-CFG search; do not run the Cartesian product.

- [ ] **Step 10: Write the final report and lineage entry**

The report must include asset hashes, data leakage receipt, task distribution, rejected/refilled cache rows, cache coverage, disk usage, GPU smoke per iteration, trainable count/names, LR and losses by exposure, checkpoint hashes, RGB8 per-episode table, fast20 per-episode table if run, total metric comparison, gate reasons, terminal decision and exact reproduction commands. Update the architecture lineage with v13's hypothesis, implementation, measured result, and retained checkpoint.

- [ ] **Step 11: Final verification and commit**

Run the complete focused suite again, `py_compile`, `bash -n`, recursive closure validation, `git diff --check`, and verify report paths/hashes exist. Commit with `docs: record v13 scoreboost experiment` only after evidence is complete.

---

## Execution Stop Conditions

Stop immediately and preserve the last atomic artifact when any of these occurs:

- leakage or lineage mismatch;
- invalid/missing mask or depth cache;
- GPU6 ownership conflict;
- allocated or reserved memory reaches 22 GiB;
- NaN/Inf/OOM or nonzero gradient outside the whitelist;
- exposure100 health failure;
- RGB8 gate failure;
- fast20 gate failure.

None of these failures authorizes adding another architecture module inside v13.
