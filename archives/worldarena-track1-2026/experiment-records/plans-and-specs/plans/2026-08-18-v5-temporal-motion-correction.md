# Wan-Action-Lite+ v5 Temporal Motion Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and evaluate a zero-initialized per-arm causal temporal motion correction on top of the frozen gated-step10 incumbent within a 24-hour fail-closed experiment.

**Architecture:** The frozen parent preserves the existing support-gated residual. A new shared-weight, independently executed per-arm local causal TCN plus per-arm pooled causal context predicts motion corrections from flow and gripper delta. Arm-specific zero-init projections inject only at Wan blocks 8, 16, and 24.

**Tech Stack:** Python 3.11, PyTorch, Wan2.2 TI2V-5B, FSDP/NCCL, NumPy, pytest.

**Spec:** `docs/superpowers/specs/2026-08-18-v5-temporal-motion-correction-design.md`

## Global Constraints

- Use GPU0-6 only; never inspect, stop, reset, or allocate GPU7.
- Keep the gated-step10 parent, Wan backbone, raster schema, packing, support, and weighted-FM objective unchanged.
- Write persistent remote outputs only below `/data/di/worldarena2_track1_20260815`.
- Do not add LoRA, Pose, PRoPE, V-JEPA, Depth, GAN, trajectory loss, coordination loss, or new data.
- Stop this architecture experiment within 24 hours.

---

### Task 1: Causal temporal motion correction module

**Files:**
- Create: `src/worldarena_baseline/wan_temporal_motion_correction.py`
- Create: `tests/test_wan_temporal_motion_correction.py`

**Interfaces:**
- Consumes: raster `(B,10,81,60,80)`, support `(B,2,21,15,20)`, `action_present`, and target `seq_len`.
- Produces: `TemporalMotionCorrection.residual_components(...) -> dict[int, dict[str, Tensor]]` for blocks 8/16/24 and `router_telemetry`.

- [ ] Write failing tests for causal padding, future-impulse zero, arm separation, BOTH behavior, exact support masking, overlap legality, null action, shape validation, and zero-init outputs.
- [ ] Run `UV_CACHE_DIR=/private/tmp/worldarena-v5-uv-cache uv run pytest tests/test_wan_temporal_motion_correction.py -q` and confirm collection/contract failures.
- [ ] Implement framewise motion encoding, canonical 81-to-21 grouping, two-block depthwise causal TCN, support-weighted per-arm pooling, pooled causal context gate, independent continuous activity gates, and arm-specific zero-init projections.
- [ ] Run the focused tests and `python -m py_compile` until green.
- [ ] Record exact parameter counts and verify no opposite-arm tensor enters either arm forward.

### Task 2: Function-preserving frozen-parent wrapper and checkpoint contract

**Files:**
- Modify: `src/worldarena_baseline/wan_temporal_motion_correction.py`
- Modify: `tests/test_wan_temporal_motion_correction.py`
- Create: `src/worldarena_baseline/wan_v5_checkpoint.py`
- Create: `tests/test_wan_v5_checkpoint.py`

**Interfaces:**
- Consumes: frozen support-gated `WanActionAdapter`, frozen Wan backbone, and `TemporalMotionCorrection`.
- Produces: `ParentPlusTemporalMotionWan` and strict v5 checkpoint save/load validators.

- [ ] Write failing tests proving step-zero `torch.equal`, parent/backbone `requires_grad=False`, exact trainable whitelist, block0 parent-only injection, and rejection of wrong parent SHA/world-size/config.
- [ ] Implement the wrapper that combines parent residuals with zero corrections and registers hooks at the union of parent/v5 injection points.
- [ ] Implement checkpoint metadata for parent SHA256, source/config hashes, world size/rank mapping, optimizer/scheduler state, completed step, and leakage telemetry.
- [ ] Run focused tests, `py_compile`, and `git diff --check`.

### Task 3: Router-level and model-level temporal leakage audit

**Files:**
- Create: `src/worldarena_baseline/temporal_leakage_audit.py`
- Create: `tests/test_temporal_leakage_audit.py`
- Create: `scripts/audit_wan_temporal_leakage_fsdp.py`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Consumes: the existing discovery-8 records, gated-step10 parent, optional v5 checkpoint, and temporally spliced correct/future-perturbed action conditions.
- Produces: router current/future ratio, Wan-output current/future ratio, inactive-arm leakage, spatial routing retention, and a fail-closed JSON decision.

- [ ] Write failing pure tests for temporal splice anchoring, current/future token masks, leakage-ratio calculation, 30% relative-improvement gate, and routing non-regression gate.
- [ ] Implement pure counterfactual and aggregation functions with explicit positive/negative lag semantics.
- [ ] Implement the seven-rank audit runner using identical latent/noise/timestep/context for parent and v5.
- [ ] Add dry-run CLI tests proving GPU7 exclusion, exact paths, pinned parent SHA, and no training/video generation.
- [ ] Run focused tests, `py_compile`, and `git diff --check`.

### Task 4: Seven-rank trainer, smoke, and bounded launcher

**Files:**
- Create: `scripts/train_wan_temporal_motion_v5_fsdp.py`
- Create: `scripts/run_wan_temporal_motion_v5.sh`
- Create: `scripts/start_wan_temporal_motion_v5.sh`
- Modify: `tests/test_scripts.py`
- Modify: `tests/test_wan_training_input.py`

**Interfaces:**
- Consumes: cached latent/context/action v3 rows, frozen gated-step10 parent, replay manifest, and v5 module.
- Produces: checkpoints 5/10/25/50, per-rank production smoke JSON, training log, and atomic status/provenance files.

- [ ] Write failing tests for trainable whitelist, optimizer groups (`projection=1e-4`, `router/context=5e-5`), five-step warmup, exact checkpoint schedule, weighted-FM use, resume semantics, `/data/di` output guard, and `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6`.
- [ ] Implement the trainer by reusing the existing cached-input/FSDP conventions while keeping v4 code and artifacts immutable.
- [ ] Implement the launcher with production three-cycle smoke, per-rank allocated/reserved `<22 GiB`, GPU0-6 ownership checks, hard wall-clock deadline, and checkpoint audit hooks.
- [ ] Run local non-GPU tests, remote formal-Python tests, seven-rank smoke, then start the bounded run only if every hard gate passes.

### Task 5: Checkpoint selection and matched-fast20 promotion

**Files:**
- Create: `scripts/select_wan_temporal_motion_v5.py`
- Create: `tests/test_v5_selection.py`
- Create: `reports/v5-temporal-motion-correction.md`

**Interfaces:**
- Consumes: step10/25/50 audit JSONs and existing matched-fast20 generation/evaluator contracts.
- Produces: zero, one, or at most two selected checkpoint hashes and a final experiment decision.

- [ ] Write failing tests for early stop when router improves but model leakage does not, spatial-routing regression rejection, maximum two candidates, and exact video promotion thresholds.
- [ ] Implement deterministic selection and an atomic experiment report.
- [ ] Run audit at steps 10/25/50; stop as soon as a hard stopping condition is proven.
- [ ] Generate matched-fast20 only for selected candidates using identical seed/prompt/action/CFG/sampling settings.
- [ ] Record DTW, paired wins, coverage, detector failures, black video count, and human bimanual leakage verdict; preserve gated-step10 unless every promotion gate passes.

## Self-Review

- Spec coverage: architecture, function-preserving initialization, dual leakage metrics, training, smoke, selection, and fallback each map to a task.
- Placeholder scan: no TBD/TODO or unspecified implementation step remains.
- Type consistency: the correction module, wrapper, trainer, audit, and selector share the same block set `{8,16,24}` and v3 tensor shapes.

