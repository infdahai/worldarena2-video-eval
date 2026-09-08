# Wan-Action-Lite+ v6 Gripper Trajectory Supervision Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch a bounded clean-lineage v6 experiment that directly supervises left/right gripper trajectory in predicted clean latent space.

**Architecture:** A frozen frame-local latent probe supplies differentiable position and velocity losses. A removable per-arm rank-8 delta at Wan blocks 8/16/24 is the only trainable component; a clean support-gated step10 parent is reproduced before v6.

**Tech Stack:** Python 3.10+, PyTorch/FSDP/DDP, pytest, Bash, Wan2.2 TI2V, existing v3 cache and leakage contracts.

**Spec:** `docs/superpowers/specs/2026-08-18-v6-gripper-trajectory-supervision-design.md`

## Global Constraints

- Preserve all unrelated dirty files in the current `feat/oscar-track1-baseline` worktree.
- Write persistent remote artifacts only below `/data/di/worldarena2_track1_20260815`.
- Use GPU0-6 only and never touch GPU7.
- Fail closed on contaminated parent lineage, evaluation leakage, stale hashes, invalid masks, non-finite calibration, or missing independent RGB confirmation.
- Run TDD RED before every production behavior and fresh verification before completion claims.

---

### Task 1: Frame-local probe data and model

**Files:**
- Create: `src/worldarena_baseline/wan_gripper_probe.py`
- Create: `tests/test_wan_gripper_probe.py`
- Create: `scripts/build_wan_gripper_probe_split.py`
- Create: `scripts/train_wan_gripper_probe.py`

**Interfaces:**
- Produces `GripperTrajectoryProbe`, `build_gripper_targets`, `soft_argmax_2d`, `probe_gate`, and an immutable probe checkpoint contract.
- Consumes v3 cached latent/raster rows and existing leakage receipts.

- [ ] Write failing behavioral tests for 48-channel frame-local shape, exact future independence, causal labels, latent0/visibility/velocity masks, arm separation, split disjointness, and heldout gates.
- [ ] Run the focused test and observe failure because the module is absent.
- [ ] Implement only the tested probe, target, split, checkpoint, and CLI contracts.
- [ ] Re-run the focused tests and keep them green.

### Task 2: Low-rank correction and differentiable trajectory objective

**Files:**
- Create: `src/worldarena_baseline/wan_gripper_trajectory_correction.py`
- Create: `src/worldarena_baseline/wan_gripper_trajectory_loss.py`
- Create: `tests/test_wan_gripper_trajectory_correction.py`
- Create: `tests/test_wan_gripper_trajectory_loss.py`

**Interfaces:**
- Consumes frozen `WanActionAdapter.encode_raster_arms` and frozen `GripperTrajectoryProbe`.
- Produces `GripperTrajectoryCorrection`, `ParentPlusGripperTrajectoryWan`, `predicted_clean_latent`, `masked_gripper_trajectory_loss`, sigma reliability, and B-only lambda calibration.

- [ ] Write failing tests for exact zero step0 output, nonzero B/zero A first gradient, independent left/right support, null action, padding, wrapper equality, exact `z0_hat`, invalid velocity edges, sigma masking, B-only norms, and probe freezing with gradient-through-input.
- [ ] Run focused tests and observe failures caused by missing production modules.
- [ ] Implement minimal correction, wrapper, loss, and calibration behavior.
- [ ] Re-run focused tests and keep them green.

### Task 3: Clean parent and v6 experiment contracts

**Files:**
- Create: `src/worldarena_baseline/wan_v6_training.py`
- Create: `src/worldarena_baseline/wan_v6_checkpoint.py`
- Create: `tests/test_wan_v6_training.py`
- Create: `tests/test_wan_v6_checkpoint.py`
- Create: `scripts/build_wan_v6_replay.py`

**Interfaces:**
- Produces the seven-rank replay, fixed 10/25/50/100 schedule, trainable whitelist, immutable calibration contract, and strict parent/checkpoint lineage validators.
- Rejects the historical contaminated gated-step10 checkpoint and any source/dataset/evaluation drift.

- [ ] Write failing tests for exact experiment config, replay bytes, checkpoint resume completeness, parent lineage, clean S1A125 ancestry, probe hash, sigma/lambda calibration, and correction-only trainability.
- [ ] Run focused tests and observe the missing-contract failures.
- [ ] Implement the minimal formal contracts.
- [ ] Re-run focused tests and keep them green.

### Task 4: Training, discovery, and launch pipeline

**Files:**
- Create: `scripts/train_wan_gripper_trajectory_v6_fsdp.py`
- Create: `scripts/calibrate_wan_gripper_trajectory_v6.py`
- Create: `scripts/audit_wan_gripper_trajectory_v6.py`
- Create: `scripts/run_wan_gripper_trajectory_v6.sh`
- Create: `scripts/start_wan_gripper_trajectory_v6.sh`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Consumes clean-gated parent, frozen probe, calibration, replay, clean-1000 cache, discovery manifest, and independent RGB proxy.
- Produces bound smoke/calibration/checkpoints/audits/run status under the formal artifact root.

- [ ] Write failing runtime-contract tests for exact paths, GPU0-6, GPU7 exclusion, no T5/VAE, clean parent preflight, three-step smoke, resume chain, step25 RGB anti-exploitation gate, conditional step50/100, and dry-run no-process behavior.
- [ ] Run the script-focused tests and observe the expected failures.
- [ ] Implement the trainer, calibration, audit, and fail-closed launcher.
- [ ] Run syntax, focused, and broader v3-v6 regression tests.

### Task 5: Remote formal verification and bounded execution

**Files:**
- Create: `reports/v6-gripper-trajectory-supervision-20260818.md`

**Interfaces:**
- Consumes all Task 1-4 outputs.
- Produces remote test evidence, clean parent/probe/calibration/smoke evidence, and the bounded v6 run state.

- [ ] Sync only v6-scoped files plus intentional shared-test edits to `/home/huazhi/nlh/baseline`.
- [ ] Run formal Python focused tests remotely and require zero skipped PyTorch tests.
- [ ] Reproduce and validate clean-gated-step10 without a separate fast20 run.
- [ ] Train/freeze the probe and pass all heldout gates.
- [ ] Run sigma/lambda calibration and seven-GPU production smoke.
- [ ] Launch step10, then gate step25; proceed to step50/100 only under the approved automatic gates.
- [ ] Record exact hashes, metrics, processes, artifacts, failures, and next decision in the report.

