# Wan-Action-Lite+ v3 Adaptive Compression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed 350+800-step execution contract with the approved production-shape memory gate and adaptive 200+400/600 budget.

**Architecture:** Keep cached inputs and the production video shape unchanged. Strengthen the existing FSDP probe into a three-optimizer-step hard gate, then change the Stage-1 and Stage-2 validators, launchers, checkpoint schedules, and official development schedule as one versioned policy.

**Tech Stack:** Python 3.12, PyTorch 2.8, FSDP, NCCL, Bash, pytest.

**Spec:** `docs/superpowers/specs/2026-08-15-wan-action-lite-v3-adaptive-compression-design.md`

## Global Constraints

- Exactly eight RTX 4090 24GB ranks; per-rank allocated and reserved peaks must each be below 22 GiB.
- Keep 81 frames, 480x640, BF16, micro-batch one, v3 action conditioning, and official evaluator commit `7b3feee108427bee3380064bb5154970ed7468b5`.
- Stage-1/2 may consume cached latents, cached text contexts, and cached action only; loading T5 or VAE modules fails closed.
- Persistent outputs remain below `/data/di/worldarena2_track1_20260815`.
- Use TDD and remote formal-Python verification before resuming the preparation worker.

---

### Task 1: Three-step production memory gate

**Files:**
- Modify: `scripts/probe_wan_action_fsdp.py`
- Modify: `src/worldarena_baseline/wan_probe.py`
- Test: `tests/test_wan_probe.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: the existing production-shape probe inputs and `validate_probe_telemetry`.
- Produces: a smoke report with `optimizer_steps == 3`, per-rank `step_times_seconds`, peak memory, and zero forbidden T5/VAE modules.

- [ ] Add failing tests that reject fewer than three optimizer steps, any rank at or above 22 GiB, mismatched step-time cardinality, and loaded `wan.modules.t5` or `wan.modules.vae2_2`.
- [ ] Run the focused tests and observe failures against the one-backward probe.
- [ ] Add an adapter AdamW optimizer; execute three forward/backward/step/zero-grad iterations and collect synchronized per-step durations.
- [ ] Add the runtime forbidden-module assertion before the first training iteration and bind its result into the report.
- [ ] Run focused tests, Python compilation, and diff checks.

### Task 2: Stage-1 50/125 adaptive branch contract

**Files:**
- Modify: `scripts/train_wan_action_adapter_fsdp.py`
- Modify: `scripts/run_wan_action_stage1_ab.sh`
- Modify: `scripts/verify_stage1_checkpoint.py`
- Modify: `src/worldarena_baseline/stage1_checkpoint.py`
- Test: `tests/test_wan_training_input.py`
- Test: `tests/test_stage1_checkpoint.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: the existing step-50 prefix and paired replay manifest.
- Produces: strict A/B step-125 checkpoints and a launcher whose default aggregate ceiling is 200 steps.

- [ ] Add failing tests proving A/B must start at 50 and end at 125, final filenames are `step-000125.pt`, and the old step-200 default is rejected.
- [ ] Run the focused tests and observe the old 200-step assertions fail.
- [ ] Change request validation, launcher targets, and checkpoint verification to 125 while preserving step-50 fork equivalence and replay provenance.
- [ ] Run focused tests, Bash syntax checks, Python compilation, and diff checks.

### Task 3: Stage-2 400 default with gated 600 extension

**Files:**
- Modify: `src/worldarena_baseline/wan_stage2.py`
- Modify: `scripts/run_wan_action_stage2.sh`
- Modify: `scripts/train_wan_action_adapter_fsdp.py`
- Test: `tests/test_wan_stage2.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: the selected Stage-1 step-125 checkpoint.
- Produces: a default 400-step run and an explicit `STEPS=600` continuation accepted only with a validated step-400 extension decision artifact.

- [ ] Add failing tests for default 400, rejection of 800, and rejection of 600 without a bound decision artifact containing passing trajectory/no-regression/plateau fields.
- [ ] Run focused tests and observe failures against the fixed-800 policy.
- [ ] Implement the 400-step default, strict 400/600 choices, and the artifact-bound extension gate while preserving resume provenance and the 50-step warmup.
- [ ] Run focused tests, Bash syntax checks, Python compilation, and diff checks.

### Task 4: Two-tier official development schedule

**Files:**
- Modify: `src/worldarena_baseline/official_track1_eval.py`
- Modify: `scripts/build_official_track1_eval_schedule.py`
- Test: `tests/test_official_track1_eval.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: Stage-2 checkpoint availability and optional validated step-600 extension.
- Produces: fast trajectory checkpoints 100/200/300/400 and full evaluator checkpoints 200/400, plus 600 only after extension.

- [ ] Add failing tests for the two-tier default schedule, optional step 600, and absence of step 800.
- [ ] Run focused tests and observe failures against the four-full-evaluation schedule.
- [ ] Implement the versioned two-tier schedule without changing test-1000 freeze semantics or official evaluator strictness.
- [ ] Run focused tests, Python compilation, and diff checks.

### Task 5: Remote gate and resumed launch

**Files:**
- Modify only files required by failures found in Tasks 1-4.
- Verify: `/data/di/worldarena2_track1_20260815/gates/wan-action-v3-smoke.json`

**Interfaces:**
- Consumes: reviewed Tasks 1-4 and existing 200/200 action, VAE, and T5 caches.
- Produces: a passing three-step 8-rank smoke and a resumed Stage-1 prefix launcher.

- [ ] Sync the scoped reviewed source files to `/home/huazhi/nlh/baseline`.
- [ ] Run all focused tests with `/data/di/worldarena2_track1_20260815/venv_reuse/bin/python` and require zero unexpected skips.
- [ ] Resume preparation with the pinned trusted-input and URDF hashes.
- [ ] Require the report to prove three optimizer steps, all allocated/reserved peaks below 22 GiB, finite step times, and no T5/VAE module load.
- [ ] Allow Stage-1 to start only after the smoke report validator passes.
