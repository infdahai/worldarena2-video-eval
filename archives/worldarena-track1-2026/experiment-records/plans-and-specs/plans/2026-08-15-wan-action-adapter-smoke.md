# Wan Action Adapter Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove a zero-init, timestep-gated action adapter against a tiny Wan-compatible backbone before downloading TI2V-5B weights.

**Architecture:** Encode the eight-channel raster directly to the Wan token grid, project through four zero-init residual heads, and integrate through temporary block pre-hooks so upstream Wan code remains untouched.

**Tech Stack:** Python 3.10+, PyTorch 2.10, unittest/pytest

**Spec:** `docs/superpowers/specs/2026-08-15-wan-action-adapter-smoke-design.md`

## Global Constraints

- Do not edit `/home/huazhi/nlh/Wan2.2` or its pinned upstream commit.
- Do not download Wan model weights during tiny-backbone tests.
- Use injection points `0, 8, 16, 24`.
- Require exact output identity at zero initialization.
- Remove temporary hooks even when backbone forward raises.
- Do not start distributed or multi-GPU execution in this plan.

---

### Task 1: Encoder Grid and Validation

**Files:**
- Create: `src/worldarena_baseline/wan_action_adapter.py`
- Create: `tests/test_wan_action_adapter.py`

**Interfaces:**
- Produces: `ActionRasterEncoder.forward(raster) -> Tensor[B,L,adapter_dim]`.

- [ ] Write a unittest using `(B,8,9,32,32)` and assert shape `(B,3,adapter_dim)`.
- [ ] Run it remotely and observe module import failure.
- [ ] Implement the two Conv3d stages, normalization, SiLU, flatten, and input validation.
- [ ] Run the focused remote unittest until green.

### Task 2: Zero-Init Residual Adapter

**Files:**
- Modify: `src/worldarena_baseline/wan_action_adapter.py`
- Modify: `tests/test_wan_action_adapter.py`

**Interfaces:**
- Produces: `WanActionAdapter.residuals(raster, timestep, seq_len) -> dict[int, Tensor]`.

- [ ] Write a test asserting four residuals with shape `(B,seq_len,wan_dim)` are exactly zero at initialization.
- [ ] Implement four independent zero-init linear projections and the normalized-timestep gate MLP.
- [ ] Add a test that rejects action tokens longer than `seq_len` and pads shorter tokens with zeros.
- [ ] Run the focused remote unittest until green.

### Task 3: Unmodified Backbone Wrapper

**Files:**
- Modify: `src/worldarena_baseline/wan_action_adapter.py`
- Modify: `tests/test_wan_action_adapter.py`

**Interfaces:**
- Produces: `ActionConditionedWan`, which preserves the official Wan forward signature and adds keyword-only `action_raster`.

- [ ] Build a 30-block identity backbone and write a failing test that wrapped output equals bare output at zero init.
- [ ] Implement temporary forward-pre-hooks at blocks 0, 8, 16, and 24 with `finally` cleanup.
- [ ] Write a test that makes one projection nonzero and proves output changes.
- [ ] Backpropagate output sum and assert nonzero action-raster and adapter gradients.
- [ ] Make the fake backbone raise, then assert no hook remains installed.
- [ ] Run remote unittests and the local non-torch suite.

### Task 4: Parameter and Real-Grid Report

**Files:**
- Create: `scripts/inspect_wan_action_adapter.py`
- Modify: `tests/test_wan_action_adapter.py`

**Interfaces:**
- Produces: JSON containing token grid, total/trainable parameters, injection points, and zero-init status.

- [ ] Write a test for the report on the tiny adapter with literal expected grid and injection points.
- [ ] Implement the inspection script without loading Wan weights.
- [ ] Run it remotely for `81x480x832`, `adapter_dim=256`, and `wan_dim=3072`.
- [ ] Use the report plus the disk guard to define the exact 5B weight-file download manifest.

