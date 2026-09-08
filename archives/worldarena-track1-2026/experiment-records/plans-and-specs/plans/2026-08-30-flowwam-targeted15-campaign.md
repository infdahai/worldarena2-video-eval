# FlowWAM Targeted 15-Metric Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a fail-closed, eight-GPU asynchronous FlowWAM campaign whose seven targeted mechanisms are trained on released RoboTwin data and promoted only by the corrected 15-metric protocol.

**Architecture:** A contract-v2 layer freezes source data, split, proposals, budgets, selector mode, and all hashes. A common targeted FlowWAM forward exposes per-token RGB flow-matching state; small mechanism modules add visual, instruction/contact, flow-warp, or post-RGB losses. UUID workers claim independent smoke/train/generate/score work, while an evaluation controller enforces generated9-first seed selection, GT caps, bootstrap gates, and immutable P0 fallback.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Hub, HDF5, pytest, existing WorldArena official scorers, `flock`, `nvidia-smi`.

**Spec:** `docs/superpowers/specs/2026-08-30-flowwam-targeted-15metric-campaign-design.md`

## Global Constraints

- Parent checkpoint SHA is `e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4`.
- Dataset is `YixiangChen/FlowWAM_RoboTwin` revision `506c4e014f7dbd291e7d7683c79fc685dd3e2714`; the old `FlowWAM_WorldArena` trainer contract is never loosened.
- Proposal values are ASCII strings, booleans, null, integers, lists, or objects; decimal coefficients are strings, so the restricted domain has deterministic RFC-8785 bytes without binary-float ambiguity.
- Round one uses frozen existing selector model SHA `4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa` and policy SHA `aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7`.
- Dynamic Degree, Flow Score, and Motion Smoothness use per-episode `min(candidate,matched_GT)`; no full15 value chooses seed1/seed4.
- Seven smokes are non-promoting and discarded; formal training restarts from parent and ends at steps 5/10/15/20/25.
- Hard budgets are 1,600 new raw videos, 800 full15 evaluations with 256 reserved for final holdout, and 384 GPU-hours.
- Never stop, modify, or attach to `/data/fjy`, `/data/whn`, or another user's process.

---

### Task 1: Canonical Proposal and Budget Contracts

**Files:**
- Create: `src/worldarena_baseline/targeted15_proposal.py`
- Create: `tests/test_targeted15_proposal.py`

**Interfaces:**
- Produces: `canonical_proposal_bytes(payload: Mapping[str, object]) -> bytes`
- Produces: `proposal_sha256(payload: Mapping[str, object]) -> str`
- Produces: `validate_proposal(payload: Mapping[str, object]) -> Mapping[str, object]`
- Produces: `BudgetLedger.reserve(proposal_id: str, videos: int, full15: int, gpu_hours_milli: int) -> None`

- [ ] **Step 1: Write failing tests for restricted canonicalization and null-preimage hashing**

```python
def test_proposal_hash_replaces_only_self_pin():
    proposal = valid_proposal()
    proposal["pins"]["proposal_sha256"] = "0" * 64
    digest = proposal_sha256(proposal)
    proposal["pins"]["proposal_sha256"] = digest
    assert proposal_sha256(proposal) == digest

def test_proposal_rejects_float_and_implicit_selector():
    proposal = valid_proposal()
    proposal["mechanism"]["lr"] = 1e-6
    with pytest.raises(Targeted15ContractError, match="float"):
        validate_proposal(proposal)
    del valid_proposal()["selector_mode"]
```

- [ ] **Step 2: Run the tests and verify red**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_targeted15_proposal.py -q`

Expected: import failure for `worldarena_baseline.targeted15_proposal`.

- [ ] **Step 3: Implement exact schema, sentinel, SHA, and atomic budget reservations**

```python
SELECTOR_SENTINEL = {
    "contract": "selector-fit-manifest/1",
    "mode": "not_applicable",
    "reason": "frozen_existing",
    "rows": [],
}

def proposal_sha256(payload):
    preimage = copy.deepcopy(payload)
    preimage["pins"]["proposal_sha256"] = None
    return hashlib.sha256(canonical_proposal_bytes(preimage)).hexdigest()
```

Reject floats, non-ASCII strings, missing exact top-level fields, unpinned manifests, selector defaults, budget overcommit, and a registered digest unequal to the null-preimage digest. Write the ledger under an exclusive queue-level flock using temp-file plus `os.replace`.

- [ ] **Step 4: Run proposal tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_targeted15_proposal.py -q`

Expected: all pass.

- [ ] **Step 5: Commit only Task 1 files**

```bash
git add src/worldarena_baseline/targeted15_proposal.py tests/test_targeted15_proposal.py
git commit -m "feat: freeze targeted15 proposal contracts"
```

### Task 2: RoboTwin Contract-v2 Split and Smoke Manifest

**Files:**
- Create: `src/worldarena_baseline/flowwam_randomized500_v2.py`
- Create: `scripts/build_flowwam_targeted15_manifests.py`
- Create: `tests/test_flowwam_randomized500_v2.py`

**Interfaces:**
- Consumes: `randomized500_contract.Randomized500Plan`
- Produces: `build_split_manifests(extracted_root: Path, pins: Mapping[str, str]) -> SplitArtifacts`
- Produces: `build_smoke_manifest(task: str, episode_index: int) -> Mapping[str, object]`

- [ ] **Step 1: Write failing tests for immutable source and four-way disjointness**

```python
def test_v2_source_and_split_are_exact(tmp_path):
    artifacts = build_split_manifests(fake_first12(tmp_path), PINS)
    assert artifacts.source_repo == "YixiangChen/FlowWAM_RoboTwin"
    assert artifacts.source_revision == "506c4e014f7dbd291e7d7683c79fc685dd3e2714"
    identities = [set(a.identities) for a in artifacts.all_splits]
    assert all(not a & b for i, a in enumerate(identities) for b in identities[i+1:])
```

- [ ] **Step 2: Verify the test fails**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_randomized500_v2.py -q`

- [ ] **Step 3: Implement the v2 source lock and exact splits**

Model-train uses the first six frozen tasks at episodes 0-399; selector-fit uses the same tasks at 400-499; dev uses `press_stapler` and `turn_switch`; final uses `adjust_bottle`, `click_alarmclock`, `place_object_stand`, and `stamp_seal`. Use the spec salts for deterministic dev/final order. Every row carries task, episode, RGB/flow/action/instruction relative paths and SHA256 values. Smoke is exactly `click_bell/episode0` and cannot be used as a formal checkpoint.

- [ ] **Step 4: Run data-contract tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_randomized500_v2.py tests/test_randomized500_contract.py -q`

- [ ] **Step 5: Commit only Task 2 files**

```bash
git add src/worldarena_baseline/flowwam_randomized500_v2.py scripts/build_flowwam_targeted15_manifests.py tests/test_flowwam_randomized500_v2.py
git commit -m "feat: add RoboTwin targeted15 split contract"
```

### Task 3: Common Targeted Forward and Loss Primitives

**Files:**
- Create: `src/worldarena_baseline/flowwam_targeted_forward.py`
- Create: `src/worldarena_baseline/flowwam_targeted_losses.py`
- Create: `tests/test_flowwam_targeted_losses.py`

**Interfaces:**
- Produces: `TargetedForwardOutput(rgb_z, rgb_noisy, rgb_pred, rgb_target, timestep, sigma, pred_x0, per_token_fm)`
- Produces: `sample_targeted_sigma(generator, scheduler) -> SigmaSample`
- Produces: `build_contact_weights(rgb, raw_flow, latent_shape) -> Tensor`
- Produces: `compose_four_rgb_flows(flow: Tensor) -> Tensor`
- Produces: `visual_loss`, `instruction_loss`, `contact_fm_loss`, and `flow_warp_loss`

- [ ] **Step 1: Write red tests for sampler, masks, flow composition, and loss gradients**

```python
def test_four_unit_flows_compose_to_four_pixels():
    flow = torch.zeros(4, 2, 8, 8); flow[:, 0] = 1
    composed, valid = compose_four_rgb_flows(flow)
    assert torch.allclose(composed[valid][:, 0], torch.tensor(4.0))

def test_contact_weights_map_121_rgb_to_31_latents():
    weights = build_contact_weights(rgb121(), flow120(), (31, 60, 80))
    assert weights.shape == (31, 60, 80)
    assert weights.mean().item() == pytest.approx(1.0, abs=1e-6)
```

- [ ] **Step 2: Verify the tests fail**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_targeted_losses.py -q`

- [ ] **Step 3: Implement the exposed forward and exact Section 5 math**

Reuse official Stage-1 encoding and dual-stream inference, but accept externally supplied noise/timestep so correct and wrong prompts share them. Implement the 75/25 sigma sampler, predicted-x0, per-token FM, fixed Laplacian, final-block DINO patch cosine, I2 source-aligned masks, four-flow composition, and C2 warp. Add one-pixel and four-step composition assertions with maximum error below `1e-4`.

- [ ] **Step 4: Run focused tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_targeted_losses.py -q`

- [ ] **Step 5: Commit only Task 3 files**

```bash
git add src/worldarena_baseline/flowwam_targeted_forward.py src/worldarena_baseline/flowwam_targeted_losses.py tests/test_flowwam_targeted_losses.py
git commit -m "feat: add FlowWAM targeted loss primitives"
```

### Task 4: Seven Mechanism Trainer and Smoke Receipts

**Files:**
- Create: `scripts/train_flowwam_targeted15.py`
- Create: `src/worldarena_baseline/flowwam_targeted_training.py`
- Create: `tests/test_flowwam_targeted_training.py`

**Interfaces:**
- Consumes: validated proposal v2, split manifest, parent SHA, optional DINO/cache SHAs
- Produces: `run_smoke(proposal, sample, output_root) -> SmokeReceipt`
- Produces: `run_quantum(proposal, start_receipt, target_step) -> CheckpointReceipt`

- [ ] **Step 1: Write red tests for whitelists, smoke discard, and formal endpoints**

```python
@pytest.mark.parametrize("family", ["V1","V2","I1","I2","C1","C2"])
def test_only_declared_parameters_are_trainable(family):
    model = tiny_flowwam()
    apply_proposal(model, proposal(family))
    assert trainable_names(model) == set(proposal(family)["mechanism"]["module_whitelist"])

def test_smoke_checkpoint_cannot_resume_formal_training(tmp_path):
    receipt = fake_smoke_receipt(tmp_path)
    with pytest.raises(Targeted15TrainingError, match="non-promoting"):
        formal_resume(receipt)
```

- [ ] **Step 2: Verify red**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_targeted_training.py -q`

- [ ] **Step 3: Implement V1/I1/I2/C1/C2/V2 dispatch and receipts**

The CLI accepts only proposal path, manifest path, parent path, output root, and `--mode smoke|train-quantum`. It derives family, LR, rank, coefficients, target step, and trainable modules from the signed proposal. Smoke writes peak allocated/reserved memory, gradient norms, loss terms, one deterministic inference SHA, structure results, and `promoting=false`. Formal steps are exactly 5/10/15/20/25 and restart from the parent after smoke.

- [ ] **Step 4: Run trainer contract tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_targeted_training.py tests/test_flowwam_targeted_losses.py -q`

- [ ] **Step 5: Commit only Task 4 files**

```bash
git add scripts/train_flowwam_targeted15.py src/worldarena_baseline/flowwam_targeted_training.py tests/test_flowwam_targeted_training.py
git commit -m "feat: add targeted15 FlowWAM trainer"
```

### Task 5: Lightweight R1 Refiner

**Files:**
- Create: `src/worldarena_baseline/flowwam_temporal_refiner.py`
- Create: `scripts/train_flowwam_temporal_refiner.py`
- Create: `tests/test_flowwam_temporal_refiner.py`

**Interfaces:**
- Produces: `TemporalResidualRefiner.forward(stage1_rgb, robot_contact_mask) -> refined_rgb`
- Produces: `refiner_loss(refined_rgb, target_rgb, residual) -> Mapping[str, Tensor]`

- [ ] **Step 1: Write red tests for topology and invariants**

```python
def test_refiner_preserves_frame_zero_and_contract():
    refined = model(stage1, mask)
    assert torch.equal(refined[:, :, 0], stage1[:, :, 0])
    assert refined.shape == stage1.shape
    assert refined.min() >= 0 and refined.max() <= 1
```

- [ ] **Step 2: Verify red**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_temporal_refiner.py -q`

- [ ] **Step 3: Implement exact Conv3d residual topology and I2-mask reuse**

Use the spec's 3→128 input convolution, eight depthwise/pointwise blocks, 128→3 output, reflection windowing, clipped residual, action attenuation, refined-RGB SmoothL1/Laplacian, and residual temporal penalty. Require the I2 mask-builder SHA in the proposal and receipt.

- [ ] **Step 4: Run tests and commit scoped files**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_temporal_refiner.py -q`

```bash
git add src/worldarena_baseline/flowwam_temporal_refiner.py scripts/train_flowwam_temporal_refiner.py tests/test_flowwam_temporal_refiner.py
git commit -m "feat: add FlowWAM temporal residual refiner"
```

### Task 6: Generated9-First Full15 Gate Controller

**Files:**
- Create: `src/worldarena_baseline/targeted15_gate.py`
- Create: `scripts/evaluate_flowwam_targeted15_gate.py`
- Create: `tests/test_targeted15_gate.py`

**Interfaces:**
- Produces: `select_seed(rows, selector_model, policy) -> SelectedVideo`
- Produces: `correct_motion(candidate15, matched_gt15) -> Mapping[str, float]`
- Produces: `aggregate_gate(candidate_rows, p0_rows, level) -> GateReceipt`

- [ ] **Step 1: Write red tests proving full15 cannot influence seed selection**

```python
def test_full15_oracle_cannot_change_selection():
    rows = dual_seed_rows(seed1_full15=0.1, seed4_full15=0.9)
    chosen = select_seed(rows, frozen_selector(), frozen_policy())
    del rows[chosen.seed]["full15"]
    assert select_seed(rows, frozen_selector(), frozen_policy()) == chosen
```

- [ ] **Step 2: Add GT-cap, bootstrap, reuse, and budget tests**

Assert three capped metrics use per-row min, P0 rows are reused by SHA, n24/n64/n128 are cumulative, final reserve cannot be borrowed, and the final holdout can open once.

- [ ] **Step 3: Implement controller and exact gate receipts**

Use 10,000 task-stratified paired-bootstrap replicates and the proposal-derived RNG seed. Store all 15 point deltas/LCBs, nonmotion12, corrected15/EWMScore_P, cap hit/headroom, strict wins, target/protection aggregates, and the sole terminal decision.

- [ ] **Step 4: Run tests and commit**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_targeted15_gate.py -q`

```bash
git add src/worldarena_baseline/targeted15_gate.py scripts/evaluate_flowwam_targeted15_gate.py tests/test_targeted15_gate.py
git commit -m "feat: enforce generated9-first 15-metric gates"
```

### Task 7: UUID-Level Asynchronous Campaign Queue

**Files:**
- Modify: `src/worldarena_baseline/flowwam_async_orchestrator.py`
- Modify: `scripts/run_flowwam_async_gpu_worker.py`
- Create: `scripts/build_flowwam_targeted15_campaign.py`
- Create: `tests/test_flowwam_targeted15_orchestrator.py`

**Interfaces:**
- Consumes: proposal registry, budget ledger, split/scorer manifests
- Produces: unique work DAG for smoke, 5-step quanta, step25 generation, generated9, full15, gates, and final holdout

- [ ] **Step 1: Write red scheduler tests**

Test that work keys exclude physical UUID, claims include UUID, seven smokes can run concurrently, step25 generation does not wait for peers, score backlog borrows service GPUs, and stale recovery needs two free observations over ten minutes plus exclusive flock.

- [ ] **Step 2: Verify red**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_targeted15_orchestrator.py -q`

- [ ] **Step 3: Implement targeted work types without modifying banned builders**

Add proposal-v2 validation before registration, queue-level scheduler flock, capability-class matching, business-receipt reconciliation, budget reservation, eight-video generation shards, and analysis-triggered `close|promote|combine|new-proposal`. Keep the historical `build_eight_arm_campaign` untouched and unreachable from the new launcher.

- [ ] **Step 4: Run queue suites and commit scoped files**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_flowwam_async_orchestrator.py tests/test_flowwam_targeted15_orchestrator.py -q`

```bash
git add src/worldarena_baseline/flowwam_async_orchestrator.py scripts/run_flowwam_async_gpu_worker.py scripts/build_flowwam_targeted15_campaign.py tests/test_flowwam_targeted15_orchestrator.py
git commit -m "feat: schedule targeted15 campaign asynchronously"
```

### Task 8: Remote Freeze, Smoke Launch, and Autonomous Iteration

**Files:**
- Create: `scripts/launch_flowwam_targeted15_campaign.py`
- Create: `tests/test_launch_flowwam_targeted15_campaign.py`

**Interfaces:**
- Produces: immutable remote deployment manifest and eight worker commands

- [ ] **Step 1: Write red launcher tests**

Assert the launcher refuses missing receipts, mismatched source/parent/scorer/proposal SHAs, active external compute PIDs, duplicate work, or uncooled UUIDs. Assert partial legal GPUs launch independently.

- [ ] **Step 2: Implement dry-run-first launcher**

The launcher writes exact file hashes and commands in dry-run mode, then requires the same deployment-manifest SHA for execute mode. It never uses `pkill`, `killall`, DDP, or broad PID matching.

- [ ] **Step 3: Run all focused tests**

Run: `UV_CACHE_DIR=.uv-cache uv run pytest tests/test_targeted15_proposal.py tests/test_flowwam_randomized500_v2.py tests/test_flowwam_targeted_losses.py tests/test_flowwam_targeted_training.py tests/test_flowwam_temporal_refiner.py tests/test_targeted15_gate.py tests/test_flowwam_targeted15_orchestrator.py tests/test_launch_flowwam_targeted15_campaign.py -q`

- [ ] **Step 4: Deploy only hashed files and run remote dry-run**

Compare local/remote SHA256 for every deployed file. Verify all eight UUIDs, compute PIDs via `/proc`, disk, source/parent/scorer manifests, queue flock, and budget ledger. Register all seven smoke work items; each legal UUID claims independently.

- [ ] **Step 5: Execute and verify initial asynchronous launch**

Require each launched smoke receipt to record UUID, PID/PPID/PGID, user, cwd, command, peak memory, gradients, checkpoint reload, inference SHA, and structure. Failed work closes only its arm; ready work continues. Update the heartbeat with exact queue and receipt paths.

- [ ] **Step 6: Commit launcher files**

```bash
git add scripts/launch_flowwam_targeted15_campaign.py tests/test_launch_flowwam_targeted15_campaign.py
git commit -m "feat: launch targeted15 campaign safely"
```

## Self-Review

- Spec coverage: source/splits, seven mechanisms, selector isolation, corrected15, GT cap, budgets, UUID scheduling, stale recovery, final holdout, and submission boundary each map to a task.
- Placeholder scan: no TBD/TODO/"implement later" steps remain.
- Type consistency: proposal, manifest, receipt, selector, gate, and queue interfaces use the same names throughout.
- Execution order: Tasks 1-3 may run in parallel after Task 1 interface freeze; Tasks 4 and 5 follow Task 3; Task 6 can run beside Tasks 3-5; Task 7 consumes Tasks 1 and 6; Task 8 consumes all prior tasks.
