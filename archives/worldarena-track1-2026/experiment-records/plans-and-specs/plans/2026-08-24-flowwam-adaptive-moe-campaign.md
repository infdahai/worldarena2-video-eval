# FlowWAM Adaptive MoE Campaign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and receipt-gate a four-GPU seed1/seed4 adaptive candidate system with Action-Flow Critic, confidence cascade, zero-cost post-processing diagnostics, and one bounded preference-alignment branch while preserving Official Stage-1 / flow20 / seed4 as the immutable rollback champion.

**Architecture:** A deterministic 160/40 zero-overlap manifest is materialized into four 50-episode paired shards. GPU0/GPU4/GPU5/GPU6 each generate adjacent seed1/seed4 candidates for one shard; the pinned raw evaluator and latest GT-capped scorer label them. Deployment-available action, flow, SAM3, VLM, and image features feed Logistic Regression input/post-generation selectors and a frozen confidence cascade. Independent zero-cost Flow oracle and background-projection gates run from existing artifacts; training branches unlock only after high-confidence training winners exist.

**Tech Stack:** Python 3.10, NumPy, h5py, OpenCV, scikit-learn, PyTorch/RAFT, SAM3, pinned WorldArena evaluator, Bash/flock, four RTX 4090 GPUs.

**Spec:** `docs/superpowers/specs/2026-08-24-flowwam-adaptive-candidate-system-design.md`

## Global Constraints

- Immutable rollback champion: Official FlowWAM Stage-1, flow20, seed4, raw15 mean `0.6586597567502485`.
- Current formal champion after the one-time clean50 gate: frozen P0 post-generation selector, corrected/raw means `0.6592060841461832`/`0.662720295147882`, bound by selector clean50 receipt SHA256 `1d72aa6b1a6f260c963756309f3c74a57504381b929663b9361a8201248b187e`.
- Selector universe: exactly 200 of the existing 1,785 leakage-clean rows; deterministic 160 train / 40 holdout.
- Candidate seeds: exactly seed1 and seed4; 121 frames; 640x480; black0.
- Current free GPUs: physical 0,4,5,6 only after a fresh PID/UUID/lock check; GPU1/2/3/7 remain forbidden while occupied.
- Four shards: 50 episodes and 100 videos per GPU; each episode's seed1/seed4 run on the same GPU and same runner in adjacent order.
- Holdout never participates in feature fitting, thresholds, templates, or training.
- Accuracy is diagnostic; corrected uplift and task-stratified paired bootstrap are the selector gates.
- At most one new clean50 candidate across all new branches.
- SeedVR2 SP2 is terminally closed by `seedvr2-official-sp2-stopped.complete.json`; no retry.
- All state is atomic and receipt-gated under `/data/di/worldarena2_track1_20260815/runs/flowwam-official/adaptive-candidate-system/`.

---

### Task 1: Deterministic selector manifest and four-shard materializer

**Files:**
- Create: `src/worldarena_baseline/adaptive_manifest.py`
- Create: `scripts/build_flowwam_adaptive_manifest.py`
- Create: `tests/test_adaptive_manifest.py`

**Interfaces:**
- Consumes: v16 clean-1785 JSONL, dev-fast20 IDs, dev-clean50 IDs, source FlowWAM/RoboTwin files.
- Produces: `build_selector_manifest(rows, excluded_ids, total=200, holdout=40) -> list[AdaptiveEpisode]`; `materialize_selector_shards(...) -> dict[int, list[str]]`; manifest/shard SHA receipts.

- [ ] **Step 1: Write failing tests for Hamilton allocation, stable salts, zero overlap, 160/40 split, and 50 episodes per GPU**

```python
def test_selector_manifest_is_deterministic_and_four_way_balanced():
    rows = make_rows({"task_a": 120, "task_b": 80, "task_c": 40})
    result = build_selector_manifest(rows, excluded_ids=set(), total=200, holdout=40)
    assert len(result) == 200
    assert sum(row.split == "holdout" for row in result) == 40
    assert Counter(row.physical_gpu for row in result) == {0: 50, 4: 50, 5: 50, 6: 50}
    assert result == build_selector_manifest(rows, excluded_ids=set(), total=200, holdout=40)
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `pytest -q tests/test_adaptive_manifest.py`
Expected: import failure for `worldarena_baseline.adaptive_manifest`.

- [ ] **Step 3: Implement immutable dataclasses, Hamilton apportionment, salted ordering, exclusion checks, and atomic receipts**

The manifest row must contain sample/task/source hashes/split/GPU/shard order and literal salts. Materialization must use symlinks inside lineage-local shard roots and reject any existing non-matching path.

- [ ] **Step 4: Run focused tests and CLI dry-run**

Run: `pytest -q tests/test_adaptive_manifest.py`
Run: `python scripts/build_flowwam_adaptive_manifest.py --help`
Expected: PASS and exit0.

### Task 2: Four-GPU adjacent paired candidate launcher

**Files:**
- Create: `scripts/run_flowwam_adaptive_shard.py`
- Create: `scripts/run_flowwam_adaptive_four_gpu.sh`
- Create: `tests/test_adaptive_launcher.py`
- Modify: `scripts/run_flowwam_official_stage1.py`

**Interfaces:**
- Consumes: Task1 shard manifest and materialized input/robot-only roots.
- Produces: per-shard seed1/seed4 outputs, per-video validation sidecars, `candidates/seed{1,4}/shard-<gpu>.complete.json`, aggregate generation receipts.

- [ ] **Step 1: Write failing launcher contract tests**

```python
def test_four_gpu_plan_pairs_seeds_per_episode():
    plan = build_paired_generation_plan(samples=[f"e{i}" for i in range(200)])
    assert {gpu: len(rows) for gpu, rows in plan.items()} == {0: 50, 4: 50, 5: 50, 6: 50}
    assert all([job.seed for job in rows[:2]] == [1, 4] for rows in plan.values())
```

- [ ] **Step 2: Verify RED**

Run: `pytest -q tests/test_adaptive_launcher.py`
Expected: missing planner/launcher failure.

- [ ] **Step 3: Add exact-sample filtering to Stage-1 and implement one-runner adjacent seed execution**

`run_flowwam_official_stage1.py` receives `--sample-manifest`; it rejects count/order mismatch and includes its SHA in the receipt. The shard wrapper acquires the physical UUID-specific lock, rejects foreign compute PIDs, runs seed1 then seed4 per sample without remapping, and resumes only missing/hash-invalid rows.

- [ ] **Step 4: Run focused tests and four dry-runs**

Run: `pytest -q tests/test_adaptive_launcher.py tests/test_flowwam_official_pipeline.py`
Expected: PASS; each dry-run reports 50 episodes, 100 videos, correct UUID, and `starts_gpu=false`.

- [ ] **Step 5: Sync scoped files and launch the four remote shards**

Before launch, recheck GPU0/4/5/6 compute PIDs and locks. Require the remote manifest receipt. Start one guarded background launcher per GPU; record launcher PID/log but do not claim completion until all four shard receipts and 400 validated videos exist.

### Task 3: Latest-WA2 GT reference and paired corpus scoring

**Files:**
- Create: `src/worldarena_baseline/adaptive_scores.py`
- Create: `scripts/score_flowwam_adaptive_candidates.py`
- Create: `tests/test_adaptive_scores.py`
- Modify: `src/worldarena_baseline/latest_wa2_scorer.py`

**Interfaces:**
- Consumes: 400 generated videos, pinned raw metric CSVs, GT motion results receipt.
- Produces: one row per episode with seed1/seed4 raw/corrected/non-motion means, signed margins, winner labels, and `scores/latest-wa2.complete.json`.

- [ ] **Step 1: Write failing pairing and mismatch tests**

```python
def test_pair_scores_fail_when_candidate_or_gt_is_missing():
    with pytest.raises(ValueError, match="complete seed1/seed4 pair"):
        pair_candidate_scores(seed1=[row("e1")], seed4=[], expected=1)
```

- [ ] **Step 2: Verify RED, implement pairing/finite/hash validation, then verify GREEN**

Run: `pytest -q tests/test_adaptive_scores.py tests/test_latest_wa2_scorer.py`
Expected: PASS after implementation.

- [ ] **Step 3: Produce the remote GT-motion receipt before candidate scoring**

Run the pinned GT evaluator once; require 200 matched episode caps for all three motion columns and a receipt hash. Candidate raw scoring may run in four independent evaluator lineages; final pairing waits for all receipts.

### Task 4: Deployment-available action and Action-Flow Critic features

**Files:**
- Create: `src/worldarena_baseline/adaptive_features.py`
- Create: `scripts/extract_flowwam_adaptive_features.py`
- Create: `tests/test_adaptive_features.py`

**Interfaces:**
- Consumes: request action/calibration, renderer flow/masks, generated RAFT, SAM3 trajectories/object tracks, VLM/Base values.
- Produces: fixed input/post-generation schemas, validity flags, `selectors/action-flow-critic.complete.json`.

- [ ] **Step 1: Write failing tests for cosine/EPE/direction/magnitude/phase/leakage and global schema removal**

```python
def test_unavailable_deployment_feature_is_removed_globally():
    schema = freeze_feature_schema(train_rows_with_one_missing_calibration())
    assert "action_projection_DTW" not in schema.names
```

- [ ] **Step 2: Verify RED, implement pure NumPy feature functions, then verify GREEN**

Run: `pytest -q tests/test_adaptive_features.py`
Expected: PASS without loading a GPU model.

- [ ] **Step 3: Add pinned GPU extractors and receipt checks**

RAFT/SAM3 workers consume disjoint candidate shards. A row is accepted only when desired/generated flow geometry, arm identity, timestamps, and video SHA match the manifest.

### Task 5: Logistic selectors, cascade, bootstrap, and final holdout gate

**Files:**
- Create: `src/worldarena_baseline/adaptive_selector.py`
- Create: `scripts/fit_flowwam_adaptive_selectors.py`
- Create: `tests/test_adaptive_selector.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: Task3 labels and Task4 frozen feature tables.
- Produces: serialized input/postgen pipelines, frozen `tau_high`/`tau_post`, deterministic holdout report, selection receipt.

- [ ] **Step 1: Add scikit-learn dependency and write failing leakage/cascade/bootstrap tests**

```python
def test_holdout_cannot_change_thresholds():
    fitted = fit_cascade(train_rows, random_state=20260824)
    assert fitted.thresholds == fit_cascade(train_rows, random_state=20260824).thresholds
    assert evaluate_holdout(fitted, holdout_rows).thresholds == fitted.thresholds
```

- [ ] **Step 2: Verify RED, implement exact pipelines/grid/tie-breaks, then verify GREEN**

Run: `pytest -q tests/test_adaptive_selector.py`
Expected: 10,000-replicate bootstrap and serialized-model replay are bitwise deterministic.

- [ ] **Step 3: Fit on160, freeze artifacts, then open40 exactly once**

The selector advances only on corrected uplift `>=0.003`, one-sided 90% lower bound `>0`, raw/non-motion guards, black0, and both seeds selected at least5/40.

### Task 6: Flow oracle and Action-Support Background Projection

**Files:**
- Create: `src/worldarena_baseline/adaptive_postprocess.py`
- Create: `scripts/run_flowwam_zero_cost_diagnostics.py`
- Create: `tests/test_adaptive_postprocess.py`

**Interfaces:**
- Consumes: existing flow16/20/24 Breadth20 receipts; existing seed4 frames, first frame, masks, RAFT, SAM3.
- Produces: `flow-oracle/flow16-20-24.complete.json`; projected videos and `background-projection/quality-gate.complete.json`.

- [ ] **Step 1: Write failing tests for common-episode oracle, fixed blend, feather continuity, and no dynamic-support changes**
- [ ] **Step 2: Verify RED, implement fixed contracts, verify GREEN**

Run: `pytest -q tests/test_adaptive_postprocess.py`

- [ ] **Step 3: Run zero-cost remote diagnostics and close failing branches**

Do not generate flow16/24. Flow oracle needs uplift `>=0.004` and at least6/20 non-flow20 wins. The original Background Projection branch is closed on missing support assets. Its one independently receipt-gated Background-v2 successor uses one fixed mask/blend/denoise/sharpen contract with no grid. It advances only when corrected Breadth20 improves over P0 by at least `0.005`, the frozen five-metric visual mean also improves by at least `0.005`, Trajectory/Instruction/Interaction do not regress, black remains zero, and the seam diagnostic passes.

### Task 7: Action-to-text Breadth20

**Files:**
- Create: `src/worldarena_baseline/adaptive_action_text.py`
- Create: `scripts/run_flowwam_action_text_breadth20.sh`
- Create: `tests/test_adaptive_action_text.py`

**Interfaces:**
- Consumes: action sequence and original instruction.
- Produces: one deterministic suffix, one seed4 Breadth20 lineage, quality receipt.

- [ ] **Step 1: Write failing deterministic/no-hallucinated-object tests**
- [ ] **Step 2: Verify RED, implement one template, verify GREEN**
- [ ] **Step 3: Launch once on a free GPU and apply the frozen gate**

### Task 8: High-confidence Winner-SFT and Preference-LoRA step25

**Files:**
- Create: `src/worldarena_baseline/adaptive_preference.py`
- Create: `scripts/train_flowwam_preference_lora.py`
- Create: `scripts/run_flowwam_adaptive_training.sh`
- Create: `tests/test_adaptive_preference.py`
- Modify: `scripts/train_flowwam_v16_lora.py`

**Interfaces:**
- Consumes: only160 training rows with raw/corrected agreement and margin>=0.003.
- Produces: balanced pseudo-target receipt, Winner-SFT step25 artifacts, Preference-LoRA step25 artifacts, and for each new checkpoint one paired seed1/seed4 Breadth20 gate that selects that checkpoint's own better inference seed.

- [ ] **Step 1: Write failing holdout-exclusion, seed-balance, shared-noise, and loss-formula tests**
- [ ] **Step 2: Verify RED, implement minimal training adapters, verify GREEN**
- [ ] **Step 3: Run production-shape smoke for each branch on separate free GPUs**
- [ ] **Step 4: Train exactly step25, validate hashes, generate only seed1 and seed4 under each new checkpoint on the same Breadth20 inputs, select that checkpoint's better seed, and run the frozen matched comparison**

No training starts if fewer than40 total or fewer than10 per winner seed survive. The paired check is not authorization for seed5+ or any seed grid. Winner-SFT step50, when authorized by its step25 gate, repeats the same seed1/seed4 comparison because fine-tuning may change the preferred inference seed.

After P0 passed clean50, the frozen P0 model/policy must be replayed once on the exact Official seed1/seed4 Breadth20 pair to create `selector/breadth20/p0-selector-breadth20.complete.json`. Every P2 global seed is compared directly to that P0 Breadth20 baseline, never only to Official+seed4. Clean50 eligibility requires corrected mean `>= P0+0.003`, raw mean `>= P0`, 12-metric non-motion `>= P0+0.002`, Trajectory and JEPA each `>= P0-0.005`, at least12/20 strict paired corrected wins, and black0. No episode-wise P2 seed oracle or new selector is allowed. Winner-SFT step50 remains disabled unless step25 first passes this P0 gate, corrected uplift is at least `0.006`, and Instruction, Interaction, and Image each do not regress relative to P0.

### Task 8B: Frozen ExpertPool-v2 closure

The four-expert zero-cost oracle authorizes one conservative three-output system: frozen P0, Winner-SFT step25 seed4 at LoRA scale `0.5`, and the single fixed ActionText expert. P0 remains the default and the expert gate may abstain. Preference-LoRA is oracle evidence only and receives no bulk generation. No new architecture, LoRA, seed, prompt, or expert may be added.

Use the original selector160 training rows plus a new frozen fresh40 that has no overlap with the old holdout or formal test. Once fresh40 is defined, partial results must never change features, thresholds, classifier, templates, or the expert set. Its terminal tiers are frozen before scoring:

- uplift `<0.004`: close ExpertPool;
- `0.004 <= uplift < 0.006`: record the gain but retain P0-only deployment complexity;
- `0.006 <= uplift < 0.008`: eligible for the one clean50 allocation;
- `0.008 <= uplift <= 0.012`: strong main-line evidence;
- uplift `>0.012`: freeze the main line immediately.

Every positive tier additionally requires deterministic one-sided bootstrap lower bound `>0`, no raw or non-motion regression, no structural Trajectory/JEPA regression, finite rows, black0, and exact lineage hashes.

### Research freeze deadline

All model and policy selection ends by `2026-08-26T20:00:00+08:00`. At the deadline every research branch must have a terminal pass/closure/contractual-skip receipt. The final system freezes immediately and starts formal test-1000 that evening. Formal generation is episode-sharded and streamed: each episode generates adjacent Official seed1/seed4 candidates and, only if ExpertPool passed, Winner scale0.5; scoring and selection begin as soon as a shard completes. Formal test results never feed training, thresholds, features, or model selection.

### Task 9: Single clean50 allocation, final decision, and reports

**Files:**
- Create: `scripts/finalize_flowwam_adaptive_campaign.py`
- Create: `tests/test_adaptive_final_gate.py`
- Modify: `reports/2026-08-24-wan-action-v1-v15-total-experiment-report.md`
- Modify: `reports/2026-08-20-v15-flowwam-balanced-campaign.md`
- Modify: `reports/2026-08-23-worldarena-track1-final-score-report.md`

**Interfaces:**
- Consumes: all branch terminal receipts.
- Produces: at most one clean50 evaluation, `final-decision.complete.json`, reconciled reports.

- [ ] **Step 1: Write failing single-allocation and corrected-first gate tests**
- [ ] **Step 2: Implement terminal decision logic and verify tests**
- [ ] **Step 3: Run at most one eligible clean50 candidate**
- [ ] **Step 4: Update all three reports, re-read against receipts, and preserve deployment SHA gate**

Final replacement of the current P0 champion requires corrected mean `> P0 +0.003`, non-motion `>= P0-0.001`, Trajectory/JEPA `>= P0-0.01`, raw15 `>= P0-0.003`, 50 finite rows, black0, and exact lineage hashes. Official+seed4 remains the immutable rollback package.
