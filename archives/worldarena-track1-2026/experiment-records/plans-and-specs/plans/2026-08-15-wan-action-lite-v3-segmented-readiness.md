# Wan-Action-Lite+ v3 Segmented Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` (recommended) or
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Start formal Stage-1 as soon as the independently verified Wan
components and deterministic train-40 subset are ready, without waiting for the
last dev/blind archive or an unrelated full-snapshot marker.

**Architecture:** A segmented downloader writes a distinct receipt and never the
full snapshot manifest. A fail-closed Stage-1 verifier binds the DiT, VAE, T5,
tokenizer, 40 deterministic train archives, and four Stage-1 tasks to fixed HF
revisions and SHA-256 values. The preparation worker consumes that receipt,
extracts only train-40, builds the canonical train manifest, and leaves the full
clean-50 downloader running independently for later evaluation.

**Tech Stack:** Python 3.10+, `huggingface_hub`, SHA-256, atomic JSON/file
publication, pytest, Bash launchers, Wan2.2, PyTorch/FSDP.

**Spec:** `docs/superpowers/plans/2026-08-15-wan-action-lite-v3.md`, Task 0.

## Global Constraints

- Persistent outputs remain below `/data/di/worldarena2_track1_20260815`.
- The only external lookup is the exact-name `/data/fjy` lookup already
  authorized by the user; training never reads external paths.
- No hard links or symlinks are accepted as copied inputs.
- Fixed model revision: `921dbaf3f1674a56f47e83fb80a34bac8a8f203e`.
- Fixed clean-50 revision: `506c4e014f7dbd291e7d7683c79fc685dd3e2714`.
- A segmented receipt must never be named `.worldarena_manifest.json`.
- Each downloader uses `max_workers=1`; unrelated processes and GPUs are not
  touched.

---

### Task 1: Segmented download receipt

**Files:**
- Modify: `scripts/download_hf_snapshot_guarded.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: existing repository/revision metadata and `missing_assets()`.
- Produces: `--manifest-path PATH`; default remains
  `LOCAL_DIR/.worldarena_manifest.json`, while segmented callers provide a
  distinct path under `gates/`.

- [ ] Add a failing CLI test proving a custom receipt path is accepted only
  below the formal artifact root and is not the full-manifest basename.
- [ ] Run `pytest tests/test_scripts.py -k download_hf_snapshot_guarded -q` and
  observe failure because `--manifest-path` is absent.
- [ ] Add `--revision` and `--manifest-path`, pass the immutable revision to
  both `HfApi` and `snapshot_download`, and atomically publish the receipt.
- [ ] Re-run the focused test and `python -m py_compile`.
- [ ] Sync the script and test to `/home/huazhi/nlh/baseline` without starting
  any GPU process.

### Task 2: Stage-1 component and train-40 verifier

**Files:**
- Create: `src/worldarena_baseline/segmented_inputs.py`
- Create: `scripts/verify_wan_action_stage1_inputs.py`
- Test: `tests/test_segmented_inputs.py`

**Interfaces:**
- Produces:
  `verify_stage1_inputs(model_root: Path, snapshot_root: Path, model_assets:
  Sequence[PinnedAsset], clean_assets: Sequence[PinnedAsset], seed: int) -> dict`.
- Receipt contract: `wan-action-lite-v3-stage1-inputs/1`, containing immutable
  revisions, component SHA-256 maps, exact 40-task list, exact four-task list,
  and canonical receipt digest.

- [ ] Write failing tests for missing assets, size/hash mismatch, symlinks,
  deterministic 40/5/5 splitting, and the four lexicographically first train
  tasks.
- [ ] Run `pytest tests/test_segmented_inputs.py -q` and observe import failure.
- [ ] Implement the minimal verifier with path containment, regular-file,
  byte-size, SHA-256, split-count, and atomic-write checks.
- [ ] Run the focused tests, `py_compile`, and `git diff --check`.

### Task 3: Preparation worker consumes Stage-1 readiness

**Files:**
- Modify: `src/worldarena_baseline/v3_preparation.py`
- Modify: `scripts/run_wan_action_v3_preparation.py`
- Modify: `scripts/extract_robotwin_clean50_guarded.py`
- Modify: `scripts/build_robotwin_clean50_manifest.py`
- Test: `tests/test_v3_preparation.py`
- Test: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `gates/stage1-input-verification.v3.json` from Task 2.
- Produces: canonical `clean50_train.jsonl` with 2,000 rows and
  `stage1-source-4tasks.jsonl` with 200 rows while dev/blind remain optional.

- [ ] Add failing tests proving Stage-1 proceeds with the verified 40 tasks and
  rejects 39 tasks, an unverified archive, receipt drift, or a full marker that
  disagrees with the segmented receipt.
- [ ] Run the focused tests and observe the old worker still waits for both full
  manifests.
- [ ] Change `wait_inputs`, input verification, extraction, manifest building,
  and resume revalidation to consume the Stage-1 receipt; preserve all existing
  hash/provenance and disk gates.
- [ ] Gate URDF mapping on the frame-0 settled reference for 100 episodes at
  `<10 mm / <2 degrees`, keep all-frame drive-target versus physics-endpose
  maxima as non-gating tracking diagnostics, and retain full-trajectory
  projection/depth/edge-clamp gates.
- [ ] Re-run the complete v3 preparation and script suites locally and with the
  formal remote Python.
- [ ] Restart only the owned preparation worker, run its dry-run, and allow
  non-dry execution to reach the existing all-eight-GPU gate.

### Task 4: Operational handoff

**Files:**
- Modify: `.superpowers/sdd/2026-08-15-wan-action-lite-v3/task-6-orchestrator-report.md`

**Interfaces:**
- Consumes: reviewed Task 1-3 code and remote receipts.
- Produces: auditable process IDs, download ETA, receipt hashes, preparation
  phase, and the first valid Stage-1 checkpoint path.

- [ ] Stop only the two verified project download PIDs.
- [ ] Continue T5 and the six missing train-40 archives with one worker each;
  keep the one remaining dev archive as lower-priority background work.
- [ ] Generate and validate the Stage-1 input receipt, restart the preparation
  worker, and report the exact current phase.
- [ ] When all eight GPUs are free, run the 8-rank smoke; launch Stage-1 only if
  every existing hard gate passes.
