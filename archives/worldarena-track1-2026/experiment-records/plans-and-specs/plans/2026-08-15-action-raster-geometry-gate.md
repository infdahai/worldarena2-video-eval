# Action Raster Geometry Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce and remotely validate a deterministic eight-channel action raster for Track 1 without downloading or loading Wan2.2.

**Architecture:** Separate camera/URDF projection from pure 2D rasterization, then audit real episodes through a CLI that emits JSON metrics and previews. Put disk reservation in a standalone guard so every later model or dataset downloader can call the same contract.

**Tech Stack:** Python 3.10+, NumPy, OpenCV, h5py, yourdfpy, pytest

**Spec:** `docs/superpowers/specs/2026-08-15-action-raster-geometry-gate-design.md`

## Global Constraints

- Preserve `/data/di/worldarena2_track1_baseline` as read-only input.
- Store new remote artifacts below `/data/di/worldarena2_track1_20260815`.
- Keep at least 200 GiB free on `/data` and 100 GiB free on `/` after any planned download.
- Budget twice the expected payload size unless the downloader proves a smaller temporary-file bound.
- Use `http://127.0.0.1:7890` for remote HTTP and HTTPS proxy variables.
- Do not download Wan2.2, SAM3, or RoboTwin training data in this plan.
- Do not start GPU inference or training in this plan.

---

### Task 1: Disk Budget Guard

**Files:**
- Create: `src/worldarena_baseline/disk_budget.py`
- Create: `tests/test_disk_budget.py`

**Interfaces:**
- Consumes: filesystem free-byte values supplied by `shutil.disk_usage` or tests.
- Produces: `DownloadBudget`, `evaluate_download_budget(...)`, and `require_download_budget(...)` for later download scripts.

- [ ] **Step 1: Write the failing tests**

```python
def test_budget_accepts_payload_while_preserving_both_reserves():
    report = evaluate_download_budget(
        payload_bytes=10,
        data_free_bytes=100,
        root_free_bytes=50,
        data_reserve_bytes=60,
        root_reserve_bytes=40,
        temporary_multiplier=2,
    )
    assert report.allowed is True
    assert report.required_data_bytes == 20


def test_budget_rejects_payload_that_consumes_data_reserve():
    report = evaluate_download_budget(
        payload_bytes=21,
        data_free_bytes=100,
        root_free_bytes=50,
        data_reserve_bytes=60,
        root_reserve_bytes=40,
        temporary_multiplier=2,
    )
    assert report.allowed is False
    assert report.reason == "data_reserve"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_disk_budget.py -q`

Expected: collection fails because `worldarena_baseline.disk_budget` does not exist.

- [ ] **Step 3: Implement the immutable report and pure evaluator**

Implement `DownloadBudget` as a frozen dataclass. Reject negative sizes and a
temporary multiplier below one. Set `required_data_bytes` to
`ceil(payload_bytes * temporary_multiplier)`. Check the root reserve even when
the payload destination is `/data`, because caches or logs must not consume the
root safety margin.

- [ ] **Step 4: Add and test the real-filesystem wrapper**

Add a test that supplies a temporary directory and zero reserves, then call:

```python
require_download_budget(
    payload_bytes=1,
    data_path=tmp_path,
    root_path=tmp_path,
    data_reserve_bytes=0,
    root_reserve_bytes=0,
)
```

Expected: it returns an allowed report. Run the test once before implementation
to observe `ImportError` or `AttributeError`, implement the wrapper using
`shutil.disk_usage`, and rerun until green.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_disk_budget.py -q`

Run: `uv run pytest -q`

Expected: all tests pass.

### Task 2: Stable Projected Dual-Arm Geometry

**Files:**
- Modify: `src/worldarena_baseline/skeleton.py`
- Modify: `tests/test_skeleton.py`

**Interfaces:**
- Consumes: `(T, 14)` joint actions and explicit image dimensions.
- Produces: `ProjectedArmTrajectory` with `left_points`, `right_points`, `left_visible`, `right_visible`, `left_gripper`, and `right_gripper` arrays.

- [ ] **Step 1: Write a failing renderer geometry test**

Use the existing minimal Aloha URDF fixture and two literal joint14 actions.
Call `renderer.project_actions(actions, num_frames=5)` and assert:

```python
assert trajectory.left_points.shape == (5, 9, 2)
assert trajectory.right_points.shape == (5, 9, 2)
assert trajectory.left_visible.shape == (5, 9)
assert trajectory.right_visible.shape == (5, 9)
np.testing.assert_allclose(trajectory.left_gripper, 1.0)
np.testing.assert_allclose(trajectory.right_gripper, 1.0)
```

The nine points are base, six arm-link endpoints, and two finger endpoints.

- [ ] **Step 2: Run the focused test and observe the missing method failure**

Run: `uv run pytest tests/test_skeleton.py::test_project_actions_preserves_arm_identity_and_gripper_state -q`

Expected: fail because `project_actions` is not implemented.

- [ ] **Step 3: Implement `ProjectedArmTrajectory` and `project_actions`**

Reuse `resample_actions`, `action_to_urdf_config`, `_world_positions`, and
`project_world_points`. Store each arm in a separate array; do not encode arm
identity using RGB colors. Validate input through the existing joint14 mapping.

- [ ] **Step 4: Refactor `render_actions` to consume projected geometry**

Preserve the existing rendered-video behavior while removing duplicate FK and
projection work. Keep the old public API and its tests green.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_skeleton.py -q`

Run: `uv run pytest -q`

Expected: all tests pass with no warnings.

### Task 3: Eight-Channel Action Raster

**Files:**
- Create: `src/worldarena_baseline/action_raster.py`
- Create: `tests/test_action_raster.py`

**Interfaces:**
- Consumes: `ProjectedArmTrajectory`, output height/width, Gaussian sigma, and arm thickness.
- Produces: `rasterize_action(trajectory, ...) -> np.ndarray` with shape `(T, 8, H, W)` and `swap_arms(trajectory)` / `reverse_time(trajectory)`.

- [ ] **Step 1: Write a failing literal-coordinate heatmap and identity test**

Create a two-frame trajectory in a `16x20` raster with the left gripper at
`(4, 5)` and right gripper at `(15, 10)`. Assert the left and right heatmap
argmax positions are exactly `(5, 4)` and `(10, 15)`, and assert no value is
written into the opposite heatmap at either peak.

- [ ] **Step 2: Run the test and observe module import failure**

Run: `uv run pytest tests/test_action_raster.py::test_raster_keeps_left_and_right_gripper_peaks_separate -q`

Expected: collection fails because `worldarena_baseline.action_raster` does not exist.

- [ ] **Step 3: Implement heatmap and localized gripper state channels**

Use a clipped Gaussian centered on the final two finger points' mean. Multiply
each heatmap by its clipped `[0, 1]` gripper scalar for state channels 6 and 7.
Return zeros for an invisible or non-finite gripper.

- [ ] **Step 4: Write a failing flow test**

Use a left arm that moves exactly two pixels right between frames and a static
right arm. Assert frame-zero flow is all zero, left x-flow contains `2 / width`,
left y-flow is zero, and both right flow channels are zero.

- [ ] **Step 5: Implement dense arm flow on visible segment support**

Rasterize each visible arm segment with OpenCV. Use the mean displacement of
the segment endpoints, normalize x by width and y by height, clip to `[-1, 1]`,
and average overlapping segment values through an accumulation/count buffer.

- [ ] **Step 6: Write failing counterfactual tests**

Assert `swap_arms` exchanges complete channel groups and `reverse_time` reverses
the trajectory and produces changed non-static flow. Expectations use literal
channel indices from the design contract.

- [ ] **Step 7: Implement counterfactual transformations and validate inputs**

Reject mismatched time lengths, invalid point shapes, non-positive image sizes,
non-positive sigma, and non-positive arm thickness with clear `ValueError`s.

- [ ] **Step 8: Run focused and full tests**

Run: `uv run pytest tests/test_action_raster.py -q`

Run: `uv run pytest -q`

Expected: all tests pass.

### Task 4: Episode Audit CLI and Preview

**Files:**
- Create: `src/worldarena_baseline/action_audit.py`
- Create: `tests/test_action_audit.py`
- Modify: `src/worldarena_baseline/cli.py`

**Interfaces:**
- Consumes: dataset root, URDF path, output directory, episode count, and image dimensions.
- Produces: one compressed raster `.npz`, one preview `.mp4`, and one metrics `.json` per episode plus `summary.json`.

- [ ] **Step 1: Write failing pure metric tests**

Create a literal raster with one visible left peak, no right peak, and one
nonzero left flow frame. Assert exact visibility, coverage, finite-value, and
motion metrics without mocking filesystem or OpenCV.

- [ ] **Step 2: Implement `measure_raster` and progression gate evaluation**

Return JSON-serializable Python scalars. Treat NaN, infinity, shape mismatch,
and counterfactual identity mismatch as hard failures; treat low visibility as
`needs_calibration`.

- [ ] **Step 3: Write a failing CLI integration test**

Build a temporary minimal dataset with one HDF5 episode, JSON instruction, PNG
first frame, and minimal URDF. Invoke `main([...])`, then assert the three
episode artifacts and summary exist and can be parsed/read by their real
libraries.

- [ ] **Step 4: Implement stratified selection and artifact generation**

Reuse `discover_episodes` and `select_length_stratified_episode_ids`. Write
artifacts atomically through a same-directory temporary path and rename only
after successful encoding.

- [ ] **Step 5: Run focused and full tests**

Run: `uv run pytest tests/test_action_audit.py -q`

Run: `uv run pytest -q`

Expected: all tests pass.

### Task 5: Remote 20-Episode Geometry Gate

**Files:**
- Create: `scripts/run_action_geometry_gate.sh`
- Modify: `tests/test_scripts.py`

**Interfaces:**
- Consumes: `/home/huazhi/nlh/track1_data`, existing Aloha URDF, and the preserved existing virtual environment.
- Produces: `/data/di/worldarena2_track1_20260815/eval_cache/action_geometry_gate_20/summary.json` and episode artifacts.

- [ ] **Step 1: Write a failing script contract test**

Run the script in `DRY_RUN=1` with temporary path variables. Assert it prints
the exact dataset, URDF, output directory, episode count, and zero GPU commands.

- [ ] **Step 2: Implement the guarded remote script**

Set `HF_HOME`, `TORCH_HOME`, `XDG_CACHE_HOME`, and `PIP_CACHE_DIR` below the new
artifact root. Refuse an output path outside that root. Invoke only the action
audit CLI with `--count 20`; do not invoke a downloader or CUDA command.

- [ ] **Step 3: Run local script and full test suites**

Run: `uv run pytest tests/test_scripts.py -q`

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Synchronize only changed source, tests, docs, and script**

Use `rsync` exclusions for `.git`, `.venv`, `.uv-cache`, `__pycache__`, and
`.pytest_cache`. Confirm destination size with `du -sh` and both filesystem
reserves with `df -h` before and after synchronization.

- [ ] **Step 5: Execute the remote 20-episode CPU gate**

Use the existing virtual environment under the preserved baseline directory.
No GPU process should appear before, during, or after the audit. Capture the
summary, failed episode IDs, and `needs_calibration` episode IDs.

- [ ] **Step 6: Decide the next subproject from evidence**

If all hard gates pass and visibility is acceptable, freeze the raster contract
and write the Wan2.2 adapter spec. If visibility fails, stop before model work
and calibrate camera/robot transforms using paired training RGB or verified
RoboTwin scene configuration.

