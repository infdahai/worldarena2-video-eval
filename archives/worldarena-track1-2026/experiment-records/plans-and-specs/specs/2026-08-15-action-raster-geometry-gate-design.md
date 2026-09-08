# Action Raster Geometry Gate Design

## Objective

Build a model-independent control pipeline that converts WorldArena Track 1
`joint14` trajectories into an eight-channel, left/right-stable action raster.
The pipeline must prove its geometry, temporal alignment, channel semantics,
and counterfactual behavior before any Wan2.2 weights are downloaded or
trained.

## Scope

This subsystem includes:

- validating Track 1 episode structure and trajectory shapes;
- projecting dual-arm Aloha keypoints into image space;
- constructing an eight-channel action raster;
- generating left/right-swap and time-reversal counterfactuals;
- producing JSON metrics and visual previews for a stratified episode set;
- refusing large downloads when disk reserve constraints are not met.

It explicitly excludes:

- Wan2.2 model loading, LoRA, FSDP, or sequence parallelism;
- SAM3 inference or mask generation;
- region-weighted flow-matching loss;
- candidate generation and final benchmark submission.

Those are separate subprojects gated by this subsystem.

## Observed Data Contract

The remote Track 1 split contains 1,000 episodes. A sampled episode has:

- `joint_action/vector`: `(123, 14)` float64;
- `endpose/left_endpose`: `(123, 7)` float64;
- `endpose/right_endpose`: `(123, 7)` float64;
- per-arm gripper arrays of length 123;
- one RGB first frame at `320x240`;
- one existing skeleton control at `640x480`, 81 frames, 24 fps.

The existing control videos are evidence and a regression reference, not
ground truth. They use shared RGB colors for both arms and therefore cannot
provide fixed left/right identity to a learned adapter.

## Architecture

The subsystem has four isolated layers:

1. `skeleton.py` exposes a geometry API that converts `joint14` into per-arm
   projected keypoints with a stable left/right label. Camera and robot
   transforms are explicit constructor inputs rather than hidden module state.
2. `action_raster.py` consumes projected keypoints and gripper scalars and
   produces a model-independent `(T, 8, H, W)` float32 tensor.
3. `action_audit.py` validates visibility, clipping, temporal motion, channel
   identity, and counterfactual transformations and writes a JSON report plus
   preview artifacts.
4. `disk_budget.py` checks expected payload and temporary-file allowance
   against hard reserve thresholds before an external download starts.

The geometry/raster boundary is intentional: raster unit tests use literal
2D coordinates and do not depend on URDF, while geometry tests can change
camera calibration without changing channel semantics.

## Eight-Channel Contract

Channels are always ordered as:

| Index | Name | Meaning |
| --- | --- | --- |
| 0 | `left_gripper_heatmap` | Gaussian centered at the left gripper |
| 1 | `right_gripper_heatmap` | Gaussian centered at the right gripper |
| 2 | `left_arm_flow_x` | normalized horizontal motion on the left arm support |
| 3 | `left_arm_flow_y` | normalized vertical motion on the left arm support |
| 4 | `right_arm_flow_x` | normalized horizontal motion on the right arm support |
| 5 | `right_arm_flow_y` | normalized vertical motion on the right arm support |
| 6 | `left_gripper_state` | left open/close scalar localized by its heatmap |
| 7 | `right_gripper_state` | right open/close scalar localized by its heatmap |

Coordinates are in raster pixel space. Flow values are divided by image width
or height and clipped to `[-1, 1]`. Frame zero flow is exactly zero. Heatmaps
and gripper-state channels are in `[0, 1]`. Off-screen points contribute zero
and never wrap around an image boundary.

## Counterfactual Contract

Two deterministic counterfactuals are required:

- `swap_arms`: swaps left/right geometry and gripper state before rasterizing;
- `reverse_time`: reverses the full trajectory before rasterizing.

Swapping arms must exchange channel groups `(0,2,3,6)` and `(1,4,5,7)` with
no cross-channel leakage. Reversing time must change the flow direction while
preserving the raster shape and value ranges.

## Gate Metrics

For each episode, the audit records:

- fraction of frames with a visible left gripper;
- fraction of frames with a visible right gripper;
- fraction of projected arm points inside the frame;
- heatmap peak and nonzero coverage per identity;
- mean absolute flow per arm;
- fraction of frames where swap counterfactuals exchange identity channels;
- fraction of frames where reverse-time flow disagrees with the original.

Initial progression gates are:

- all 20 stratified smoke episodes load and rasterize without shape errors;
- no NaN or infinity in geometry, raster, metrics, or preview artifacts;
- channel ranges and frame-zero flow satisfy the channel contract exactly;
- arm swap correctness is 100% by tensor equality;
- time reversal changes at least one non-static flow element whenever the
  original trajectory contains visible motion;
- any episode with less than 50% visibility for both grippers is flagged for
  camera calibration review rather than silently accepted.

The 50% visibility threshold is a diagnostic threshold, not a claim about the
final benchmark optimum. After camera calibration is accepted on 20 episodes,
the same audit runs on a 100-episode stratified set.

## Disk and Network Safety

All external payloads, caches, and temporary files belong below
`/data/di/worldarena2_track1_20260815`. The root filesystem is never a model or
dataset destination.

Before a download, the caller supplies the expected payload bytes. The guard
budgets `2 * payload` by default to cover partial and temporary files, then
requires:

- at least 200 GiB free on `/data` after the budget;
- at least 100 GiB free on `/` after the budget.

Network commands use the existing Clash HTTP proxy at `127.0.0.1:7890` and
explicit cache roots below the Track 1 artifact directory. A failed disk guard
prevents the download command from running.

## Remote Execution Boundary

Development and tests run first in the local `baseline` repository. Only the
new source, tests, and scripts are synchronized to `/home/huazhi/nlh/baseline`.
Large artifacts remain under `/data`; the old
`/data/di/worldarena2_track1_baseline` directory remains read-only and reusable.

No GPU process starts in this subsystem.
