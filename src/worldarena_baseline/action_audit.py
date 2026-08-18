from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from .action_raster import rasterize_action, reverse_time, swap_arms
from .manifest import EpisodeSpec
from .pipeline import load_joint_actions, select_length_stratified_episode_ids
from .skeleton import AlohaSkeletonRenderer, ProjectedArmTrajectory
from .video import write_video


def measure_raster(raster: np.ndarray) -> dict[str, bool | float]:
    values = np.asarray(raster)
    if values.ndim != 4 or values.shape[1] != 8:
        raise ValueError("raster must have shape (T, 8, H, W)")
    if values.shape[0] < 1 or values.shape[2] < 1 or values.shape[3] < 1:
        raise ValueError("raster dimensions must be positive")

    finite = bool(np.isfinite(values).all())
    measured = np.nan_to_num(values, nan=0.0, posinf=2.0, neginf=-2.0)
    left_heatmap = measured[:, 0]
    right_heatmap = measured[:, 1]
    left_flow = measured[:, 2:4]
    right_flow = measured[:, 4:6]
    heatmap_state = measured[:, [0, 1, 6, 7]]
    flow = measured[:, 2:6]
    return {
        "finite": finite,
        "frame_zero_flow_zero": bool(np.count_nonzero(flow[0]) == 0),
        "left_gripper_visible_fraction": float(
            np.mean(np.any(left_heatmap > 0, axis=(1, 2)))
        ),
        "right_gripper_visible_fraction": float(
            np.mean(np.any(right_heatmap > 0, axis=(1, 2)))
        ),
        "left_heatmap_coverage": float(np.count_nonzero(left_heatmap) / left_heatmap.size),
        "right_heatmap_coverage": float(
            np.count_nonzero(right_heatmap) / right_heatmap.size
        ),
        "left_heatmap_peak": float(np.max(left_heatmap)),
        "right_heatmap_peak": float(np.max(right_heatmap)),
        "left_mean_absolute_flow": float(np.mean(np.abs(left_flow))),
        "right_mean_absolute_flow": float(np.mean(np.abs(right_flow))),
        "heatmap_state_range_valid": bool(
            np.min(heatmap_state) >= 0 and np.max(heatmap_state) <= 1
        ),
        "flow_range_valid": bool(np.min(flow) >= -1 and np.max(flow) <= 1),
    }


def evaluate_progression_gate(
    metrics: Mapping[str, bool | float],
) -> dict[str, bool | list[str]]:
    hard_checks = {
        "non_finite_values": bool(metrics["finite"]),
        "frame_zero_flow_nonzero": bool(metrics["frame_zero_flow_zero"]),
        "heatmap_state_out_of_range": bool(metrics["heatmap_state_range_valid"]),
        "flow_out_of_range": bool(metrics["flow_range_valid"]),
    }
    if "arm_swap_correct" in metrics:
        hard_checks["arm_swap_incorrect"] = bool(metrics["arm_swap_correct"])
    if "reverse_time_changed_when_moving" in metrics:
        hard_checks["reverse_time_unchanged"] = bool(
            metrics["reverse_time_changed_when_moving"]
        )
    hard_failures = [name for name, passed in hard_checks.items() if not passed]
    both_low_visibility = (
        float(metrics["left_gripper_visible_fraction"]) < 0.5
        and float(metrics["right_gripper_visible_fraction"]) < 0.5
    )
    calibration_reasons = (
        ["both_grippers_low_visibility"] if both_low_visibility else []
    )
    return {
        "hard_pass": not hard_failures,
        "hard_failures": hard_failures,
        "needs_calibration": bool(calibration_reasons),
        "calibration_reasons": calibration_reasons,
    }


def _points_inside_fraction(
    points: np.ndarray, visible: np.ndarray, *, height: int, width: int
) -> float:
    values = np.asarray(points)
    mask = (
        np.asarray(visible, dtype=np.bool_)
        & np.isfinite(values).all(axis=-1)
        & (values[..., 0] >= 0)
        & (values[..., 0] < width)
        & (values[..., 1] >= 0)
        & (values[..., 1] < height)
    )
    return float(np.mean(mask))


def _preview_frames(raster: np.ndarray) -> np.ndarray:
    left = np.asarray(raster[:, 0], dtype=np.float32)
    right = np.asarray(raster[:, 1], dtype=np.float32)
    flow = np.sqrt(np.sum(np.asarray(raster[:, 2:6], dtype=np.float32) ** 2, axis=1))
    preview = np.stack([left, np.clip(flow, 0, 1), right], axis=-1)
    return np.rint(np.clip(preview, 0, 1) * 255).astype(np.uint8)


def _write_json_atomic(path: Path, payload: Mapping) -> None:
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(partial, path)


def _write_raster_atomic(path: Path, raster: np.ndarray) -> None:
    partial = path.with_name(f"{path.stem}.partial.npz")
    np.savez_compressed(partial, raster=raster.astype(np.float16))
    os.replace(partial, path)


def audit_episode(
    episode: EpisodeSpec,
    renderer: AlohaSkeletonRenderer,
    output_dir: Path | str,
    *,
    num_frames: int = 81,
    fps: float = 24.0,
    sigma: float = 3.0,
    arm_thickness: int = 3,
) -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trajectory = renderer.project_actions(
        load_joint_actions(episode), num_frames=num_frames
    )
    raster = rasterize_action(
        trajectory,
        height=renderer.height,
        width=renderer.width,
        sigma=sigma,
        arm_thickness=arm_thickness,
    )
    swapped_raster = rasterize_action(
        swap_arms(trajectory),
        height=renderer.height,
        width=renderer.width,
        sigma=sigma,
        arm_thickness=arm_thickness,
    )
    reversed_raster = rasterize_action(
        reverse_time(trajectory),
        height=renderer.height,
        width=renderer.width,
        sigma=sigma,
        arm_thickness=arm_thickness,
    )
    left_channels = [0, 2, 3, 6]
    right_channels = [1, 4, 5, 7]
    arm_swap_correct = bool(
        np.array_equal(swapped_raster[:, left_channels], raster[:, right_channels])
        and np.array_equal(swapped_raster[:, right_channels], raster[:, left_channels])
    )
    has_motion = bool(np.count_nonzero(raster[:, 2:6]))
    reverse_time_changed = bool(
        not has_motion
        or not np.array_equal(reversed_raster[:, 2:6], raster[:, 2:6])
    )
    metrics = measure_raster(raster)
    metrics.update(
        {
            "episode_id": episode.episode_id,
            "scene": episode.scene,
            "raster_shape": list(raster.shape),
            "left_points_inside_fraction": _points_inside_fraction(
                trajectory.left_points,
                trajectory.left_visible,
                height=renderer.height,
                width=renderer.width,
            ),
            "right_points_inside_fraction": _points_inside_fraction(
                trajectory.right_points,
                trajectory.right_visible,
                height=renderer.height,
                width=renderer.width,
            ),
            "arm_swap_correct": arm_swap_correct,
            "reverse_time_changed_when_moving": reverse_time_changed,
        }
    )
    metrics["gate"] = evaluate_progression_gate(metrics)

    stem = f"episode_{episode.episode_id:06d}"
    _write_raster_atomic(output / f"{stem}.raster.npz", raster)
    video_path = output / f"{stem}.preview.mp4"
    partial_video = output / f"{stem}.preview.partial.mp4"
    write_video(_preview_frames(raster), partial_video, fps=fps)
    os.replace(partial_video, video_path)
    _write_json_atomic(output / f"{stem}.metrics.json", metrics)
    return metrics


def run_action_audit(
    episodes: list[EpisodeSpec],
    renderer: AlohaSkeletonRenderer,
    output_dir: Path | str,
    *,
    count: int,
    num_frames: int = 81,
    fps: float = 24.0,
    sigma: float = 3.0,
    arm_thickness: int = 3,
) -> dict:
    selected_ids = set(select_length_stratified_episode_ids(episodes, count=count))
    selected = [episode for episode in episodes if episode.episode_id in selected_ids]
    records = [
        audit_episode(
            episode,
            renderer,
            output_dir,
            num_frames=num_frames,
            fps=fps,
            sigma=sigma,
            arm_thickness=arm_thickness,
        )
        for episode in selected
    ]
    summary = {
        "episode_count": len(records),
        "episode_ids": [int(record["episode_id"]) for record in records],
        "hard_pass": all(bool(record["gate"]["hard_pass"]) for record in records),
        "hard_failure_episode_ids": [
            int(record["episode_id"])
            for record in records
            if not bool(record["gate"]["hard_pass"])
        ],
        "needs_calibration_episode_ids": [
            int(record["episode_id"])
            for record in records
            if bool(record["gate"]["needs_calibration"])
        ],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(output / "summary.json", summary)
    return summary
