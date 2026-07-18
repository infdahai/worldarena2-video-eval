from __future__ import annotations

from collections.abc import Sequence
import os
from pathlib import Path

import h5py
import numpy as np

from .manifest import EpisodeSpec
from .skeleton import AlohaSkeletonRenderer
from .video import probe_video, write_video


def select_length_stratified_episode_ids(episodes: Sequence, count: int) -> list[int]:
    if count < 1 or not episodes:
        raise ValueError("episodes and a positive count are required")
    ordered = sorted(episodes, key=lambda item: (item.trajectory_length, item.episode_id))
    count = min(count, len(ordered))
    indices = np.linspace(0, len(ordered) - 1, count).astype(int)
    return [ordered[index].episode_id for index in indices]


def select_smoke_episode_ids(episodes: Sequence, count: int = 3) -> list[int]:
    return select_length_stratified_episode_ids(episodes, count)


def load_joint_actions(episode: EpisodeSpec) -> np.ndarray:
    with h5py.File(episode.hdf5_path, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"])
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"episode{episode.episode_id}: expected joint14, got {actions.shape}")
    return actions


def prepare_control_video(
    episode: EpisodeSpec,
    renderer: AlohaSkeletonRenderer,
    controls_dir: Path | str,
    *,
    num_frames: int = 81,
    fps: float = 24.0,
) -> Path:
    output = Path(controls_dir) / episode.output_name
    if output.is_file():
        probe = probe_video(output)
        if (
            probe.frame_count == num_frames
            and probe.width == renderer.width
            and probe.height == renderer.height
        ):
            return output
    frames = renderer.render_actions(load_joint_actions(episode), num_frames=num_frames)
    partial = output.with_name(f"{output.stem}.partial.mp4")
    write_video(frames, partial, fps=fps)
    probe = probe_video(partial)
    if probe.frame_count != num_frames:
        raise ValueError(f"control video has {probe.frame_count} frames, expected {num_frames}")
    os.replace(partial, output)
    return output
