"""Neutral RoboTwin episode discovery and decoding helpers.

These contracts are shared by the current Wan cache pipeline and development
holdouts.  They intentionally do not depend on the retired VACE LoRA stack.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import h5py
import numpy as np


def sample_name(task: str, variant: str, episode_index: int) -> str:
    """Return a stable name which cannot collide across RoboTwin variants."""
    return f"{task}__{variant}__episode_{episode_index:06d}"


def decode_jpeg_frame(encoded: np.ndarray) -> np.ndarray:
    """Decode the uint8 JPEG vectors stored in RoboTwin HDF5 files to RGB."""
    if isinstance(encoded, (bytes, np.bytes_)):
        payload = np.frombuffer(encoded, dtype=np.uint8)
    else:
        payload = np.asarray(encoded, dtype=np.uint8).reshape(-1)
    frame = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("could not decode RoboTwin JPEG frame")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def resample_frame_indices(source_length: int, *, num_frames: int) -> np.ndarray:
    """Choose nearest source frames while preserving the first and last frame."""
    if source_length < 1:
        raise ValueError("source_length must be positive")
    if num_frames < 1:
        raise ValueError("num_frames must be positive")
    return np.rint(np.linspace(0, source_length - 1, num_frames)).astype(np.int64)


def load_robotwin_episode(
    hdf5_path: Path | str,
    instruction_path: Path | str,
    *,
    width: int,
    height: int,
    num_frames: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Load one public RoboTwin episode as RGB frames, joint14 and instruction."""
    with h5py.File(hdf5_path, "r") as handle:
        try:
            actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
            encoded_frames = handle["observation/head_camera/rgb"]
        except KeyError as exc:
            raise ValueError(f"missing RoboTwin field in {hdf5_path}: {exc}") from exc
        if actions.ndim != 2 or actions.shape[1] != 14:
            raise ValueError(f"expected joint14 actions in {hdf5_path}, got {actions.shape}")
        if len(encoded_frames) != len(actions):
            raise ValueError("head-camera RGB and joint14 trajectory lengths differ")
        indices = resample_frame_indices(len(encoded_frames), num_frames=num_frames)
        frames = [
            cv2.resize(
                decode_jpeg_frame(encoded_frames[int(index)]),
                (width, height),
                interpolation=cv2.INTER_AREA,
            )
            for index in indices
        ]

    payload = json.loads(Path(instruction_path).read_text())
    prompts = payload.get("seen")
    if not isinstance(prompts, list) or not prompts or not isinstance(prompts[0], str):
        raise ValueError(f"missing non-empty seen instruction list: {instruction_path}")
    return np.stack(frames), actions, prompts[0]


def discover_robotwin_episodes(root: Path | str) -> list[tuple[int, Path, Path]]:
    """Return complete ``data/episodeN.hdf5`` / instruction JSON pairs."""
    root_path = Path(root)
    episodes: list[tuple[int, Path, Path]] = []
    for hdf5_path in root_path.glob("data/episode*.hdf5"):
        suffix = hdf5_path.stem.removeprefix("episode")
        if not suffix.isdigit():
            continue
        instruction_path = root_path / "instructions" / f"{hdf5_path.stem}.json"
        if instruction_path.is_file():
            episodes.append((int(suffix), hdf5_path, instruction_path))
    return sorted(episodes, key=lambda item: item[0])
