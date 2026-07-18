from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import imageio.v3 as iio
import numpy as np


@dataclass(frozen=True)
class VideoProbe:
    frame_count: int
    width: int
    height: int
    fps: float | None


def probe_video(path: Path | str) -> VideoProbe:
    frame_count = 0
    width = height = 0
    for frame in iio.imiter(path, plugin="pyav"):
        if frame_count == 0:
            height, width = frame.shape[:2]
        frame_count += 1
    if frame_count == 0:
        raise ValueError(f"video has no decodable frames: {path}")
    try:
        metadata = iio.immeta(path, plugin="pyav")
        fps = float(metadata["fps"]) if "fps" in metadata else None
    except Exception:
        fps = None
    return VideoProbe(frame_count, width, height, fps)


def write_video(frames: np.ndarray, output: Path | str, *, fps: float) -> None:
    values = np.asarray(frames)
    if values.ndim != 4 or values.shape[-1] != 3 or values.dtype != np.uint8:
        raise ValueError("frames must be uint8 with shape (T, H, W, 3)")
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = iio.imopen(path, "w", plugin="pyav")
    try:
        writer.init_video_stream("libx264", fps=fps)
        for frame in values:
            writer.write_frame(np.ascontiguousarray(frame))
    finally:
        writer.close()


def assigned_to_worker(
    episode_ids: Iterable[int], worker_index: int, worker_count: int
) -> list[int]:
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("invalid worker index or count")
    return [
        episode_id
        for position, episode_id in enumerate(episode_ids)
        if position % worker_count == worker_index
    ]
