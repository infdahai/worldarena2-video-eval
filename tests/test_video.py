from __future__ import annotations

from pathlib import Path

import imageio.v3 as iio
import numpy as np

from worldarena_baseline.video import assigned_to_worker, probe_video, write_video


def test_write_video_preserves_frame_count_and_dimensions(tmp_path: Path) -> None:
    frames = np.zeros((5, 48, 64, 3), dtype=np.uint8)
    frames[:, 12:36, 16:48, 1] = 255
    output = tmp_path / "control.mp4"

    write_video(frames, output, fps=24.0)

    decoded = np.stack(list(iio.imiter(output, plugin="pyav")))
    assert decoded.shape == (5, 48, 64, 3)
    assert output.stat().st_size > 0
    probe = probe_video(output)
    assert probe.frame_count == 5
    assert (probe.width, probe.height) == (64, 48)


def test_assigned_to_worker_is_deterministic_and_complete() -> None:
    episode_ids = list(range(1, 11))

    partitions = [assigned_to_worker(episode_ids, index, 3) for index in range(3)]

    assert partitions[0] == [1, 4, 7, 10]
    assert partitions[1] == [2, 5, 8]
    assert partitions[2] == [3, 6, 9]
    assert sorted(item for partition in partitions for item in partition) == episode_ids
