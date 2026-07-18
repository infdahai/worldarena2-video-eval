from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np

from worldarena_baseline import worker
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


def test_worker_does_not_load_model_for_an_empty_shard(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(worker, "discover_episodes", lambda _: [SimpleNamespace(episode_id=1)])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "worker",
            "--dataset-root", str(tmp_path),
            "--controls-dir", str(tmp_path),
            "--output-dir", str(tmp_path / "videos"),
            "--checkpoint", str(tmp_path),
            "--oscar-repo", str(tmp_path),
            "--gpu-index", "0",
            "--worker-index", "1",
            "--worker-count", "2",
        ],
    )

    assert worker.main() == 0
    assert '"event": "empty_assignment"' in capsys.readouterr().out
