from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from worldarena_baseline.manifest import discover_episodes
from worldarena_baseline.naming import output_filename
from worldarena_baseline import pipeline
from worldarena_baseline.pipeline import select_smoke_episode_ids
from worldarena_baseline.resample import resample_actions


def _write_episode(root: Path, episode_id: int, actions: np.ndarray) -> None:
    scene = "fixed_scene_task"
    for kind in ("data", "instructions", "first_frame"):
        (root / kind / scene).mkdir(parents=True, exist_ok=True)

    with h5py.File(root / "data" / scene / f"episode{episode_id}.hdf5", "w") as handle:
        handle.create_dataset("joint_action/vector", data=actions)
    (root / "instructions" / scene / f"episode{episode_id}.json").write_text(
        json.dumps({"instruction": f"task {episode_id}"}), encoding="utf-8"
    )
    (root / "first_frame" / scene / f"episode{episode_id}.png").write_bytes(b"png")


def test_resample_actions_preserves_endpoints_and_shape() -> None:
    actions = np.stack(
        [np.zeros(14), np.full(14, 10.0), np.full(14, 20.0)], axis=0
    )

    sampled = resample_actions(actions, num_frames=5)

    assert sampled.shape == (5, 14)
    np.testing.assert_allclose(sampled[:, 0], [0.0, 5.0, 10.0, 15.0, 20.0])
    np.testing.assert_array_equal(sampled[0], actions[0])
    np.testing.assert_array_equal(sampled[-1], actions[-1])


def test_resample_single_action_repeats_it() -> None:
    action = np.arange(14, dtype=np.float64)[None, :]

    sampled = resample_actions(action, num_frames=81)

    assert sampled.shape == (81, 14)
    np.testing.assert_array_equal(sampled[0], action[0])
    np.testing.assert_array_equal(sampled[-1], action[0])


def test_discover_episodes_sorts_and_validates_joint14(tmp_path: Path) -> None:
    _write_episode(tmp_path, 10, np.zeros((3, 14)))
    _write_episode(tmp_path, 2, np.zeros((7, 14)))

    episodes = discover_episodes(tmp_path)

    assert [episode.episode_id for episode in episodes] == [2, 10]
    assert [episode.trajectory_length for episode in episodes] == [7, 3]
    assert episodes[0].instruction == "task 2"
    assert episodes[0].output_name == "episode_000002.mp4"


def test_discover_episodes_rejects_non_joint14(tmp_path: Path) -> None:
    _write_episode(tmp_path, 1, np.zeros((3, 7)))

    with pytest.raises(ValueError, match="expected joint14"):
        discover_episodes(tmp_path)


@pytest.mark.parametrize(
    ("episode_id", "expected"),
    [(1, "episode_000001.mp4"), (1000, "episode_001000.mp4")],
)
def test_output_filename_is_zero_padded(episode_id: int, expected: str) -> None:
    assert output_filename(episode_id) == expected


def test_output_filename_rejects_non_positive_id() -> None:
    with pytest.raises(ValueError, match="positive"):
        output_filename(0)


def test_select_smoke_episodes_covers_short_median_and_long() -> None:
    class Episode:
        def __init__(self, episode_id: int, trajectory_length: int) -> None:
            self.episode_id = episode_id
            self.trajectory_length = trajectory_length

    episodes = [Episode(4, 40), Episode(1, 10), Episode(3, 30), Episode(2, 20)]

    assert select_smoke_episode_ids(episodes, count=3) == [1, 2, 4]


def test_length_stratified_selection_is_deterministic_and_capped() -> None:
    class Episode:
        def __init__(self, episode_id: int, trajectory_length: int) -> None:
            self.episode_id = episode_id
            self.trajectory_length = trajectory_length

    episodes = [Episode(4, 40), Episode(1, 10), Episode(3, 30), Episode(2, 20)]

    assert pipeline.select_length_stratified_episode_ids(episodes, count=3) == [1, 2, 4]
    assert pipeline.select_length_stratified_episode_ids(episodes, count=10) == [1, 2, 3, 4]
