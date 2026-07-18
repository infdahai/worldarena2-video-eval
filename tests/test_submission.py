from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from worldarena_baseline.submission import (
    MODEL_METADATA,
    build_submission_archive,
    render_model_readme,
    validate_submission_archive,
)


def test_model_readme_has_required_hybrid_metadata() -> None:
    readme = render_model_readme()

    assert readme.startswith("---\n")
    for key, value in MODEL_METADATA.items():
        assert f"{key}: {value}" in readme
    assert "control_type: hybrid" in readme


def test_build_and_validate_submission_archive(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    for episode_id in (1, 2, 3):
        (videos / f"episode_{episode_id:06d}.mp4").write_bytes(b"mp4")
    (tmp_path / "model_readme.md").write_text(render_model_readme(), encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"

    build_submission_archive(tmp_path, archive)
    report = validate_submission_archive(archive, expected_episodes=3)

    assert report.video_count == 3
    assert report.video_names[0] == "videos/episode_000001.mp4"
    assert report.video_names[-1] == "videos/episode_000003.mp4"


def test_validator_rejects_unsafe_member(tmp_path: Path) -> None:
    archive = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive, "w:gz") as handle:
        payload = b"bad"
        info = tarfile.TarInfo("../escape.mp4")
        info.size = len(payload)
        handle.addfile(info, io.BytesIO(payload))

    with pytest.raises(ValueError, match="unsafe archive path"):
        validate_submission_archive(archive, expected_episodes=0)


def test_validator_requires_exact_episode_set(tmp_path: Path) -> None:
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "episode_000001.mp4").write_bytes(b"mp4")
    (tmp_path / "model_readme.md").write_text(render_model_readme(), encoding="utf-8")
    archive = tmp_path / "submission.tar.gz"
    build_submission_archive(tmp_path, archive)

    with pytest.raises(ValueError, match="expected 2 videos"):
        validate_submission_archive(archive, expected_episodes=2)
