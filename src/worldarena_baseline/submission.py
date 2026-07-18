from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import yaml


MODEL_METADATA = {
    "model_name": "OSCAR-2B-WorldArena2-Baseline",
    "version": "oscar-2b-zero-shot-5step-v1",
    "organization": "Huazhi AI",
    "release_year": 2026,
    "source_type": "open_source",
    "control_type": "hybrid",
}
REQUIRED_METADATA = tuple(MODEL_METADATA)
VIDEO_RE = re.compile(r"^videos/episode_(\d{6})\.mp4$")


@dataclass(frozen=True)
class SubmissionReport:
    video_count: int
    video_names: tuple[str, ...]
    metadata: dict[str, object]


def render_model_readme() -> str:
    metadata = yaml.safe_dump(MODEL_METADATA, sort_keys=False).strip()
    return (
        f"---\n{metadata}\n"
        "code_url: https://github.com/wuzy2115/oscar-public\n"
        "---\n\n"
        "# OSCAR-2B WorldArena 2.0 Baseline\n\n"
        "Zero-shot Track 1 baseline conditioned on the official first frame, "
        "instruction, and a temporally resampled dual-arm joint14 skeleton.\n"
    )


def build_submission_archive(staging_dir: Path | str, archive_path: Path | str) -> None:
    staging = Path(staging_dir)
    archive = Path(archive_path)
    readme = staging / "model_readme.md"
    videos = staging / "videos"
    if not readme.is_file() or not videos.is_dir():
        raise ValueError("staging directory must contain model_readme.md and videos/")
    video_files = sorted(videos.glob("*.mp4"))
    if any(path.is_symlink() or not path.is_file() for path in video_files):
        raise ValueError("videos must be regular files")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as handle:
        handle.add(readme, arcname="model_readme.md", recursive=False)
        for video in video_files:
            handle.add(video, arcname=f"videos/{video.name}", recursive=False)


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not member.issym()
        and not member.islnk()
        and (member.isfile() or member.isdir())
    )


def _read_metadata(handle: tarfile.TarFile, member: tarfile.TarInfo) -> dict[str, object]:
    stream = handle.extractfile(member)
    if stream is None:
        raise ValueError("model_readme.md is not a regular file")
    text = stream.read().decode("utf-8")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise ValueError("model_readme.md must start with YAML front matter")
    front_matter = text.split("---\n", 2)[1]
    metadata = yaml.safe_load(front_matter)
    if not isinstance(metadata, dict):
        raise ValueError("invalid model metadata")
    missing = [key for key in REQUIRED_METADATA if key not in metadata]
    if missing:
        raise ValueError(f"missing model metadata: {', '.join(missing)}")
    return metadata


def validate_submission_archive(
    archive_path: Path | str, expected_episodes: int = 1000
) -> SubmissionReport:
    with tarfile.open(archive_path, "r:gz") as handle:
        members = handle.getmembers()
        for member in members:
            if not _safe_member(member):
                raise ValueError(f"unsafe archive path or member type: {member.name}")
        readmes = [member for member in members if member.name == "model_readme.md"]
        if len(readmes) != 1:
            raise ValueError("archive must contain exactly one model_readme.md")
        metadata = _read_metadata(handle, readmes[0])
        video_names = tuple(
            sorted(member.name for member in members if member.isfile() and VIDEO_RE.match(member.name))
        )
        if len(video_names) != expected_episodes:
            raise ValueError(f"expected {expected_episodes} videos, found {len(video_names)}")
        expected_names = tuple(
            f"videos/episode_{episode_id:06d}.mp4"
            for episode_id in range(1, expected_episodes + 1)
        )
        if video_names != expected_names:
            raise ValueError("video episode set is incomplete or misnumbered")
        return SubmissionReport(len(video_names), video_names, metadata)
