"""Persistent-output boundary for Wan-Action video generation."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Mapping


PROJECT_ARTIFACT_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _is_symlink_lstat(path: Path) -> bool:
    try:
        return stat.S_ISLNK(path.lstat().st_mode)
    except FileNotFoundError:
        return False


def _reject_symlink_components(path: Path, root: Path, label: str) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be lexically beneath artifact root: {root}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if _is_symlink_lstat(current):
            raise ValueError(f"{label} must not contain an existing symlink: {current}")


def resolve_video_output_under_root(
    output: Path | str, artifact_root: Path | str
) -> tuple[Path, Path]:
    """Independently lstat and resolve the video and its JSON sidecar."""
    root = Path(artifact_root).resolve(strict=False)
    raw_output = Path(os.path.abspath(os.fspath(output)))
    raw_sidecar = raw_output.with_suffix(raw_output.suffix + ".json")
    for candidate, label in (
        (raw_output, "video output"),
        (raw_sidecar, "sidecar output"),
    ):
        _reject_symlink_components(candidate, root, label)
        resolved = candidate.resolve(strict=False)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} must resolve beneath artifact root: {root}") from exc
    output_resolved = raw_output.resolve(strict=False)
    if output_resolved == root:
        raise ValueError("video output must be a file beneath artifact root")
    return output_resolved, raw_sidecar.resolve(strict=False)


def resolve_project_video_output(
    output: Path | str,
    *,
    artifact_root: Path | str = PROJECT_ARTIFACT_ROOT,
) -> tuple[Path, Path]:
    root = Path(artifact_root).resolve(strict=False)
    expected = PROJECT_ARTIFACT_ROOT.resolve(strict=False)
    if root != expected:
        raise ValueError(f"artifact root must be the project artifact root: {expected}")
    return resolve_video_output_under_root(output, root)


def write_json_sidecar_atomic(path: Path | str, payload: Mapping) -> None:
    """Atomically replace a regular sidecar without ever following a symlink."""
    target = Path(path)
    if _is_symlink_lstat(target):
        raise ValueError(f"sidecar output must not be an existing symlink: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if _is_symlink_lstat(target.parent):
        raise ValueError(f"sidecar parent must not be an existing symlink: {target.parent}")
    partial = target.with_name(f".{target.name}.partial-{os.getpid()}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(partial, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if _is_symlink_lstat(target):
            raise ValueError(f"sidecar output must not be an existing symlink: {target}")
        os.replace(partial, target)
    finally:
        try:
            partial.unlink()
        except FileNotFoundError:
            pass
