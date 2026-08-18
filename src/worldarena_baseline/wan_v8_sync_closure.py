"""Explicit v8 source-closure anchors; later tasks expand this frozen list."""

from __future__ import annotations

from pathlib import Path


V8_CLOSURE_ANCHORS = (
    "src/worldarena_baseline/wan_v8_data.py",
    "src/worldarena_baseline/wan_v8_sync_closure.py",
    "scripts/prepare_wan_v8_data.py",
)


def validate_v8_source_anchors(source_root: Path | str) -> tuple[Path, ...]:
    """Fail before sync when a committed v8 entrypoint is missing or a symlink."""
    root = Path(source_root)
    resolved: list[Path] = []
    for relative in V8_CLOSURE_ANCHORS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v8 closure anchor is not a regular file: {relative}")
        resolved.append(path)
    return tuple(resolved)
