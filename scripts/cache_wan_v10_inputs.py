#!/usr/bin/env python3
"""Plan or build only the missing offline inputs for Wan v10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from worldarena_baseline.wan_v10_data import build_v10_cache_extension


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _formal(path: Path) -> Path:
    result = path.resolve(strict=False)
    if result != FORMAL_ROOT and FORMAL_ROOT not in result.parents:
        raise ValueError(f"v10 cache path escapes formal root: {path}")
    return result


def _rows(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v10 cache manifest: {path}") from exc
    if not rows:
        raise ValueError(f"v10 cache manifest is empty: {path}")
    return rows


def _write_jsonl_atomic(path: Path, rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        partial.unlink(missing_ok=True)


def plan_extension(source_manifest: Path, cached_manifest: Path, output: Path) -> int:
    extension = build_v10_cache_extension(_rows(source_manifest), _rows(cached_manifest))
    _write_jsonl_atomic(output, extension)
    return len(extension)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan",), required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--cached-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.source_manifest, args.cached_manifest, args.output):
        _formal(path)
    return args


def main() -> None:
    args = parse_args()
    count = plan_extension(args.source_manifest, args.cached_manifest, args.output)
    print(json.dumps({"event": "v10_cache_extension", "rows": count, "output": str(args.output)}))


if __name__ == "__main__":
    main()
