#!/usr/bin/env python3
"""Validate the exact committed source closure before a Wan v7 remote sync."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    receipt = validate_sync_closure(args.repo_root)
    if args.output is not None:
        _atomic_json(args.output, receipt)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
