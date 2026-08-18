#!/usr/bin/env python3
"""Build the immutable 500-step, single-GPU v10 balanced replay."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from worldarena_baseline.wan_v10_replay import build_v10_replay


ROOT = Path("/data/di/worldarena2_track1_20260815")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.optimizer_manifest, args.output):
        resolved = path.resolve(strict=False)
        if resolved != ROOT and ROOT not in resolved.parents:
            raise ValueError(f"v10 replay path escapes formal root: {path}")
    rows = [
        json.loads(line)
        for line in args.optimizer_manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    roles = {str(row["sample"]): str(row["v10_stratum"]) for row in rows}
    if len(rows) != 2060 or len(roles) != 2060:
        raise ValueError("v10 replay requires exact optimizer-full-action-2060")
    replay = build_v10_replay(roles, steps=500, world_size=1, seed=20260819)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            for row in replay:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(json.dumps({"contract": "wan-v10-replay/1", "rows": len(replay), "output": str(args.output)}))


if __name__ == "__main__":
    main()
