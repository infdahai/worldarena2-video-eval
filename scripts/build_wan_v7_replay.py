#!/usr/bin/env python3
"""Derive the immutable 50-step Wan v7 replay from the v6 clean-1000 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

from worldarena_baseline.wan_v7_training import (
    build_v7_replay_from_v6,
    build_v7_single_gpu_replay,
)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-replay", type=Path, required=True)
    parser.add_argument("--clean-1000-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topology", choices=("seven-rank", "single-gpu"), default="seven-rank")
    args = parser.parse_args(argv)

    source = json.loads(args.v6_replay.read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(args.clean_1000_manifest.read_bytes()).hexdigest()
    if args.topology == "seven-rank":
        payload = build_v7_replay_from_v6(
            source, clean1000_manifest_sha256=manifest_hash
        )
    else:
        # This validates the CLI manifest against the source-controlled pin;
        # the single-GPU builder then selects only physical rank six.
        build_v7_replay_from_v6(source, clean1000_manifest_sha256=manifest_hash)
        payload = build_v7_single_gpu_replay(source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, args.output)


if __name__ == "__main__":
    main()
