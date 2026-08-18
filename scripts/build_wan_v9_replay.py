#!/usr/bin/env python3
"""Derive the immutable 500-step v9 single-GPU replay from approved v8 replay."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


_NEGATIVE_CYCLE = (
    "shift+1", "reverse", "shift-1", "swap", "shift+1",
    "shift-1", "shift+1", "reverse", "shift-1", "swap",
)


def negative_family_for_step(step: int) -> str:
    if type(step) is not int or step <= 0:
        raise ValueError("v9 replay step must be a positive integer")
    return _NEGATIVE_CYCLE[(step - 1) % len(_NEGATIVE_CYCLE)]


def _seed(label: str, step: int, sample: str) -> int:
    return int.from_bytes(hashlib.sha256(f"v9:{label}:{step}:{sample}".encode()).digest()[:8], "big")


def build(source_rows: list[dict[str, object]], audit_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if len(source_rows) != 250 or [row.get("optimizer_step") for row in source_rows] != list(range(1, 251)):
        raise ValueError("v9 replay source must be the exact v8 250-step replay")
    audit = {str(row.get("sample")) for row in audit_rows}
    if len(audit) != 20:
        raise ValueError("v9 replay requires exact audit20")
    result = []
    for step in range(1, 501):
        source = source_rows[(step - 1) % len(source_rows)]
        sample = str(source.get("sample"))
        if not sample or sample in audit:
            raise ValueError("v9 replay source leaks audit20")
        result.append({
            "optimizer_step": step,
            "rank": 0,
            "sample": sample,
            "sample_role": source.get("sample_role", "unknown"),
            "noise_seed": _seed("noise", step, sample),
            "timestep_seed": _seed("timestep", step, sample),
            "negative_family": negative_family_for_step(step),
        })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v8-replay", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = [json.loads(line) for line in args.v8_replay.read_text().splitlines() if line]
    audit = [json.loads(line) for line in args.audit_manifest.read_text().splitlines() if line]
    rows = build(source, audit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f".{args.output.name}.{os.getpid()}.partial")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, args.output)
    finally:
        if partial.exists():
            partial.unlink()
    print(f"rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
