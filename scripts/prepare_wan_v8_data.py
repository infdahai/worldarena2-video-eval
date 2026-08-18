#!/usr/bin/env python3
"""CPU-only completion and freezing of Wan v8 correct-SE(3) data artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldarena_baseline.action_condition import EpisodeTimeline
from worldarena_baseline.wan_balanced_sampling import validate_strata_receipt
from worldarena_baseline.wan_se3_condition import build_se3_condition, validate_se3_cache, write_se3_cache_atomic
from worldarena_baseline.wan_v8_data import build_v8_data_receipt, validate_v8_correct_cache


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_formal(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != FORMAL_ROOT and FORMAL_ROOT not in resolved.parents:
        raise ValueError(f"v8 persistent artifact escapes formal root: {path}")
    return resolved


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL: {path}") from exc
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    _require_formal(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _require_formal(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def _relative_file(root: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"v8 row lacks {label}")
    resolved = (root / relative).resolve(strict=True)
    if root.resolve() not in resolved.parents:
        raise ValueError(f"v8 {label} escapes dataset root")
    return resolved


def _presence(row: dict[str, Any], arm: str, length: int) -> np.ndarray | None:
    value = row.get(f"{arm}_present")
    if value is None:
        return None
    array = np.asarray(value)
    if array.shape == () and array.dtype == np.dtype(bool):
        return np.full(length, bool(array.item()), dtype=bool)
    if array.shape != (length,) or array.dtype != np.dtype(bool):
        raise ValueError(f"{arm}_present has invalid shape")
    return array


def _cache_correct_sidecar(row: dict[str, Any], *, dataset_root: Path, cache_root: Path, manifest_sha256: str) -> None:
    sample = row.get("sample")
    if not isinstance(sample, str) or not sample:
        raise ValueError("v8 row lacks sample")
    hdf5_path = _relative_file(dataset_root, row.get("hdf5"), label="hdf5")
    sidecar = cache_root / "wan_v8_se3_conditions" / f"{sample}.npz"
    episode_sha = _sha256(hdf5_path)
    if sidecar.is_file() and not sidecar.is_symlink():
        try:
            cached = validate_se3_cache(sidecar, manifest_sha256)
            if str(cached["source_episode_sha256"].item()) == episode_sha:
                return
        except ValueError:
            pass
    with h5py.File(hdf5_path, "r") as handle:
        left = np.asarray(handle["endpose/left_endpose"], dtype=np.float64)
        right = np.asarray(handle["endpose/right_endpose"], dtype=np.float64)
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7 or len(left) < 1:
        raise ValueError(f"raw endpose trajectories must share shape (T,7): {hdf5_path}")
    condition = build_se3_condition(
        left,
        right,
        EpisodeTimeline.build(source_length=len(left), num_frames=81),
        left_present=_presence(row, "left", len(left)),
        right_present=_presence(row, "right", len(right)),
    )
    write_se3_cache_atomic(
        sidecar,
        condition,
        source_episode_sha256=episode_sha,
        source_manifest_sha256=manifest_sha256,
    )


def prepare(
    *,
    dataset_root: Path,
    cache_root: Path,
    cached_manifest: Path,
    probe_split_path: Path,
    dev_manifest: Path,
    strata_receipt: Path,
    strata_manifest: Path,
    output_root: Path,
    world_size: int,
    steps: int,
) -> dict[str, Any]:
    """Complete only correct SE(3), then atomically freeze v8 inputs."""
    for path in (dataset_root, cache_root, cached_manifest, probe_split_path, dev_manifest, strata_receipt, strata_manifest, output_root):
        _require_formal(path)
    rows = _read_jsonl(cached_manifest)
    if len(rows) != 1785:
        raise ValueError(f"v8 requires exactly clean-1785 cached rows, got {len(rows)}")
    manifest_digest = _sha256(cached_manifest)
    for row in rows:
        _cache_correct_sidecar(
            row, dataset_root=dataset_root, cache_root=cache_root, manifest_sha256=manifest_digest
        )
    samples = tuple(str(row["sample"]) for row in rows)
    correct = validate_v8_correct_cache(
        cache_root, expected_samples=samples, source_manifest_sha256=manifest_digest
    )
    try:
        probe_split = json.loads(probe_split_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("cannot read v6 probe split") from exc
    dev_rows = _read_jsonl(dev_manifest)
    strata = validate_strata_receipt(strata_receipt, manifest_path=strata_manifest)
    if set(strata["roles"]) != set(samples):
        raise ValueError("v8 strata identities differ from cached clean-1785 identities")
    receipt = build_v8_data_receipt(
        cached_rows=rows,
        probe_split=probe_split,
        dev_rows=dev_rows,
        sample_roles=strata["roles"],
        steps=steps,
        world_size=world_size,
    )
    _write_jsonl_atomic(output_root / "audit20.jsonl", [row for row in rows if row["sample"] in set(receipt.audit_samples)])
    _write_jsonl_atomic(output_root / "replay250.jsonl", list(receipt.replay))
    payload = {
        "contract": "wan-v8-direct-action-band-data/1",
        "cached_manifest": str(cached_manifest),
        "cached_manifest_sha256": manifest_digest,
        "probe_split": str(probe_split_path),
        "probe_split_sha256": _sha256(probe_split_path),
        "dev_manifest": str(dev_manifest),
        "dev_manifest_sha256": _sha256(dev_manifest),
        "strata_receipt": str(strata_receipt),
        "strata_receipt_sha256": _sha256(strata_receipt),
        "strata_manifest": str(strata_manifest),
        "strata_manifest_sha256": _sha256(strata_manifest),
        "audit_samples": list(receipt.audit_samples),
        "optimizer_sample_count": len(receipt.optimizer_samples),
        "replay_rows": len(receipt.replay),
        "world_size": world_size,
        "steps": steps,
        "correct_cache": correct,
    }
    _write_json_atomic(output_root / "receipt.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--cached-manifest", type=Path, required=True)
    parser.add_argument("--probe-split", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--strata-receipt", type=Path, required=True)
    parser.add_argument("--strata-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = prepare(**vars(args))
    print(json.dumps(payload, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
