#!/usr/bin/env python3
"""Build provenance-bound raw-pose SE(3) sidecars for the clean-1000 pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any
from dataclasses import dataclass

import h5py
import numpy as np

from worldarena_baseline.action_condition import EpisodeTimeline
from worldarena_baseline.dataset_leakage import validate_training_manifest_receipt
from worldarena_baseline.wan_action_video_contract import (
    PROJECT_ARTIFACT_ROOT,
    resolve_project_video_output,
)
from worldarena_baseline.wan_data_scaling import validate_cached_manifest_identity
from worldarena_baseline.wan_se3_condition import (
    build_se3_condition,
    validate_se3_cache,
    write_se3_cache_atomic,
)
from worldarena_baseline.wan_v7_lineage import (
    derive_discovery8,
    discovery8_contract_from_pin,
    render_discovery8_jsonl,
)


FORMAL_ROOT = PROJECT_ARTIFACT_ROOT
FORMAL_SOURCE_ROOT = Path("/home/huazhi/nlh/baseline")
FORMAL_CLEAN1000_MANIFEST = (
    FORMAL_ROOT / "data_selection/v4-clean-scale-20260817-r3/clean-1000.jsonl"
)
FORMAL_LEAKAGE_RECEIPT = (
    FORMAL_ROOT / "data_selection/v4-clean-scale-20260817-r3/clean-data-scale-receipt.json"
)
FORMAL_DISCOVERY_MANIFEST = FORMAL_ROOT / "eval/v7-se3-discovery-8/discovery-8.jsonl"
FORMAL_DEV_FAST20_MANIFEST = FORMAL_ROOT / "eval/dev-fast-20-v3/dev-fast-20.jsonl"
FORMAL_TRUSTED_LINEAGE_PINS = (
    FORMAL_SOURCE_ROOT / "source_inputs/trusted-wan-v7-se3-lineage-pins.json"
)
_LINEAGE_PIN_SCHEMA = "wan-action-v7-se3-lineage-pins/3"
_SHA256 = re.compile(r"[0-9a-f]{64}")
TRUSTED_LINEAGE_PINS_SHA256 = "50ba7efe44a6232962a55b90c9b3fca02aea839a1565e59cc7c9b396e420f6c8"
_REQUIRED_HASHED_LINEAGE_ARTIFACTS = (
    "clean1000_manifest",
    "data_leakage_receipt",
    "dev_fast20_manifest",
)


@dataclass(frozen=True)
class _LineageAuthority:
    """Private seam for tests; production authority always uses fixed paths."""

    clean1000_manifest: Path
    data_leakage_receipt: Path
    discovery_manifest: Path
    dev_fast20_manifest: Path
    trusted_pins: Path
    expected_pins_sha256: str


def require_formal_cache_root(cache_root: Path | str) -> Path:
    """Return a symlink-safe cache root beneath the fixed project artifact root."""
    requested = Path(os.path.abspath(os.fspath(cache_root)))
    try:
        sentinel, _ = resolve_project_video_output(
            requested / ".wan-v7-se3-cache-root-boundary",
            artifact_root=FORMAL_ROOT,
        )
    except ValueError as exc:
        raise ValueError(
            f"cache root must be beneath project artifact root: {FORMAL_ROOT}"
        ) from exc
    return sentinel.parent


def _formal_output_path(path: Path) -> Path:
    try:
        output, _ = resolve_project_video_output(path, artifact_root=FORMAL_ROOT)
    except ValueError as exc:
        raise ValueError(
            f"cache output must be beneath project artifact root: {FORMAL_ROOT}"
        ) from exc
    return output


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"manifest rows must be JSON objects: {path}")
    return rows


def _load_trusted_lineage_pins(
    authority: _LineageAuthority,
) -> tuple[dict[str, str], dict[str, object]]:
    if _SHA256.fullmatch(authority.expected_pins_sha256) is None:
        raise ValueError("trusted lineage pins hash must come from an independent upstream")
    if _sha256_file(authority.trusted_pins) != authority.expected_pins_sha256:
        raise ValueError("trusted lineage pins differ from independent upstream hash")
    try:
        payload = json.loads(authority.trusted_pins.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("trusted lineage pins are unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "artifacts"}:
        raise ValueError("trusted lineage pins have an invalid schema")
    if payload["schema"] != _LINEAGE_PIN_SCHEMA:
        raise ValueError("trusted lineage pins contract differs")
    artifacts = payload["artifacts"]
    required = {
        *_REQUIRED_HASHED_LINEAGE_ARTIFACTS,
        "discovery_manifest",
        "official_test_manifest",
    }
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError("trusted lineage pins have an invalid artifact schema")
    hashes: dict[str, str] = {}
    for name in _REQUIRED_HASHED_LINEAGE_ARTIFACTS:
        pin = artifacts[name]
        if (
            not isinstance(pin, dict)
            or set(pin) != {"sha256"}
            or not isinstance(pin["sha256"], str)
            or _SHA256.fullmatch(pin["sha256"]) is None
        ):
            raise ValueError(f"trusted lineage {name} pin must contain a lowercase SHA-256 value")
        hashes[name] = pin["sha256"]
    if artifacts["official_test_manifest"] != {"status": "unavailable"}:
        raise ValueError("official test must remain unavailable and excluded from Stage-A")
    discovery_contract = discovery8_contract_from_pin(
        artifacts["discovery_manifest"],
        clean1000_manifest_sha256=hashes["clean1000_manifest"],
    )
    return hashes, discovery_contract


def _validate_lineage(authority: _LineageAuthority) -> dict[str, object]:
    """Validate fixed lineage files against an independently authenticated pin file."""
    pins, discovery_contract = _load_trusted_lineage_pins(authority)
    artifacts = {
        "clean1000_manifest": authority.clean1000_manifest,
        "data_leakage_receipt": authority.data_leakage_receipt,
        "dev_fast20_manifest": authority.dev_fast20_manifest,
    }
    for name, path in artifacts.items():
        if _sha256_file(path) != pins[name]:
            raise ValueError(f"trusted lineage pin differs for {name}")
    validate_cached_manifest_identity(
        authority.clean1000_manifest, authority.clean1000_manifest, expected_rows=1000
    )
    clean1000_rows = _read_jsonl(authority.clean1000_manifest)
    discovery_rows = derive_discovery8(clean1000_rows, discovery_contract)
    dev_fast20_rows = _read_jsonl(authority.dev_fast20_manifest)
    report = validate_training_manifest_receipt(
        authority.clean1000_manifest,
        authority.data_leakage_receipt,
        {
            "dev-fast20": dev_fast20_rows,
        },
    )
    return {**report, "discovery_rows": discovery_rows}


def _validate_lineage_for_testing(authority: _LineageAuthority) -> dict[str, object]:
    """Explicit private testing seam; production code never accepts this authority."""
    return _validate_lineage(authority)


def _production_lineage_authority() -> _LineageAuthority:
    return _LineageAuthority(
        clean1000_manifest=FORMAL_CLEAN1000_MANIFEST,
        data_leakage_receipt=FORMAL_LEAKAGE_RECEIPT,
        discovery_manifest=FORMAL_DISCOVERY_MANIFEST,
        dev_fast20_manifest=FORMAL_DEV_FAST20_MANIFEST,
        trusted_pins=FORMAL_TRUSTED_LINEAGE_PINS,
        expected_pins_sha256=TRUSTED_LINEAGE_PINS_SHA256,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {relative}") from exc
    return path


def _presence_from_row(row: dict[str, Any], name: str, source_length: int) -> np.ndarray | None:
    value = row.get(f"{name}_present")
    if value is None:
        return None
    values = np.asarray(value)
    if values.shape == () and values.dtype == np.dtype(bool):
        return np.full(source_length, bool(values.item()), dtype=bool)
    if values.shape != (source_length,) or values.dtype != np.dtype(bool):
        raise ValueError(f"{name}_present must be a bool or {source_length} booleans")
    return values


def _quarantine_sidecar(path: Path, *, cache_root: Path, sample: str) -> Path:
    cache_root = require_formal_cache_root(cache_root)
    path = _formal_output_path(path)
    digest = _sha256_file(path) if path.is_file() and not path.is_symlink() else "unreadable"
    root = cache_root / ".quarantine" / "v7-se3"
    _formal_output_path(root / ".wan-v7-se3-quarantine-boundary")
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{sample}-{digest[:16]}"
    suffix = 1
    while destination.exists():
        destination = root / f"{sample}-{digest[:16]}-{suffix}"
        suffix += 1
    _formal_output_path(destination / path.name)
    destination.mkdir()
    os.replace(path, destination / path.name)
    return destination


def cache_manifest_row(
    row: dict[str, Any],
    *,
    dataset_root: Path,
    cache_root: Path,
    source_manifest_sha256: str,
) -> dict[str, Any]:
    """Cache one raw HDF5 episode and rebuild only its corrupt sidecar."""
    cache_root = require_formal_cache_root(cache_root)
    sample = row.get("sample")
    hdf5_relative = row.get("hdf5")
    if not isinstance(sample, str) or not sample or not isinstance(hdf5_relative, str):
        raise ValueError("clean-1000 row requires non-empty sample and hdf5 fields")
    hdf5_path = _resolve_relative(dataset_root, hdf5_relative)
    relative = f"wan_v7_se3_conditions/{sample}.npz"
    sidecar = _formal_output_path(cache_root / relative)
    episode_sha256 = _sha256_file(hdf5_path)
    if sidecar.exists() or sidecar.is_symlink():
        try:
            cached = validate_se3_cache(sidecar, source_manifest_sha256)
            if str(cached["source_episode_sha256"].item()) != episode_sha256:
                raise ValueError("cached source episode hash differs")
            return {**row, "se3_condition": relative}
        except ValueError:
            _quarantine_sidecar(sidecar, cache_root=cache_root, sample=sample)

    with h5py.File(hdf5_path, "r") as handle:
        try:
            left = np.asarray(handle["endpose/left_endpose"], dtype=np.float64)
            right = np.asarray(handle["endpose/right_endpose"], dtype=np.float64)
        except KeyError as exc:
            raise ValueError(f"missing raw endpose field: {hdf5_path}") from exc
    if left.shape != right.shape or left.ndim != 2 or left.shape[1] != 7 or len(left) < 1:
        raise ValueError(f"raw endpose trajectories must share non-empty shape (T,7): {hdf5_path}")
    timeline = EpisodeTimeline.build(source_length=len(left), num_frames=81)
    condition = build_se3_condition(
        left,
        right,
        timeline,
        left_present=_presence_from_row(row, "left", len(left)),
        right_present=_presence_from_row(row, "right", len(right)),
    )
    write_se3_cache_atomic(
        sidecar,
        condition,
        source_episode_sha256=episode_sha256,
        source_manifest_sha256=source_manifest_sha256,
    )
    return {**row, "se3_condition": relative}


def _write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path = _formal_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def _write_discovery_atomic(path: Path, rows: list[dict[str, object]]) -> None:
    """Publish the exact source-pinned discovery-8 selection before caching."""

    path = _formal_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("wb") as handle:
            handle.write(render_discovery8_jsonl(rows))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.worker_count <= 0 or not 0 <= args.worker_index < args.worker_count:
        raise SystemExit("worker index must be in [0, worker-count) with a positive worker count")
    cache_root = require_formal_cache_root(args.cache_root)
    authority = _production_lineage_authority()
    lineage = _validate_lineage(authority)
    _write_discovery_atomic(
        authority.discovery_manifest,
        list(lineage["discovery_rows"]),
    )
    manifest_bytes = authority.clean1000_manifest.read_bytes()
    rows = _read_jsonl(authority.clean1000_manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    output_rows = [
        cache_manifest_row(
            row,
            dataset_root=args.dataset_root,
            cache_root=cache_root,
            source_manifest_sha256=manifest_sha256,
        )
        for row in rows[args.worker_index :: args.worker_count]
    ]
    output = cache_root / "manifests" / f"wan-v7-se3-worker-{args.worker_index:02d}.jsonl"
    _write_jsonl_atomic(output, output_rows)
    print(f"wrote={output} rows={len(output_rows)}", flush=True)


if __name__ == "__main__":
    main()
