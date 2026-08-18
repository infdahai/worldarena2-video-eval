#!/usr/bin/env python3
"""Build provenance-bound compact transition caches for Wan v9."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from worldarena_baseline.wan_v9_transition import (
    TransitionFeatures,
    build_transition_features,
    fit_normalization_statistics,
    physical_states_from_normalized_inverse,
    validate_split_identities,
    validate_transition_cache,
    write_transition_cache_atomic,
)


FORMAL_ROOT = Path("/data/di/worldarena2_track1_20260815")
_VARIANTS = {
    "reverse": ("reverse", "reverse", 0),
    "shift+1": ("shift_plus", "shift", 1),
    "shift-1": ("shift_minus", "shift", -1),
    "swap": ("swap", "swap", 0),
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read JSONL: {path}") from exc
    if not rows:
        raise ValueError(f"JSONL is empty: {path}")
    return rows


def _require_formal(path: Path) -> Path:
    resolved = path.resolve(strict=False)
    if resolved != FORMAL_ROOT and FORMAL_ROOT not in resolved.parents:
        raise ValueError(f"v9 persistent path escapes formal root: {path}")
    return resolved


def _load_correct(row: dict[str, Any], *, cache_root: Path, clean_sha: str) -> tuple[TransitionFeatures, Path, Path]:
    from worldarena_baseline.wan_se3_condition import validate_se3_cache

    sample = str(row["sample"])
    raster_path = cache_root / str(row.get("action_raster", f"action_rasters_v3/{sample}.npz"))
    se3_path = cache_root / "wan_v8_se3_conditions" / f"{sample}.npz"
    with np.load(raster_path, allow_pickle=False) as archive:
        raster = np.asarray(archive["raster"], dtype=np.float32)
    se3 = validate_se3_cache(se3_path, clean_sha)
    states = physical_states_from_normalized_inverse(
        se3["arm_transform"], se3["arm_present"], float(se3["motion_scale"].item())
    )
    return build_transition_features(states, np.asarray(se3["arm_present"], dtype=bool), raster), raster_path, se3_path


def _states_from_joint14(actions: np.ndarray, renderer) -> tuple[np.ndarray, np.ndarray]:
    from worldarena_baseline.action_condition import EpisodeTimeline
    from worldarena_baseline.wan_se3_condition import build_se3_condition

    left, right = renderer.fk_endposes(actions)
    condition = build_se3_condition(
        left, right, EpisodeTimeline.build(source_length=len(actions), num_frames=81)
    )
    return (
        physical_states_from_normalized_inverse(
            condition.arm_transform, condition.arm_present, condition.motion_scale
        ),
        condition.arm_present,
    )


def _joint_counterfactual(actions: np.ndarray, family: str, direction: int) -> np.ndarray:
    correct = np.asarray(actions, dtype=np.float64)
    if correct.ndim != 2 or correct.shape[1] != 14 or len(correct) < 2:
        raise ValueError("v9 counterfactual requires joint14 actions")
    anchor = correct[:1]
    if family == "reverse":
        result = anchor - (correct - anchor)
    elif family == "swap":
        result = np.repeat(anchor, len(correct), axis=0)
        result[:, :7] += correct[:, 7:14] - correct[:1, 7:14]
        result[:, 7:14] += correct[:, :7] - correct[:1, :7]
    elif family == "shift" and direction == 1:
        result = correct.copy()
        result[1:] = correct[:-1]
    elif family == "shift" and direction == -1:
        result = correct.copy()
        result[1:-1] = correct[2:]
        result[-1] = correct[-1]
    else:
        raise ValueError("v9 counterfactual family/direction is invalid")
    result[:, [6, 13]] = np.clip(result[:, [6, 13]], 0.0, 1.0)
    if not np.array_equal(result[0], correct[0]) or not np.isfinite(result).all():
        raise ValueError("v9 counterfactual changed anchor or became non-finite")
    return result


def _load_wrong(path: Path, label: str, actions: np.ndarray, renderer) -> TransitionFeatures:
    cache_label, family, direction = _VARIANTS[label]
    with np.load(path, allow_pickle=False) as archive:
        raster = np.asarray(archive[f"{cache_label}_raster"], dtype=np.float32)
        cached_present = np.asarray(archive[f"{cache_label}_arm_present"], dtype=bool)
    counterfactual = _joint_counterfactual(actions, family, direction)
    states, present = _states_from_joint14(counterfactual, renderer)
    if not np.array_equal(present, cached_present):
        raise ValueError("v9 FK presence differs from rendered counterfactual cache")
    return build_transition_features(states, present, raster)


def _joint14(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2:
        raise ValueError("v9 source joint action must have shape (T>=2,14)")
    return actions


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    try:
        with partial.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def prepare(
    *,
    clean_manifest: Path,
    audit_manifest: Path,
    dev_manifest: Path,
    output_root: Path,
    dry_run: bool,
    dataset_root: Path | None = None,
    cache_root: Path | None = None,
    negative_root: Path | None = None,
    urdf: Path | None = None,
) -> dict[str, Any]:
    clean_rows = _read_jsonl(clean_manifest)
    audit_rows = _read_jsonl(audit_manifest)
    dev_rows = _read_jsonl(dev_manifest)
    optimizer = validate_split_identities(clean_rows, audit_rows, dev_rows)
    receipt: dict[str, Any] = {
        "contract": "wan-v9-transition-preparation/1",
        "clean_manifest_sha256": _sha(clean_manifest),
        "audit_manifest_sha256": _sha(audit_manifest),
        "dev_manifest_sha256": _sha(dev_manifest),
        "optimizer_sample_count": len(optimizer),
        "audit_sample_count": len(audit_rows),
        "starts_cache_work": not dry_run,
    }
    if dry_run:
        return receipt
    if None in (dataset_root, cache_root, negative_root, urdf):
        raise ValueError("non-dry v9 transition preparation requires all source paths")
    assert dataset_root is not None and cache_root is not None and negative_root is not None and urdf is not None
    for path in (clean_manifest, audit_manifest, dev_manifest, output_root, dataset_root, cache_root, negative_root):
        _require_formal(path)
    if not urdf.is_file() or urdf.is_symlink():
        raise ValueError("v9 URDF must be a regular file")
    row_by_sample = {str(row["sample"]): row for row in clean_rows}
    correct: dict[str, tuple[TransitionFeatures, Path, Path]] = {}
    for sample in (*optimizer, *(str(row["sample"]) for row in audit_rows)):
        correct[sample] = _load_correct(row_by_sample[sample], cache_root=cache_root, clean_sha=receipt["clean_manifest_sha256"])
    statistics = fit_normalization_statistics({sample: correct[sample][0] for sample in optimizer})
    _write_json_atomic(output_root / "normalization.json", statistics)
    urdf_sha = _sha(urdf)
    from worldarena_baseline.skeleton import AlohaSkeletonRenderer

    renderer = AlohaSkeletonRenderer(urdf, width=80, height=60)
    for sample in (*optimizer, *(str(row["sample"]) for row in audit_rows)):
        row = row_by_sample[sample]
        hdf5 = (dataset_root / str(row["hdf5"])).resolve(strict=True)
        if dataset_root.resolve() not in hdf5.parents:
            raise ValueError("v9 HDF5 escapes dataset root")
        features, raster_path, _ = correct[sample]
        write_transition_cache_atomic(
            output_root / sample / "correct.npz", features,
            sample=sample, variant="correct", source_hdf5_sha256=_sha(hdf5),
            source_action_sha256=_sha(raster_path), source_counterfactual_sha256="",
            urdf_sha256=urdf_sha, clean_manifest_sha256=receipt["clean_manifest_sha256"],
            normalization_receipt_sha256=str(statistics["receipt_sha256"]),
        )
        if sample not in set(optimizer):
            continue
        negative_path = negative_root / f"{sample}.npz"
        actions = _joint14(hdf5)
        for variant in _VARIANTS:
            write_transition_cache_atomic(
                output_root / sample / f"{variant}.npz",
                _load_wrong(negative_path, variant, actions, renderer),
                sample=sample, variant=variant, source_hdf5_sha256=_sha(hdf5),
                source_action_sha256=_sha(raster_path), source_counterfactual_sha256=_sha(negative_path),
                urdf_sha256=urdf_sha, clean_manifest_sha256=receipt["clean_manifest_sha256"],
                normalization_receipt_sha256=str(statistics["receipt_sha256"]),
            )
    for sample in optimizer:
        for variant in ("correct", *_VARIANTS):
            validate_transition_cache(
                output_root / sample / f"{variant}.npz", expected_sample=sample,
                expected_variant=variant, expected_clean_manifest_sha256=receipt["clean_manifest_sha256"],
                expected_normalization_receipt_sha256=str(statistics["receipt_sha256"]),
            )
    receipt.update({"normalization_receipt_sha256": statistics["receipt_sha256"], "complete_optimizer_variants": len(optimizer) * 5})
    _write_json_atomic(output_root / "receipt.json", receipt)
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--dev-manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--negative-root", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(prepare(**vars(args)), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
