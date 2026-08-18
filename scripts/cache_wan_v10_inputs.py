#!/usr/bin/env python3
"""Plan or build only the missing offline inputs for Wan v10."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from worldarena_baseline.wan_v10_data import build_v10_cache_extension
from worldarena_baseline.wan_v10_data import (
    build_v10_relation_features,
    fit_v10_relation_normalization,
    relation_features_from_v9_transition,
    write_v10_relation_cache_atomic,
)


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


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v10 JSON receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"v10 JSON receipt must be an object: {path}")
    return value


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


def _sha(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _v9_features(
    path: Path, *, sample: str, variant: str, receipt: dict[str, Any],
    exact_states: np.ndarray, exact_present: np.ndarray,
):
    from worldarena_baseline.wan_v9_transition import validate_transition_cache

    payload = validate_transition_cache(
        path, expected_sample=sample, expected_variant=variant,
        expected_clean_manifest_sha256=str(receipt["clean_manifest_sha256"]),
        expected_normalization_receipt_sha256=str(receipt["normalization_receipt_sha256"]),
    )
    features = relation_features_from_v9_transition(
        {name: np.asarray(payload[name]) for name in (
            "translation", "rotation", "image", "gripper", "arm_present", "motion_active"
        )},
        exact_states=exact_states,
        exact_state_present=exact_present,
    )
    return features, {
        name: str(np.asarray(payload[name]).item())
        for name in ("source_hdf5_sha256", "source_action_sha256", "urdf_sha256")
    }


def _source_actions(row: dict[str, Any], dataset_root: Path):
    import h5py
    from worldarena_baseline.skeleton import CameraCalibration

    hdf5_path = (dataset_root / str(row["hdf5"])).resolve(strict=True)
    if dataset_root.resolve() not in hdf5_path.parents:
        raise ValueError("v10 HDF5 escapes dataset root")
    with h5py.File(hdf5_path, "r") as handle:
        actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
        intrinsic = np.asarray(handle["observation/head_camera/intrinsic_cv"], dtype=np.float64)
        extrinsic = np.asarray(handle["observation/head_camera/extrinsic_cv"], dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 2:
        raise ValueError("v10 joint action must have shape (T>=2,14)")
    if not np.allclose(intrinsic, intrinsic[:1]) or not np.allclose(extrinsic, extrinsic[:1]):
        raise ValueError("v10 does not permit time-varying calibration")
    return actions, CameraCalibration(intrinsic[0], extrinsic[0], source_hash=_sha(hdf5_path)), hdf5_path


def _fresh_features(actions, camera, renderer, raster=None):
    from worldarena_baseline.action_condition import EpisodeTimeline
    from worldarena_baseline.action_raster import rasterize_v3_action
    from worldarena_baseline.wan_se3_condition import build_se3_condition
    from worldarena_baseline.wan_v9_transition import physical_states_from_normalized_inverse

    if raster is None:
        raster = rasterize_v3_action(
            renderer.project_actions(actions, num_frames=81, camera_calibration=camera)
        ).astype(np.float32)
    left, right = renderer.fk_endposes(actions)
    condition = build_se3_condition(
        left, right, EpisodeTimeline.build(source_length=len(actions), num_frames=81)
    )
    states = physical_states_from_normalized_inverse(
        condition.arm_transform, condition.arm_present, condition.motion_scale
    )
    return build_v10_relation_features(states, condition.arm_present, raster)


def _states_from_actions(actions, renderer):
    from worldarena_baseline.action_condition import EpisodeTimeline
    from worldarena_baseline.wan_se3_condition import build_se3_condition
    from worldarena_baseline.wan_v9_transition import physical_states_from_normalized_inverse

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


def build_relations(args: argparse.Namespace) -> int:
    from worldarena_baseline.skeleton import AlohaSkeletonRenderer
    from worldarena_baseline.wan_v8_counterfactual import build_joint14_counterfactual

    optimizer = _rows(args.optimizer_manifest)
    audit = _rows(args.audit_manifest)
    rows = optimizer + audit
    if len({str(row["sample"]) for row in rows}) != len(rows):
        raise ValueError("v10 optimizer/audit identities overlap")
    renderer = AlohaSkeletonRenderer(args.urdf, width=80, height=60)
    urdf_sha = _sha(args.urdf)
    legacy_receipt = _read_json(args.v9_transition_root / "receipt.json")
    feature_sets: dict[str, dict[str, Any]] = {}
    provenance: dict[str, tuple[str, str]] = {}
    for index, row in enumerate(rows, start=1):
        sample = str(row["sample"])
        legacy = args.v9_transition_root / sample
        variants = None
        action_path = Path(str(row["action_raster_path"])).resolve(strict=True)
        _formal(action_path)
        actions, camera, hdf5_path = _source_actions(row, args.dataset_root)
        current_provenance = {
            "source_hdf5_sha256": _sha(hdf5_path),
            "source_action_sha256": _sha(action_path),
            "urdf_sha256": urdf_sha,
        }
        if all((legacy / f"{variant}.npz").is_file() for variant in ("correct", "reverse", "swap")):
            try:
                action_variants = {
                    "correct": actions,
                    "reverse": build_joint14_counterfactual(actions, "reverse"),
                    "swap": build_joint14_counterfactual(actions, "swap"),
                }
                exact = {
                    variant: _states_from_actions(value, renderer)
                    for variant, value in action_variants.items()
                }
                loaded = {
                    variant: _v9_features(
                        legacy / f"{variant}.npz", sample=sample, variant=variant,
                        receipt=legacy_receipt, exact_states=exact[variant][0],
                        exact_present=exact[variant][1],
                    ) for variant in ("correct", "reverse", "swap")
                }
                if any(provenance != current_provenance for _features, provenance in loaded.values()):
                    raise ValueError("v9 transition provenance differs from current v10 source")
                variants = {variant: loaded[variant][0] for variant in loaded}
            except ValueError:
                variants = None
        if variants is None:
            with np.load(action_path, allow_pickle=False) as archive:
                correct_raster = np.asarray(archive["raster"], dtype=np.float32)
            variants = {
                "correct": _fresh_features(actions, camera, renderer, correct_raster),
                "reverse": _fresh_features(
                    build_joint14_counterfactual(actions, "reverse"), camera, renderer
                ),
                "swap": _fresh_features(
                    build_joint14_counterfactual(actions, "swap"), camera, renderer
                ),
            }
        feature_sets[sample] = variants
        provenance[sample] = (
            current_provenance["source_hdf5_sha256"],
            current_provenance["source_action_sha256"],
        )
        if index % 100 == 0:
            print(json.dumps({"event": "v10_relation_load", "completed": index, "total": len(rows)}), flush=True)
    normalization = fit_v10_relation_normalization(
        {str(row["sample"]): feature_sets[str(row["sample"])]["correct"] for row in optimizer}
    )
    normalization_path = args.output_root / "normalization.json"
    normalization_path.parent.mkdir(parents=True, exist_ok=True)
    partial = normalization_path.with_name(f".{normalization_path.name}.{os.getpid()}.partial")
    partial.write_text(json.dumps(normalization, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, normalization_path)
    normalization_sha = str(normalization["receipt_sha256"])
    cache_root = args.output_root / "relations"
    for index, row in enumerate(rows, start=1):
        sample = str(row["sample"])
        hdf5_sha, action_sha = provenance[sample]
        write_v10_relation_cache_atomic(
            cache_root / f"{sample}.npz",
            sample=sample,
            variants=feature_sets[sample],
            source_hdf5_sha256=hdf5_sha,
            source_action_sha256=action_sha,
            source_urdf_sha256=urdf_sha,
            normalization_sha256=normalization_sha,
        )
        if index % 100 == 0:
            print(json.dumps({"event": "v10_relation_write", "completed": index, "total": len(rows)}), flush=True)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "relations"), required=True)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--cached-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--optimizer-manifest", type=Path)
    parser.add_argument("--audit-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--v9-transition-root", type=Path)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    required = (
        (args.source_manifest, args.cached_manifest, args.output)
        if args.mode == "plan"
        else (
            args.optimizer_manifest, args.audit_manifest, args.dataset_root,
            args.v9_transition_root, args.output_root,
        )
    )
    if any(path is None for path in required) or (args.mode == "relations" and args.urdf is None):
        parser.error(f"--mode {args.mode} lacks required paths")
    for path in required:
        assert path is not None
        _formal(path)
    if args.mode == "relations" and (not args.urdf.is_file() or args.urdf.is_symlink()):
        parser.error("--urdf must be a regular file")
    return args


def main() -> None:
    args = parse_args()
    if args.mode == "plan":
        count = plan_extension(args.source_manifest, args.cached_manifest, args.output)
        event = "v10_cache_extension"
    else:
        count = build_relations(args)
        event = "v10_relation_cache"
    print(json.dumps({"event": event, "rows": count}, sort_keys=True))


if __name__ == "__main__":
    main()
