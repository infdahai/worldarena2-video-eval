from __future__ import annotations

import json
import os
import hashlib
import re
from dataclasses import asdict
from pathlib import Path
from typing import Protocol

import h5py
import numpy as np

from .action_audit import evaluate_progression_gate
from .action_condition import causal_time_groups, EpisodeTimeline, encode_endpose_condition
from .action_raster import rasterize_v3_action
from .robotwin_manifest import RobotwinTrainingEpisode
from .skeleton import CameraCalibration, ProjectedArmTrajectory


V3_POSE_FEATURE_NAMES = (
    "dx_cam",
    "dy_cam",
    "dz_cam",
    "abs_z_cam",
    "rel_rot6d_0",
    "rel_rot6d_1",
    "rel_rot6d_2",
    "rel_rot6d_3",
    "rel_rot6d_4",
    "rel_rot6d_5",
    "gripper",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
_V3_CACHE_KEYS = {
    "raster", "left_pose", "right_pose", "condition_support", "loss_weight",
    "schema_version", "pose_statistics_sha256",
}


def validate_v3_condition_payload(payload: object, expected_statistics_hash: str) -> None:
    files = set(payload.files) if hasattr(payload, "files") else set(payload)  # type: ignore[arg-type]
    if files != _V3_CACHE_KEYS:
        raise ValueError("cached condition has invalid schema-v3 keys")
    def value(key: str) -> np.ndarray:
        return np.asarray(payload[key])  # type: ignore[index]
    schema = value("schema_version")
    if schema.dtype != np.dtype(np.int64) or schema.shape != () or int(schema) != 3:
        raise ValueError("cached condition is not schema v3")
    raster = value("raster")
    poses = (value("left_pose"), value("right_pose"))
    support, weight = value("condition_support"), value("loss_weight")
    if raster.shape != (81, 10, 60, 80) or raster.dtype != np.dtype(np.float16):
        raise ValueError("cached raster has invalid v3 shape or dtype")
    if any(pose.shape != (81, 11) or pose.dtype != np.dtype(np.float16) for pose in poses):
        raise ValueError("cached pose has invalid v3 shape or dtype")
    if support.shape != (2, 21, 15, 20) or support.dtype != np.dtype(np.float16):
        raise ValueError("cached support has invalid v3 shape or dtype")
    if weight.shape != (1, 21, 30, 40) or weight.dtype != np.dtype(np.float16):
        raise ValueError("cached loss weight has invalid v3 shape or dtype")
    if not all(np.isfinite(array).all() for array in (raster, *poses, support, weight)):
        raise ValueError("cached condition contains non-finite values")
    if not (np.all((raster[:, [0, 1, 4, 5, 6, 9]] >= 0) & (raster[:, [0, 1, 4, 5, 6, 9]] <= 1))
            and np.all((raster[:, [2, 3, 7, 8]] >= -1) & (raster[:, [2, 3, 7, 8]] <= 1))
            and np.all((support >= 0) & (support <= 1)) and np.all(weight > 0)):
        raise ValueError("cached condition has invalid v3 ranges")
    expected_support, expected_weight = _v3_support_and_loss_weight(raster)
    if not np.array_equal(support, expected_support.astype(np.float16)):
        raise ValueError("cached support does not match raster")
    if not np.array_equal(weight, expected_weight.astype(np.float16)):
        raise ValueError("cached loss weight does not match raster")
    statistics_hash = value("pose_statistics_sha256")
    if (
        statistics_hash.shape != ()
        or statistics_hash.dtype.kind != "U"
        or re.fullmatch(r"[0-9a-f]{64}", str(statistics_hash.item())) is None
    ):
        raise ValueError("cached pose statistics hash is invalid")
    if str(statistics_hash.item()) != expected_statistics_hash:
        raise ValueError("cached pose statistics hash differs")


# Kept for the producer's existing internal call sites and old callers.
_validate_v3_condition_payload = validate_v3_condition_payload


class ActionProjector(Protocol):
    height: int
    width: int

    def project_actions(
        self,
        actions: np.ndarray,
        *,
        num_frames: int,
        camera_calibration: CameraCalibration | None = None,
    ) -> ProjectedArmTrajectory: ...


def _resolve_relative(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"path escapes dataset root: {relative}") from exc
    return path


def _write_condition_atomic(
    path: Path,
    *,
    raster: np.ndarray,
    left_pose: np.ndarray,
    right_pose: np.ndarray,
    condition_support: np.ndarray,
    loss_weight: np.ndarray,
    pose_statistics_sha256: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.stem}.partial.npz")
    stored_raster = raster.astype(np.float16)
    stored_support, stored_weight = _v3_support_and_loss_weight(stored_raster)
    payload = {
        "raster": stored_raster, "left_pose": left_pose.astype(np.float16),
        "right_pose": right_pose.astype(np.float16),
        "condition_support": stored_support.astype(np.float16),
        "loss_weight": stored_weight.astype(np.float16),
        "schema_version": np.asarray(3, dtype=np.int64),
        "pose_statistics_sha256": np.asarray(pose_statistics_sha256),
    }
    _validate_v3_condition_payload(payload, pose_statistics_sha256)
    np.savez_compressed(partial, **payload)
    os.replace(partial, path)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(partial, path)


def _quarantine_action_pair(
    raster_path: Path, metrics_path: Path, *, cache_root: Path, sample: str
) -> Path | None:
    existing = [path for path in (raster_path, metrics_path) if path.exists()]
    if not existing:
        return None
    digest = hashlib.sha256()
    for path in existing:
        digest.update(path.name.encode())
        digest.update(_sha256_file(path).encode())
    quarantine_root = cache_root / ".quarantine/action"
    quarantine_root.mkdir(parents=True, exist_ok=True)
    suffix = 0
    while True:
        tail = f"-{suffix}" if suffix else ""
        final = quarantine_root / f"{sample}-{digest.hexdigest()[:16]}{tail}"
        partial = quarantine_root / f".{final.name}.partial"
        if not final.exists() and not partial.exists():
            break
        suffix += 1
    partial.mkdir()
    for path in existing:
        os.replace(path, partial / path.name)
    os.replace(partial, final)
    return final


def _causal_max_pool(
    values: np.ndarray,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    frames = np.asarray(values, dtype=np.float32)
    if frames.shape != (81, 60, 80):
        raise ValueError("v3 values must have shape (81, 60, 80)")
    if 60 % output_height or 80 % output_width:
        raise ValueError("v3 spatial pooling dimensions must divide 60x80")
    groups = causal_time_groups()
    pooled = np.empty((len(groups), output_height, output_width), dtype=np.float32)
    for output_index, group in enumerate(groups):
        frame = frames[list(group)].max(axis=0)
        pooled[output_index] = frame.reshape(
            output_height,
            60 // output_height,
            output_width,
            80 // output_width,
        ).max(axis=(1, 3))
    return pooled


def _v3_support_and_loss_weight(raster: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(raster, dtype=np.float32)
    if values.shape != (81, 10, 60, 80):
        raise ValueError("v3 raster must have shape (81, 10, 60, 80)")
    condition_support = np.stack(
        [
            _causal_max_pool(values[:, 0], output_height=15, output_width=20),
            _causal_max_pool(values[:, 5], output_height=15, output_width=20),
        ]
    )
    occupancy = np.maximum(values[:, 0], values[:, 5])
    eef = np.maximum(values[:, 1], values[:, 6])
    loss_weight = 1.0 + occupancy + 2.0 * eef
    return condition_support, _causal_max_pool(
        loss_weight, output_height=30, output_width=40
    )[None]


def _load_pose_statistics(
    path: Path | str, expected_train40_manifest_sha256: str
) -> tuple[np.ndarray, np.ndarray, str]:
    statistics_path = Path(path)
    raw = statistics_path.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid pose statistics JSON: {statistics_path}") from exc
    if payload.get("schema_version") != 3:
        raise ValueError(f"pose statistics is not schema v3: {statistics_path}")
    if tuple(payload.get("feature_names", ())) != V3_POSE_FEATURE_NAMES:
        raise ValueError(f"pose statistics feature names are invalid: {statistics_path}")
    mean = np.asarray(payload.get("mean"), dtype=np.float32)
    std = np.asarray(payload.get("std"), dtype=np.float32)
    if mean.shape != (11,) or std.shape != (11,) or not np.isfinite(mean).all():
        raise ValueError(f"pose statistics vectors are invalid: {statistics_path}")
    if not np.isfinite(std).all() or np.any(std < 1e-6):
        raise ValueError(f"pose statistics std is invalid: {statistics_path}")
    provenance = ("episode_count", "task_count", "source_manifest_sha256")
    if any(key not in payload for key in provenance):
        raise ValueError(f"pose statistics provenance is missing: {statistics_path}")
    if (
        type(payload["episode_count"]) is not int or payload["episode_count"] <= 0
        or type(payload["task_count"]) is not int or payload["task_count"] <= 0
        or not isinstance(payload["source_manifest_sha256"], str)
        or re.fullmatch(r"[0-9a-f]{64}", payload["source_manifest_sha256"]) is None
        or payload["source_manifest_sha256"] != expected_train40_manifest_sha256
    ):
        raise ValueError(f"pose statistics provenance is invalid: {statistics_path}")
    return mean, std, hashlib.sha256(raw).hexdigest()


def build_train40_pose_statistics(
    source_manifest: Path | str,
    *,
    dataset_root: Path | str,
    output: Path | str,
    expected_train40_manifest_sha256: str,
) -> Path:
    """Write schema-v3 camera-frame pose statistics from the train-40 manifest."""
    manifest_path = Path(source_manifest)
    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if expected_train40_manifest_sha256 != actual_manifest_sha256:
        raise ValueError("source manifest does not match expected internal train-40 hash")
    rows = [
        json.loads(line)
        for line in manifest_bytes.decode("utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("train-40 pose statistics manifest is empty")
    all_poses: list[np.ndarray] = []
    tasks: set[str] = set()
    root = Path(dataset_root)
    for row in rows:
        hdf5_path = _resolve_relative(root, row["hdf5"])
        with h5py.File(hdf5_path, "r") as handle:
            try:
                left_endpose = np.asarray(handle["endpose/left_endpose"], dtype=np.float64)
                right_endpose = np.asarray(handle["endpose/right_endpose"], dtype=np.float64)
                left_gripper = np.asarray(handle["joint_action/left_gripper"], dtype=np.float64)
                right_gripper = np.asarray(handle["joint_action/right_gripper"], dtype=np.float64)
                extrinsic = np.asarray(
                    handle["observation/head_camera/extrinsic_cv"], dtype=np.float64
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing pose statistics field: {hdf5_path}"
                ) from exc
        source_length = len(left_endpose)
        if source_length < 1 or any(
            len(values) != source_length
            for values in (right_endpose, left_gripper, right_gripper, extrinsic)
        ):
            raise ValueError(f"pose statistics lengths differ for {hdf5_path}")
        if extrinsic.shape[1:] != (3, 4) or not np.allclose(extrinsic, extrinsic[:1]):
            raise ValueError(f"head-camera calibration changes over time: {hdf5_path}")
        timeline = EpisodeTimeline.build(source_length=source_length, num_frames=81)
        all_poses.extend(
            [
                encode_endpose_condition(
                    left_endpose,
                    left_gripper,
                    timeline,
                    camera_extrinsic_cv=extrinsic[0],
                ),
                encode_endpose_condition(
                    right_endpose,
                    right_gripper,
                    timeline,
                    camera_extrinsic_cv=extrinsic[0],
                ),
            ]
        )
        tasks.add(str(row.get("task", "")))
    features = np.concatenate(all_poses, axis=0)
    payload = {
        "schema_version": 3,
        "feature_names": list(V3_POSE_FEATURE_NAMES),
        "mean": np.mean(features, axis=0, dtype=np.float64).tolist(),
        "std": np.maximum(np.std(features, axis=0, dtype=np.float64), 1e-6).tolist(),
        "episode_count": len(rows),
        "task_count": len(tasks),
        "source_manifest_sha256": actual_manifest_sha256,
    }
    result = Path(output)
    _write_json_atomic(result, payload)
    return result


def cache_episode_action(
    episode: RobotwinTrainingEpisode,
    *,
    dataset_root: Path | str,
    cache_root: Path | str,
    renderer: ActionProjector,
    pose_statistics_path: Path | str,
    expected_train40_manifest_sha256: str,
    source_manifest_sha256: str | None = None,
    urdf_sha256: str | None = None,
    num_frames: int = 81,
) -> dict:
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)
    if num_frames != 81:
        raise ValueError("schema v3 requires exactly 81 raster frames")
    if (renderer.height, renderer.width) != (60, 80):
        raise ValueError("schema v3 renderer must render directly at 80x60")
    pose_mean, pose_std, pose_statistics_sha256 = _load_pose_statistics(
        pose_statistics_path, expected_train40_manifest_sha256
    )
    raster_relative = f"action_rasters_v3/{episode.sample}.npz"
    metrics_relative = f"action_metrics_v3/{episode.sample}.json"
    raster_path = cache_root / raster_relative
    metrics_path = cache_root / metrics_relative
    row = {
        **asdict(episode),
        "action_raster": raster_relative,
        "action_metrics": metrics_relative,
    }
    source_binding = {}
    if source_manifest_sha256 is not None:
        if re.fullmatch(r"[0-9a-f]{64}", source_manifest_sha256) is None:
            raise ValueError("source manifest hash is invalid")
        hdf5_path = _resolve_relative(dataset_root, episode.hdf5)
        instruction_path = _resolve_relative(dataset_root, episode.instruction)
        if urdf_sha256 is None or re.fullmatch(r"[0-9a-f]{64}", urdf_sha256) is None:
            raise ValueError("formal action cache requires a valid URDF hash")
        source_binding = {
            "source_manifest_sha256": source_manifest_sha256,
            "source_row_sha256": hashlib.sha256(
                json.dumps(asdict(episode), sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "source_hdf5_sha256": _sha256_file(hdf5_path),
            "source_instruction_sha256": _sha256_file(instruction_path),
            "pose_statistics_sha256": pose_statistics_sha256,
            "urdf_sha256": urdf_sha256,
        }
    formal_binding = source_manifest_sha256 is not None
    if raster_path.is_file() != metrics_path.is_file() and formal_binding:
        _quarantine_action_pair(
            raster_path, metrics_path, cache_root=cache_root, sample=episode.sample
        )
    if raster_path.is_file() and metrics_path.is_file():
        try:
            with np.load(raster_path) as payload:
                _validate_v3_condition_payload(payload, pose_statistics_sha256)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            expected_binding = {
                **source_binding,
                "action_raster_sha256": _sha256_file(raster_path),
            }
            if any(metrics.get(key) != value for key, value in expected_binding.items()):
                raise ValueError(
                    f"cached action provenance differs for {episode.sample}"
                )
            return row
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            if not formal_binding:
                raise
            _quarantine_action_pair(
                raster_path,
                metrics_path,
                cache_root=cache_root,
                sample=episode.sample,
            )

    hdf5_path = _resolve_relative(dataset_root, episode.hdf5)
    with h5py.File(hdf5_path, "r") as handle:
        try:
            actions = np.asarray(handle["joint_action/vector"], dtype=np.float64)
            left_endpose = np.asarray(
                handle["endpose/left_endpose"], dtype=np.float64
            )
            right_endpose = np.asarray(
                handle["endpose/right_endpose"], dtype=np.float64
            )
            left_gripper = np.asarray(
                handle["joint_action/left_gripper"], dtype=np.float64
            )
            right_gripper = np.asarray(
                handle["joint_action/right_gripper"], dtype=np.float64
            )
            intrinsic = np.asarray(
                handle["observation/head_camera/intrinsic_cv"], dtype=np.float64
            )
            extrinsic = np.asarray(
                handle["observation/head_camera/extrinsic_cv"], dtype=np.float64
            )
        except KeyError as exc:
            raise ValueError(f"missing action conditioning field: {hdf5_path}") from exc
    if actions.ndim != 2 or actions.shape[1] != 14 or len(actions) < 1:
        raise ValueError(f"expected non-empty joint14 trajectory: {hdf5_path}")
    source_length = len(actions)
    expected_lengths = {
        "left_endpose": len(left_endpose),
        "right_endpose": len(right_endpose),
        "left_gripper": len(left_gripper),
        "right_gripper": len(right_gripper),
        "intrinsic": len(intrinsic),
        "extrinsic": len(extrinsic),
    }
    if any(length != source_length for length in expected_lengths.values()):
        raise ValueError(
            f"action conditioning lengths differ for {hdf5_path}: {expected_lengths}"
        )
    if not np.allclose(intrinsic, intrinsic[:1]) or not np.allclose(
        extrinsic, extrinsic[:1]
    ):
        raise ValueError(f"head-camera calibration changes over time: {hdf5_path}")

    timeline = EpisodeTimeline.build(
        source_length=source_length, num_frames=num_frames
    )
    sampled_actions = timeline.interpolate(actions)
    left_pose = encode_endpose_condition(
        left_endpose, left_gripper, timeline, camera_extrinsic_cv=extrinsic[0]
    )
    right_pose = encode_endpose_condition(
        right_endpose, right_gripper, timeline, camera_extrinsic_cv=extrinsic[0]
    )
    left_pose = (left_pose - pose_mean) / pose_std
    right_pose = (right_pose - pose_mean) / pose_std
    calibration_digest = hashlib.sha256(
        intrinsic[0].tobytes() + extrinsic[0].tobytes()
    ).hexdigest()
    camera_calibration = CameraCalibration(
        intrinsic_cv=intrinsic[0],
        extrinsic_cv=extrinsic[0],
        source_width=320,
        source_height=240,
        source_hash=calibration_digest,
    )

    trajectory = renderer.project_actions(
        sampled_actions,
        num_frames=num_frames,
        camera_calibration=camera_calibration,
    )
    raster = rasterize_v3_action(trajectory)
    condition_support, loss_weight = _v3_support_and_loss_weight(raster)
    flow = raster[:, [2, 3, 7, 8]]
    state = raster[:, [0, 1, 4, 5, 6, 9]]
    metrics = {
        "finite": bool(np.isfinite(raster).all()),
        "frame_zero_flow_zero": bool(np.count_nonzero(flow[0]) == 0),
        "heatmap_state_range_valid": bool(np.min(state) >= 0 and np.max(state) <= 1),
        "flow_range_valid": bool(np.min(flow) >= -1 and np.max(flow) <= 1),
        "left_gripper_visible_fraction": float(
            np.mean(np.any(raster[:, 1] > 0, axis=(1, 2)))
        ),
        "right_gripper_visible_fraction": float(
            np.mean(np.any(raster[:, 6] > 0, axis=(1, 2)))
        ),
    }
    metrics.update(
        {
            "sample": episode.sample,
            "task": episode.task,
            "raster_shape": list(raster.shape),
            "schema_version": 3,
            "camera_calibration_hash": calibration_digest,
            **source_binding,
        }
    )
    metrics["gate"] = evaluate_progression_gate(metrics)
    if not metrics["gate"]["hard_pass"]:
        raise ValueError(
            f"action raster failed hard gate for {episode.sample}: "
            f"{metrics['gate']['hard_failures']}"
        )
    _write_condition_atomic(
        raster_path,
        raster=raster,
        left_pose=left_pose,
        right_pose=right_pose,
        condition_support=condition_support,
        loss_weight=loss_weight,
        pose_statistics_sha256=pose_statistics_sha256,
    )
    metrics["action_raster_sha256"] = _sha256_file(raster_path)
    _write_json_atomic(metrics_path, metrics)
    return row
