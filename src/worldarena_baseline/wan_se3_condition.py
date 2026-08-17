"""Authoritative raw-pose SE(3) condition for the bounded Wan v7 probe."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Final

import numpy as np

from .action_condition import EpisodeTimeline, _rotation_matrices_wxyz, causal_time_groups, slerp_wxyz


SE3_CACHE_SCHEMA: Final = "wan-action-v7-se3-condition/1"
TEMPORAL_CONTRACT: Final = "81-to-21-causal-v3"
_CACHE_KEYS: Final = {
    "schema",
    "arm_transform",
    "arm_present",
    "anchor_arm",
    "motion_scale",
    "source_episode_sha256",
    "source_manifest_sha256",
    "temporal_contract",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class SE3Condition:
    """The two fixed arm-group actions consumed by v7 geometric attention."""

    arm_transform: np.ndarray
    arm_present: np.ndarray
    anchor_arm: str
    motion_scale: float


def _require_raw_poses(endpose: np.ndarray, timeline: EpisodeTimeline) -> np.ndarray:
    poses = np.asarray(endpose, dtype=np.float64)
    if poses.shape != (timeline.source_length, 7):
        raise ValueError("endpose must have shape (timeline.source_length, 7)")
    if not np.isfinite(poses).all():
        raise ValueError("endpose must be finite")
    # slerp_wxyz below owns the zero-norm and sign-continuity validation.
    return poses


def _require_presence(
    present: np.ndarray | None, timeline: EpisodeTimeline
) -> np.ndarray:
    if present is None:
        return np.ones(timeline.source_length, dtype=bool)
    values = np.asarray(present)
    if values.shape != (timeline.source_length,) or values.dtype != np.dtype(bool):
        raise ValueError("arm presence must have boolean shape (timeline.source_length,)")
    return values.copy()


def _sample_raw_poses(poses: np.ndarray, timeline: EpisodeTimeline) -> np.ndarray:
    sampled = np.broadcast_to(np.eye(4, dtype=np.float64), (81, 4, 4)).copy()
    sampled[:, :3, 3] = timeline.interpolate(poses[:, :3])
    sampled[:, :3, :3] = _rotation_matrices_wxyz(
        slerp_wxyz(poses[:, 3:7], timeline)
    )
    _validate_rigid_transforms(sampled, label="sampled raw endpose")
    return sampled


def _causal_latent_transforms(sampled: np.ndarray) -> np.ndarray:
    groups = causal_time_groups()
    if sampled.shape != (81, 4, 4) or len(groups) != 21:
        raise ValueError("v7 SE(3) temporal packing requires 81 sampled frames")
    # A latent frame may only depend on visual frames it has received.  The last
    # frame of each causal group is the only rigid pose representative; matrices
    # and quaternions are never averaged.
    return sampled[[group[-1] for group in groups]].copy()


def _causal_latent_presence(present: np.ndarray, timeline: EpisodeTimeline) -> np.ndarray:
    sampled = present[timeline.rgb_indices]
    return np.asarray(
        [bool(np.all(sampled[list(group)])) for group in causal_time_groups()], dtype=bool
    )


def _invert_rigid_transforms(transforms: np.ndarray) -> np.ndarray:
    values = np.asarray(transforms, dtype=np.float64)
    _validate_rigid_transforms(values, label="rigid transform")
    result = np.broadcast_to(np.eye(4, dtype=np.float64), values.shape).copy()
    result[..., :3, :3] = values[..., :3, :3].swapaxes(-1, -2)
    result[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", result[..., :3, :3], values[..., :3, 3]
    )
    return result


def _validate_rigid_transforms(transforms: np.ndarray, *, label: str) -> None:
    values = np.asarray(transforms)
    if values.ndim < 2 or values.shape[-2:] != (4, 4):
        raise ValueError(f"{label} must end in 4x4 matrices")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} contains non-finite values")
    if not np.allclose(values[..., 3, :], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-4, rtol=0):
        raise ValueError(f"{label} has an invalid homogeneous row")
    rotation = values[..., :3, :3]
    determinant = np.linalg.det(rotation)
    if not np.all(np.abs(determinant - 1.0) <= 1e-4):
        raise ValueError(f"{label} rotation determinant is outside 1 +/- 1e-4")
    orthogonality = rotation.swapaxes(-1, -2) @ rotation
    if not np.allclose(orthogonality, np.eye(3), atol=1e-4, rtol=0):
        raise ValueError(f"{label} rotation is not orthogonal within 1e-4")


def build_se3_condition(
    left_endpose: np.ndarray | None,
    right_endpose: np.ndarray | None,
    timeline: EpisodeTimeline,
    left_present: np.ndarray | None = None,
    right_present: np.ndarray | None = None,
    *,
    epsilon: float = 1e-6,
) -> SE3Condition:
    """Build anchored, normalized inverse transforms from original EEF poses.

    The function deliberately accepts no standardized 11-D cache.  Each arm is
    sampled at 81 visual frames with translation interpolation and SLERP, then
    packed causally to Wan's 21 latent frames.
    """
    if len(timeline.target_positions) != 81:
        raise ValueError("v7 SE(3) condition requires an 81-frame EpisodeTimeline")
    if epsilon <= 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and positive")

    arms = (left_endpose, right_endpose)
    raw_presence = (left_present, right_present)
    latent_transforms: list[np.ndarray] = []
    latent_presence: list[np.ndarray] = []
    identity_frames = np.broadcast_to(np.eye(4, dtype=np.float64), (21, 4, 4)).copy()
    for pose, presence in zip(arms, raw_presence, strict=True):
        if pose is None:
            if presence is not None:
                raise ValueError("absent endpose cannot have an arm presence mask")
            latent_transforms.append(identity_frames.copy())
            latent_presence.append(np.zeros(21, dtype=bool))
            continue
        raw = _require_raw_poses(pose, timeline)
        valid = _require_presence(presence, timeline)
        latent_transforms.append(_causal_latent_transforms(_sample_raw_poses(raw, timeline)))
        latent_presence.append(_causal_latent_presence(valid, timeline))

    transforms = np.stack(latent_transforms)
    presence = np.stack(latent_presence)
    if presence[0, 0]:
        anchor_arm = "left"
        reference = transforms[0, 0]
    elif presence[1, 0]:
        anchor_arm = "right"
        reference = transforms[1, 0]
    else:
        return SE3Condition(
            arm_transform=np.broadcast_to(np.eye(4, dtype=np.float32), (2, 21, 4, 4)).copy(),
            arm_present=np.zeros((2, 21), dtype=bool),
            anchor_arm="identity",
            motion_scale=1.0,
        )

    relative = _invert_rigid_transforms(reference[None])[0] @ transforms
    displacements = relative[..., :3, 3] - relative[:, :1, :3, 3]
    valid_displacements = np.linalg.norm(displacements, axis=-1)[presence]
    motion_scale = float(np.max(valid_displacements)) if len(valid_displacements) else 0.0
    applied_scale = 1.0 if motion_scale <= epsilon else motion_scale
    normalized = relative.copy()
    normalized[..., :3, 3] /= applied_scale
    action_matrix = _invert_rigid_transforms(normalized)
    action_matrix[~presence] = np.eye(4, dtype=np.float64)
    _validate_rigid_transforms(action_matrix, label="normalized inverse SE(3) condition")
    return SE3Condition(
        arm_transform=action_matrix.astype(np.float32),
        arm_present=presence.astype(bool),
        anchor_arm=anchor_arm,
        motion_scale=applied_scale,
    )


def _require_scalar_unicode(payload: dict[str, np.ndarray], key: str, expected: str | None = None) -> str:
    value = payload[key]
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"cached {key} must be a scalar unicode value")
    result = str(value.item())
    if expected is not None and result != expected:
        raise ValueError(f"cached {key} differs")
    return result


def _require_sha256(value: str, *, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"cached {label} must be a lowercase SHA-256")


def validate_se3_cache(
    path: Path | str, expected_source_sha256: str
) -> dict[str, np.ndarray]:
    """Load and fully validate a v7 sidecar bound to one source manifest hash."""
    _require_sha256(expected_source_sha256, label="expected source manifest hash")
    cache_path = Path(path)
    if not cache_path.is_file() or cache_path.is_symlink():
        raise ValueError(f"SE(3) cache is not a regular file: {cache_path}")
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"cannot load SE(3) cache: {cache_path}") from exc
    if set(payload) != _CACHE_KEYS:
        raise ValueError("cached SE(3) condition has invalid schema keys")
    _require_scalar_unicode(payload, "schema", SE3_CACHE_SCHEMA)
    _require_scalar_unicode(payload, "temporal_contract", TEMPORAL_CONTRACT)
    anchor_arm = _require_scalar_unicode(payload, "anchor_arm")
    if anchor_arm not in {"left", "right", "identity"}:
        raise ValueError("cached anchor_arm is invalid")
    episode_hash = _require_scalar_unicode(payload, "source_episode_sha256")
    manifest_hash = _require_scalar_unicode(payload, "source_manifest_sha256")
    _require_sha256(episode_hash, label="source episode hash")
    _require_sha256(manifest_hash, label="source manifest hash")
    if manifest_hash != expected_source_sha256:
        raise ValueError("cached source manifest hash differs")
    transforms = payload["arm_transform"]
    if transforms.shape != (2, 21, 4, 4) or transforms.dtype != np.dtype(np.float32):
        raise ValueError("cached arm_transform must have shape (2,21,4,4) and dtype float32")
    presence = payload["arm_present"]
    if presence.shape != (2, 21) or presence.dtype != np.dtype(bool):
        raise ValueError("cached arm_present must have shape (2,21) and dtype bool")
    motion_scale = payload["motion_scale"]
    if motion_scale.shape != () or motion_scale.dtype != np.dtype(np.float64):
        raise ValueError("cached motion_scale must be a scalar float64")
    if not np.isfinite(motion_scale.item()) or float(motion_scale.item()) <= 0:
        raise ValueError("cached motion_scale must be finite and positive")
    _validate_rigid_transforms(transforms, label="cached arm_transform")
    if np.any(~presence) and not np.array_equal(
        transforms[~presence], np.broadcast_to(np.eye(4, dtype=np.float32), (np.count_nonzero(~presence), 4, 4))
    ):
        raise ValueError("cached absent arm_transform entries must be identity")
    if anchor_arm == "identity" and (presence.any() or not np.array_equal(
        transforms, np.broadcast_to(np.eye(4, dtype=np.float32), (2, 21, 4, 4))
    )):
        raise ValueError("identity anchor requires an all-absent identity condition")
    return payload


def write_se3_cache_atomic(
    path: Path | str,
    condition: SE3Condition,
    *,
    source_episode_sha256: str,
    source_manifest_sha256: str,
) -> Path:
    """Publish a fully validated v7 sidecar with an atomic same-directory rename."""
    _require_sha256(source_episode_sha256, label="source episode hash")
    _require_sha256(source_manifest_sha256, label="source manifest hash")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": np.asarray(SE3_CACHE_SCHEMA),
        "arm_transform": np.asarray(condition.arm_transform, dtype=np.float32),
        "arm_present": np.asarray(condition.arm_present, dtype=bool),
        "anchor_arm": np.asarray(condition.anchor_arm),
        "motion_scale": np.asarray(condition.motion_scale, dtype=np.float64),
        "source_episode_sha256": np.asarray(source_episode_sha256),
        "source_manifest_sha256": np.asarray(source_manifest_sha256),
        "temporal_contract": np.asarray(TEMPORAL_CONTRACT),
    }
    partial = target.with_name(f".{target.name}.partial.npz")
    try:
        with partial.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        validate_se3_cache(partial, source_manifest_sha256)
        os.replace(partial, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if partial.exists():
            partial.unlink()
    return target
