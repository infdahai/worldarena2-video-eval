from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def causal_time_groups(*, num_frames: int = 81) -> tuple[tuple[int, ...], ...]:
    """Map 81 raster frames to Wan's causal 21-token temporal grid."""
    if num_frames != 81:
        raise ValueError("causal v3 time groups require exactly 81 frames")
    return ((0,),) + tuple(
        tuple(range(start, start + 4)) for start in range(1, num_frames, 4)
    )


@dataclass(frozen=True)
class EpisodeTimeline:
    source_length: int
    target_positions: np.ndarray
    rgb_indices: np.ndarray

    @classmethod
    def build(cls, *, source_length: int, num_frames: int = 81) -> "EpisodeTimeline":
        if source_length < 1:
            raise ValueError("source_length must be positive")
        if num_frames < 1:
            raise ValueError("num_frames must be positive")
        positions = np.linspace(0.0, float(source_length - 1), num_frames)
        return cls(
            source_length=source_length,
            target_positions=positions,
            rgb_indices=np.rint(positions).astype(np.int64),
        )

    def interpolate(self, values: np.ndarray) -> np.ndarray:
        source = np.asarray(values)
        if source.ndim < 1 or source.shape[0] != self.source_length:
            raise ValueError(
                f"values must start with source length {self.source_length}"
            )
        if self.source_length == 1:
            return np.repeat(source, len(self.target_positions), axis=0)
        flat = source.reshape(self.source_length, -1)
        source_positions = np.arange(self.source_length, dtype=np.float64)
        sampled = np.empty(
            (len(self.target_positions), flat.shape[1]), dtype=np.float64
        )
        for dimension in range(flat.shape[1]):
            sampled[:, dimension] = np.interp(
                self.target_positions,
                source_positions,
                flat[:, dimension],
            )
        return sampled.reshape((len(self.target_positions), *source.shape[1:]))


def _normalize_quaternions_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise ValueError("zero-norm quaternion")
    normalized = values / norms
    for index in range(1, len(normalized)):
        if np.dot(normalized[index - 1], normalized[index]) < 0:
            normalized[index] *= -1
    return normalized


def _slerp_pair(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    return (
        np.sin((1.0 - alpha) * theta) / sin_theta * q0
        + np.sin(alpha * theta) / sin_theta * q1
    )


def slerp_wxyz(quaternions: np.ndarray, timeline: EpisodeTimeline) -> np.ndarray:
    values = _normalize_quaternions_wxyz(quaternions)
    if len(values) != timeline.source_length:
        raise ValueError("quaternion and timeline lengths differ")
    if len(values) == 1:
        return np.repeat(values, len(timeline.target_positions), axis=0)
    output = np.empty((len(timeline.target_positions), 4), dtype=np.float64)
    for index, position in enumerate(timeline.target_positions):
        lower = min(int(np.floor(position)), len(values) - 1)
        upper = min(lower + 1, len(values) - 1)
        alpha = float(position - lower)
        output[index] = _slerp_pair(values[lower], values[upper], alpha)
    return output


def quaternion_wxyz_to_rotation_6d(quaternions: np.ndarray) -> np.ndarray:
    values = _normalize_quaternions_wxyz(quaternions)
    w, x, y, z = values.T
    matrices = np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)
    return matrices[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)


def _rotation_matrices_wxyz(quaternions: np.ndarray) -> np.ndarray:
    values = _normalize_quaternions_wxyz(quaternions)
    w, x, y, z = values.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def _rotation_matrices_to_6d(matrices: np.ndarray) -> np.ndarray:
    return np.asarray(matrices, dtype=np.float64)[:, :, :2].transpose(0, 2, 1).reshape(
        -1, 6
    )


def _validate_se3_trajectory(
    endpose: np.ndarray, opening: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    poses = np.asarray(endpose, dtype=np.float64)
    openings = np.asarray(opening, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[0] < 1 or poses.shape[1] != 7:
        raise ValueError("endpose must have shape (T, 7) for T >= 1")
    if openings.shape != (len(poses),):
        raise ValueError("opening must have shape (T,)")
    if not np.isfinite(poses).all() or not np.isfinite(openings).all():
        raise ValueError("SE(3) trajectory must be finite")
    _normalize_quaternions_wxyz(poses[:, 3:7])
    return poses, openings


def zero_se3_counterfactual(
    endpose: np.ndarray, opening: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Anchor an arm at its own frame-zero pose and opening."""
    poses, openings = _validate_se3_trajectory(endpose, opening)
    return (
        np.repeat(poses[:1], len(poses), axis=0),
        np.repeat(openings[:1], len(openings)),
    )


def _rotation_matrix_to_quaternion_wxyz(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(rotation)))
        if axis == 0:
            scale = 2 * np.sqrt(1 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    0.25 * scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2 * np.sqrt(1 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2])
            quaternion = np.array(
                [
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[0, 1] + rotation[1, 0]) / scale,
                    0.25 * scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                ]
            )
        else:
            scale = 2 * np.sqrt(1 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1])
            quaternion = np.array(
                [
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                    (rotation[0, 2] + rotation[2, 0]) / scale,
                    (rotation[1, 2] + rotation[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion if quaternion[0] >= 0 else -quaternion


def _poses_to_transforms(poses: np.ndarray) -> np.ndarray:
    transforms = np.broadcast_to(np.eye(4), (len(poses), 4, 4)).copy()
    transforms[:, :3, :3] = _rotation_matrices_wxyz(poses[:, 3:7])
    transforms[:, :3, 3] = poses[:, :3]
    return transforms


def _invert_transforms(transforms: np.ndarray) -> np.ndarray:
    inverse = np.broadcast_to(np.eye(4), np.asarray(transforms).shape).copy()
    inverse[:, :3, :3] = transforms[:, :3, :3].transpose(0, 2, 1)
    inverse[:, :3, 3] = -np.einsum(
        "tij,tj->ti", inverse[:, :3, :3], transforms[:, :3, 3]
    )
    return inverse


def _transforms_to_poses(transforms: np.ndarray) -> np.ndarray:
    result = np.empty((len(transforms), 7), dtype=np.float64)
    result[:, :3] = transforms[:, :3, 3]
    result[:, 3:7] = np.stack(
        [_rotation_matrix_to_quaternion_wxyz(matrix) for matrix in transforms[:, :3, :3]]
    )
    return result


def reverse_se3_counterfactual(
    endpose: np.ndarray, opening: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Invert each frame's relative SE(3) motion while preserving frame zero."""
    poses, openings = _validate_se3_trajectory(endpose, opening)
    transforms = _poses_to_transforms(poses)
    relative = _invert_transforms(transforms[:1]) @ transforms
    return _transforms_to_poses(transforms[:1] @ _invert_transforms(relative)), 2 * openings[:1] - openings


def _with_swapped_relative_motion(
    anchor_pose: np.ndarray, motion_pose: np.ndarray
) -> np.ndarray:
    destination = _poses_to_transforms(anchor_pose)
    source = _poses_to_transforms(motion_pose)
    relative_motion = _invert_transforms(source[:1]) @ source
    return _transforms_to_poses(destination[:1] @ relative_motion)


def swap_se3_counterfactual(
    left_endpose: np.ndarray,
    left_opening: np.ndarray,
    right_endpose: np.ndarray,
    right_opening: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Exchange arm-relative SE(3) and opening deltas while retaining each anchor."""
    left_pose, left_values = _validate_se3_trajectory(left_endpose, left_opening)
    right_pose, right_values = _validate_se3_trajectory(right_endpose, right_opening)
    if len(left_pose) != len(right_pose):
        raise ValueError("left and right SE(3) trajectories must have equal lengths")
    return (
        _with_swapped_relative_motion(left_pose, right_pose),
        _with_swapped_relative_motion(right_pose, left_pose),
        left_values[:1] + (right_values - right_values[:1]),
        right_values[:1] + (left_values - left_values[:1]),
    )


def encode_endpose_condition(
    endpose: np.ndarray,
    gripper: np.ndarray,
    timeline: EpisodeTimeline,
    *,
    camera_extrinsic_cv: np.ndarray | None = None,
) -> np.ndarray:
    poses = np.asarray(endpose, dtype=np.float64)
    openings = np.asarray(gripper, dtype=np.float64)
    if poses.shape != (timeline.source_length, 7):
        raise ValueError("endpose must have shape (source_length, 7)")
    if openings.shape != (timeline.source_length,):
        raise ValueError("gripper must have shape (source_length,)")
    extrinsic = (
        np.concatenate([np.eye(3), np.zeros((3, 1))], axis=1)
        if camera_extrinsic_cv is None
        else np.asarray(camera_extrinsic_cv, dtype=np.float64)
    )
    if extrinsic.shape != (3, 4):
        raise ValueError("camera_extrinsic_cv must have shape (3, 4)")
    world_to_camera = extrinsic[:, :3]
    world_position = timeline.interpolate(poses[:, :3])
    position = world_position @ world_to_camera.T + extrinsic[:, 3]
    relative_position = position - position[:1]
    world_rotation = _rotation_matrices_wxyz(
        slerp_wxyz(poses[:, 3:7], timeline)
    )
    camera_rotation = world_to_camera[None] @ world_rotation
    relative_rotation = camera_rotation @ camera_rotation[:1].transpose(0, 2, 1)
    rotation_6d = _rotation_matrices_to_6d(relative_rotation)
    sampled_gripper = timeline.interpolate(openings[:, None])
    return np.concatenate(
        [relative_position, position[:, 2:3], rotation_6d, sampled_gripper], axis=1
    ).astype(np.float32)
