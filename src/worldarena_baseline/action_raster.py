from __future__ import annotations

import math

import cv2
import numpy as np

from .skeleton import ProjectedArmTrajectory


def _validate_trajectory(trajectory: ProjectedArmTrajectory) -> int:
    left_points = np.asarray(trajectory.left_points)
    if left_points.ndim != 3 or left_points.shape[1:] != (9, 2):
        raise ValueError("left_points must have shape (T, 9, 2)")
    num_frames = left_points.shape[0]
    expected = {
        "right_points": (num_frames, 9, 2),
        "left_visible": (num_frames, 9),
        "right_visible": (num_frames, 9),
        "left_gripper": (num_frames,),
        "right_gripper": (num_frames,),
    }
    for field, shape in expected.items():
        if np.asarray(getattr(trajectory, field)).shape != shape:
            raise ValueError(f"{field} must have shape {shape}")
    return num_frames


def swap_arms(trajectory: ProjectedArmTrajectory) -> ProjectedArmTrajectory:
    return ProjectedArmTrajectory(
        left_points=trajectory.right_points.copy(),
        right_points=trajectory.left_points.copy(),
        left_visible=trajectory.right_visible.copy(),
        right_visible=trajectory.left_visible.copy(),
        left_gripper=trajectory.right_gripper.copy(),
        right_gripper=trajectory.left_gripper.copy(),
    )


def reverse_time(trajectory: ProjectedArmTrajectory) -> ProjectedArmTrajectory:
    return ProjectedArmTrajectory(
        left_points=trajectory.left_points[::-1].copy(),
        right_points=trajectory.right_points[::-1].copy(),
        left_visible=trajectory.left_visible[::-1].copy(),
        right_visible=trajectory.right_visible[::-1].copy(),
        left_gripper=trajectory.left_gripper[::-1].copy(),
        right_gripper=trajectory.right_gripper[::-1].copy(),
    )


def _draw_gripper_heatmap(
    image: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    *,
    sigma: float,
) -> None:
    finger_points = np.asarray(points[-2:], dtype=np.float64)
    finger_visible = np.asarray(visible[-2:], dtype=np.bool_)
    if not finger_visible.all() or not np.isfinite(finger_points).all():
        return
    center_x, center_y = finger_points.mean(axis=0)
    radius = math.ceil(3 * sigma)
    x_start = max(0, math.floor(center_x) - radius)
    x_stop = min(image.shape[1], math.floor(center_x) + radius + 1)
    y_start = max(0, math.floor(center_y) - radius)
    y_stop = min(image.shape[0], math.floor(center_y) + radius + 1)
    if x_start >= x_stop or y_start >= y_stop:
        return
    yy, xx = np.mgrid[y_start:y_stop, x_start:x_stop]
    gaussian = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2 * sigma**2)
    )
    image[y_start:y_stop, x_start:x_stop] = gaussian.astype(np.float32)


def _draw_arm_flow(
    flow_x: np.ndarray,
    flow_y: np.ndarray,
    current_points: np.ndarray,
    current_visible: np.ndarray,
    previous_points: np.ndarray,
    previous_visible: np.ndarray,
    *,
    thickness: int,
) -> None:
    accumulation_x = np.zeros_like(flow_x)
    accumulation_y = np.zeros_like(flow_y)
    counts = np.zeros_like(flow_x)
    segments = [(index, index + 1) for index in range(6)] + [(6, 7), (6, 8)]
    for start, end in segments:
        if not (
            current_visible[start]
            and current_visible[end]
            and previous_visible[start]
            and previous_visible[end]
        ):
            continue
        segment_points = np.concatenate(
            [
                current_points[[start, end]],
                previous_points[[start, end]],
            ],
            axis=0,
        )
        if not np.isfinite(segment_points).all():
            continue
        displacement = (
            current_points[[start, end]] - previous_points[[start, end]]
        ).mean(axis=0)
        normalized_x = float(np.clip(displacement[0] / flow_x.shape[1], -1, 1))
        normalized_y = float(np.clip(displacement[1] / flow_x.shape[0], -1, 1))
        mask = np.zeros(flow_x.shape, dtype=np.uint8)
        clip_limit = 100_000
        p0 = tuple(
            np.rint(np.clip(current_points[start], -clip_limit, clip_limit)).astype(int)
        )
        p1 = tuple(
            np.rint(np.clip(current_points[end], -clip_limit, clip_limit)).astype(int)
        )
        cv2.line(mask, p0, p1, 1, thickness)
        accumulation_x += mask * normalized_x
        accumulation_y += mask * normalized_y
        counts += mask
    occupied = counts > 0
    flow_x[occupied] = accumulation_x[occupied] / counts[occupied]
    flow_y[occupied] = accumulation_y[occupied] / counts[occupied]


def _draw_arm_occupancy(
    occupancy: np.ndarray,
    points: np.ndarray,
    visible: np.ndarray,
    *,
    thickness: int,
) -> None:
    segments = [(index, index + 1) for index in range(6)] + [(6, 7), (6, 8)]
    clip_limit = 100_000
    for start, end in segments:
        if not (visible[start] and visible[end]):
            continue
        segment = points[[start, end]]
        if not np.isfinite(segment).all():
            continue
        p0, p1 = (
            tuple(np.rint(np.clip(point, -clip_limit, clip_limit)).astype(int))
            for point in segment
        )
        intersects, clipped_start, clipped_end = cv2.clipLine(
            (0, 0, occupancy.shape[1], occupancy.shape[0]), p0, p1
        )
        if intersects:
            cv2.line(occupancy, clipped_start, clipped_end, 1.0, thickness)


def rasterize_v3_action(
    trajectory: ProjectedArmTrajectory,
    *,
    sigma: float = 3.0,
    arm_thickness: int = 3,
) -> np.ndarray:
    """Render the schema-v3, left/right-stable 80x60 condition raster."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if arm_thickness <= 0:
        raise ValueError("arm_thickness must be positive")
    num_frames = _validate_trajectory(trajectory)
    height, width = 60, 80
    raster = np.zeros((num_frames, 10, height, width), dtype=np.float32)
    arms = (
        (0, 1, 2, 3, 4, trajectory.left_points, trajectory.left_visible, trajectory.left_gripper),
        (5, 6, 7, 8, 9, trajectory.right_points, trajectory.right_visible, trajectory.right_gripper),
    )
    for index in range(num_frames):
        for occupancy_channel, heatmap_channel, flow_x_channel, flow_y_channel, opening_channel, points, visible, gripper in arms:
            _draw_arm_occupancy(
                raster[index, occupancy_channel],
                points[index],
                visible[index],
                thickness=arm_thickness,
            )
            _draw_gripper_heatmap(
                raster[index, heatmap_channel],
                points[index],
                visible[index],
                sigma=sigma,
            )
            raster[index, opening_channel] = raster[index, heatmap_channel] * np.clip(
                gripper[index], 0.0, 1.0
            )
            if index > 0:
                _draw_arm_flow(
                    raster[index, flow_x_channel],
                    raster[index, flow_y_channel],
                    points[index],
                    visible[index],
                    points[index - 1],
                    visible[index - 1],
                    thickness=arm_thickness,
                )
    return raster


def rasterize_action(
    trajectory: ProjectedArmTrajectory,
    *,
    height: int,
    width: int,
    sigma: float = 3.0,
    arm_thickness: int = 3,
) -> np.ndarray:
    if height <= 0:
        raise ValueError("height must be positive")
    if width <= 0:
        raise ValueError("width must be positive")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    if arm_thickness <= 0:
        raise ValueError("arm_thickness must be positive")

    num_frames = _validate_trajectory(trajectory)
    raster = np.zeros((num_frames, 8, height, width), dtype=np.float32)
    for index in range(num_frames):
        _draw_gripper_heatmap(
            raster[index, 0],
            trajectory.left_points[index],
            trajectory.left_visible[index],
            sigma=sigma,
        )
        _draw_gripper_heatmap(
            raster[index, 1],
            trajectory.right_points[index],
            trajectory.right_visible[index],
            sigma=sigma,
        )
        raster[index, 6] = raster[index, 0] * np.clip(
            trajectory.left_gripper[index], 0.0, 1.0
        )
        raster[index, 7] = raster[index, 1] * np.clip(
            trajectory.right_gripper[index], 0.0, 1.0
        )
        if index > 0:
            _draw_arm_flow(
                raster[index, 2],
                raster[index, 3],
                trajectory.left_points[index],
                trajectory.left_visible[index],
                trajectory.left_points[index - 1],
                trajectory.left_visible[index - 1],
                thickness=arm_thickness,
            )
            _draw_arm_flow(
                raster[index, 4],
                raster[index, 5],
                trajectory.right_points[index],
                trajectory.right_visible[index],
                trajectory.right_points[index - 1],
                trajectory.right_visible[index - 1],
                thickness=arm_thickness,
            )
    return raster
