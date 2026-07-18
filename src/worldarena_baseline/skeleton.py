from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import yourdfpy

from .resample import resample_actions


class AlohaSkeletonRenderer:
    """CPU renderer for RoboTwin's dual-arm Aloha joint14 trajectories."""

    CAMERA_POSITION = np.array([-0.032, -0.45, 1.35], dtype=np.float64)
    CAMERA_FORWARD = np.array([0.0, 0.6, -0.8], dtype=np.float64)
    CAMERA_LEFT = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    ROBOT_POSITION = np.array([0.0, -0.65, 0.0], dtype=np.float64)
    ROBOT_QUATERNION_WXYZ = np.array([0.707, 0.0, 0.0, 0.707], dtype=np.float64)

    def __init__(
        self,
        urdf_path: Path | str,
        *,
        width: int = 640,
        height: int = 480,
        fovy_degrees: float = 37.0,
    ) -> None:
        self.width = width
        self.height = height
        self.intrinsic = camera_intrinsic(width, height, fovy_degrees)
        self.urdf = yourdfpy.URDF.load(
            str(urdf_path), build_scene_graph=True, load_meshes=False
        )
        self.root_rotation = _quaternion_wxyz_to_matrix(self.ROBOT_QUATERNION_WXYZ)

    def _world_positions(self, link_names: list[str]) -> np.ndarray:
        graph = self.urdf.scene.graph
        points = []
        for link_name in link_names:
            transform = graph.get(frame_from="footprint", frame_to=link_name)[0]
            point = np.asarray(transform[:3, 3], dtype=np.float64)
            points.append(self.root_rotation @ point + self.ROBOT_POSITION)
        return np.asarray(points)

    def _draw_arm(self, frame: np.ndarray, prefix: str) -> None:
        arm_links = [f"{prefix}_base_link"] + [
            f"{prefix}_link{index}" for index in range(1, 7)
        ]
        arm_world = self._world_positions(arm_links)
        arm_pixels, arm_visible = project_world_points(
            arm_world,
            camera_position=self.CAMERA_POSITION,
            camera_forward=self.CAMERA_FORWARD,
            camera_left=self.CAMERA_LEFT,
            intrinsic=self.intrinsic,
        )
        for index in range(len(arm_pixels) - 1):
            if arm_visible[index] and arm_visible[index + 1]:
                _clipped_line(frame, arm_pixels[index], arm_pixels[index + 1], (255, 255, 0), 3)
        for pixel, visible in zip(arm_pixels, arm_visible):
            if visible and np.all(np.isfinite(pixel)):
                cv2.circle(
                    frame,
                    tuple(np.rint(pixel).astype(int)),
                    4,
                    (0, 0, 255),
                    -1,
                )

        fingers_world = self._world_positions(
            [f"{prefix}_link6", f"{prefix}_link7", f"{prefix}_link8"]
        )
        finger_pixels, finger_visible = project_world_points(
            fingers_world,
            camera_position=self.CAMERA_POSITION,
            camera_forward=self.CAMERA_FORWARD,
            camera_left=self.CAMERA_LEFT,
            intrinsic=self.intrinsic,
        )
        for index in (1, 2):
            if finger_visible[0] and finger_visible[index]:
                _clipped_line(frame, finger_pixels[0], finger_pixels[index], (255, 0, 0), 5)

    def render_actions(self, actions: np.ndarray, *, num_frames: int = 81) -> np.ndarray:
        sampled = resample_actions(actions, num_frames=num_frames)
        frames = np.zeros((num_frames, self.height, self.width, 3), dtype=np.uint8)
        for index, action in enumerate(sampled):
            self.urdf.update_cfg(action_to_urdf_config(action))
            self._draw_arm(frames[index], "fl")
            self._draw_arm(frames[index], "fr")
        return frames


def _quaternion_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _clipped_line(
    image: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    clamp = 100_000
    p0 = tuple(np.rint(np.clip(start, -clamp, clamp)).astype(int))
    p1 = tuple(np.rint(np.clip(end, -clamp, clamp)).astype(int))
    ok, clipped_start, clipped_end = cv2.clipLine(
        (0, 0, image.shape[1], image.shape[0]), p0, p1
    )
    if ok and clipped_start != clipped_end:
        cv2.line(image, clipped_start, clipped_end, color, thickness)


def camera_intrinsic(width: int, height: int, fovy_degrees: float) -> np.ndarray:
    """Build the pinhole intrinsic matrix used by RoboTwin's SAPIEN camera."""
    focal = (height / 2.0) / np.tan(np.deg2rad(fovy_degrees) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def project_world_points(
    points_world: np.ndarray,
    *,
    camera_position: np.ndarray,
    camera_forward: np.ndarray,
    camera_left: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points using SAPIEN's forward-left-up camera pose."""
    points = np.asarray(points_world, dtype=np.float64)
    position = np.asarray(camera_position, dtype=np.float64)
    forward = np.asarray(camera_forward, dtype=np.float64)
    left = np.asarray(camera_left, dtype=np.float64)
    forward = forward / np.linalg.norm(forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    up = up / np.linalg.norm(up)
    delta = points - position
    camera_points = np.stack(
        [-delta @ left, -delta @ up, delta @ forward], axis=1
    )
    visible = camera_points[:, 2] > 1e-6
    depth = np.maximum(camera_points[:, 2], 1e-6)
    u = intrinsic[0, 0] * camera_points[:, 0] / depth + intrinsic[0, 2]
    v = intrinsic[1, 1] * camera_points[:, 1] / depth + intrinsic[1, 2]
    return np.stack([u, v], axis=1), visible


def action_to_urdf_config(action: np.ndarray) -> dict[str, float]:
    """Map WorldArena joint14 to RoboTwin's Aloha URDF joint names."""
    values = np.asarray(action, dtype=np.float64)
    if values.shape != (14,):
        raise ValueError(f"expected joint14 action, got {values.shape}")
    config = {
        f"fl_joint{index + 1}": float(values[index]) for index in range(6)
    }
    config.update(
        {f"fr_joint{index + 1}": float(values[index + 7]) for index in range(6)}
    )
    for prefix, normalized in (("fl", values[6]), ("fr", values[13])):
        opening = -0.01 + float(np.clip(normalized, 0.0, 1.0)) * 0.055
        config[f"{prefix}_joint7"] = opening
        config[f"{prefix}_joint8"] = opening
    return config
