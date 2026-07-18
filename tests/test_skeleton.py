from __future__ import annotations

import numpy as np

from worldarena_baseline.skeleton import (
    AlohaSkeletonRenderer,
    action_to_urdf_config,
    camera_intrinsic,
    project_world_points,
)


def _write_minimal_aloha_urdf(path) -> None:
    links = ["<link name=\"footprint\"/>"]
    joints = []
    for prefix, y in (("fl", 0.3), ("fr", -0.3)):
        links.append(f"<link name=\"{prefix}_base_link\"/>")
        joints.append(
            f"<joint name=\"{prefix}_base_joint\" type=\"fixed\">"
            f"<parent link=\"footprint\"/><child link=\"{prefix}_base_link\"/>"
            f"<origin xyz=\"0.23 {y} 0.78\" rpy=\"0 0 0\"/></joint>"
        )
        parent = f"{prefix}_base_link"
        for index in range(1, 7):
            child = f"{prefix}_link{index}"
            links.append(f"<link name=\"{child}\"/>")
            joints.append(
                f"<joint name=\"{prefix}_joint{index}\" type=\"revolute\">"
                f"<parent link=\"{parent}\"/><child link=\"{child}\"/>"
                "<origin xyz=\"0.1 0 0.03\" rpy=\"0 0 0\"/>"
                "<axis xyz=\"0 1 0\"/><limit lower=\"-3.14\" upper=\"3.14\" "
                "effort=\"1\" velocity=\"1\"/></joint>"
            )
            parent = child
        for index, axis in ((7, "0 1 0"), (8, "0 -1 0")):
            child = f"{prefix}_link{index}"
            links.append(f"<link name=\"{child}\"/>")
            joints.append(
                f"<joint name=\"{prefix}_joint{index}\" type=\"prismatic\">"
                f"<parent link=\"{prefix}_link6\"/><child link=\"{child}\"/>"
                f"<origin xyz=\"0.08 0 0\" rpy=\"0 0 0\"/><axis xyz=\"{axis}\"/>"
                "<limit lower=\"-0.01\" upper=\"0.045\" effort=\"1\" velocity=\"1\"/>"
                "</joint>"
            )
    path.write_text(
        "<robot name=\"minimal_aloha\">" + "".join(links + joints) + "</robot>",
        encoding="utf-8",
    )


def test_camera_intrinsic_uses_vertical_field_of_view() -> None:
    intrinsic = camera_intrinsic(width=640, height=480, fovy_degrees=37.0)

    expected_focal = 240.0 / np.tan(np.deg2rad(37.0) / 2.0)
    np.testing.assert_allclose(intrinsic[0, 0], expected_focal)
    np.testing.assert_allclose(intrinsic[1, 1], expected_focal)
    np.testing.assert_allclose(intrinsic[:2, 2], [320.0, 240.0])


def test_projection_matches_sapien_forward_left_up_camera_convention() -> None:
    position = np.array([0.0, 0.0, 0.0])
    forward = np.array([0.0, 0.0, 1.0])
    left = np.array([-1.0, 0.0, 0.0])
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
        ]
    )

    pixels, visible = project_world_points(
        points,
        camera_position=position,
        camera_forward=forward,
        camera_left=left,
        intrinsic=camera_intrinsic(640, 480, 90.0),
    )

    np.testing.assert_allclose(pixels[0], [320.0, 240.0])
    assert pixels[1, 0] > pixels[0, 0]
    assert pixels[2, 1] > pixels[0, 1]
    assert visible.tolist() == [True, True, True]


def test_action_to_urdf_config_maps_both_arms_and_normalized_grippers() -> None:
    action = np.arange(14, dtype=np.float64)
    action[6] = 0.0
    action[13] = 1.0

    config = action_to_urdf_config(action)

    assert config["fl_joint1"] == 0.0
    assert config["fl_joint6"] == 5.0
    assert config["fr_joint1"] == 7.0
    assert config["fr_joint6"] == 12.0
    assert config["fl_joint7"] == -0.01
    assert config["fl_joint8"] == -0.01
    assert config["fr_joint7"] == 0.045
    assert config["fr_joint8"] == 0.045


def test_aloha_renderer_outputs_moving_dual_arm_skeleton(tmp_path) -> None:
    urdf_path = tmp_path / "aloha.urdf"
    _write_minimal_aloha_urdf(urdf_path)
    actions = np.zeros((2, 14), dtype=np.float64)
    actions[:, (6, 13)] = 1.0
    actions[1, 1:5] = [0.4, -0.6, 0.5, -0.3]
    actions[1, 8:12] = [-0.4, 0.6, -0.5, 0.3]

    renderer = AlohaSkeletonRenderer(urdf_path, width=320, height=240)
    frames = renderer.render_actions(actions, num_frames=5)

    assert frames.shape == (5, 240, 320, 3)
    assert frames.dtype == np.uint8
    assert np.count_nonzero(frames) > 0
    assert not np.array_equal(frames[0], frames[-1])
