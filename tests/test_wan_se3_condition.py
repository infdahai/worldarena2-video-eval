from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from worldarena_baseline.action_condition import EpisodeTimeline
from worldarena_baseline.wan_se3_condition import (
    SE3Condition,
    build_se3_condition,
    validate_se3_cache,
    write_se3_cache_atomic,
)


def _cache_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "cache_wan_v7_se3_conditions.py"
    spec = importlib.util.spec_from_file_location("cache_wan_v7_se3_conditions", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _timeline() -> EpisodeTimeline:
    return EpisodeTimeline.build(source_length=2, num_frames=81)


def _pose(x: float, y: float, z: float, *, quarter_turn: bool = False) -> np.ndarray:
    if quarter_turn:
        return np.array([x, y, z, np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)])
    return np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])


def _trajectories() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.stack([_pose(1.0, 0.0, 0.0), _pose(3.0, 0.0, 0.0, quarter_turn=True)]),
        np.stack([_pose(0.0, 2.0, 0.0), _pose(0.0, 4.0, 0.0)]),
    )


def _matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    rotation = matrix[:3, :3]
    trace = float(np.trace(rotation))
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                0.25 * scale,
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
            ]
        )
    else:
        raise AssertionError("fixture only uses positive-trace rotations")
    return np.concatenate([matrix[:3, 3], quaternion / np.linalg.norm(quaternion)])


def _pose_to_matrix(pose: np.ndarray) -> np.ndarray:
    _, x, y, z = pose[3:]
    w = pose[3]
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), pose[0]],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w), pose[1]],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y), pose[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _left_multiply_all(
    trajectories: tuple[np.ndarray, np.ndarray], global_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    return tuple(
        np.stack([_matrix_to_pose(global_transform @ _pose_to_matrix(pose)) for pose in arm])
        for arm in trajectories
    )  # type: ignore[return-value]


def _condition() -> SE3Condition:
    return build_se3_condition(*_trajectories(), timeline=_timeline())


def test_interpolation_uses_causal_group_endpoint_and_slerp() -> None:
    """Catches an implementation that averages rotations or maps latent one before RGB frame 4."""
    result = _condition()

    assert result.arm_transform.shape == (2, 21, 4, 4)
    assert result.arm_transform.dtype == np.dtype(np.float32)
    assert result.arm_present.dtype == np.dtype(bool)
    assert result.anchor_arm == "left"
    assert result.motion_scale == pytest.approx(2.0)
    np.testing.assert_allclose(result.arm_transform[0, 0], np.eye(4), atol=1e-6)
    np.testing.assert_allclose(result.arm_transform[0, 1, 0, 3], -0.04984587, atol=1e-6)
    np.testing.assert_allclose(
        result.arm_transform[0, 1, :3, :3],
        np.array([[0.99691733, 0.0784591, 0.0], [-0.0784591, 0.99691733, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-6,
    )


def test_common_global_transform_does_not_change_condition() -> None:
    """Catches anchoring in camera/world coordinates instead of the selected EEF frame."""
    angle = np.deg2rad(15.0)
    global_transform = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0, 10.0],
            [np.sin(angle), np.cos(angle), 0.0, -3.0],
            [0.0, 0.0, 1.0, 7.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    actual = _condition()
    transformed = build_se3_condition(
        *_left_multiply_all(_trajectories(), global_transform), timeline=_timeline()
    )

    np.testing.assert_allclose(actual.arm_transform, transformed.arm_transform, atol=1e-5)


def test_missing_left_uses_right_anchor_but_left_stays_absent() -> None:
    """Catches treating a numeric identity placeholder as a valid left action stream."""
    result = build_se3_condition(
        left_endpose=None,
        right_endpose=_trajectories()[1],
        timeline=_timeline(),
    )

    assert result.anchor_arm == "right"
    assert not result.arm_present[0].any()
    assert result.arm_present[1].all()
    np.testing.assert_array_equal(result.arm_transform[0], np.broadcast_to(np.eye(4), (21, 4, 4)))


def test_no_valid_anchor_zeros_both_streams() -> None:
    """Catches a future-frame pose being promoted to an anchor after both frame-zero streams are absent."""
    left, right = _trajectories()
    result = build_se3_condition(
        left,
        right,
        timeline=_timeline(),
        left_present=np.array([False, True]),
        right_present=np.array([False, True]),
    )

    assert result.anchor_arm == "identity"
    assert not result.arm_present.any()
    np.testing.assert_array_equal(result.arm_transform, np.broadcast_to(np.eye(4), (2, 21, 4, 4)))


def test_cache_rejects_standardized_11d_pose_and_wrong_source_hash(tmp_path: Path) -> None:
    """Catches accepting the legacy standardized 11-D pose cache or stale clean-1000 binding."""
    path = tmp_path / "bad.npz"
    np.savez_compressed(
        path,
        schema=np.asarray("wan-action-v7-se3-condition/1"),
        arm_transform=np.zeros((2, 21, 11), dtype=np.float32),
        arm_present=np.ones((2, 21), dtype=bool),
        anchor_arm=np.asarray("left"),
        motion_scale=np.asarray(1.0, dtype=np.float64),
        source_episode_sha256=np.asarray("a" * 64),
        source_manifest_sha256=np.asarray("a" * 64),
        temporal_contract=np.asarray("81-to-21-causal-v3"),
    )
    with pytest.raises(ValueError, match="2,21,4,4"):
        validate_se3_cache(path, expected_source_sha256="a" * 64)

    good_path = tmp_path / "good.npz"
    write_se3_cache_atomic(
        good_path,
        _condition(),
        source_episode_sha256="b" * 64,
        source_manifest_sha256="a" * 64,
    )
    with pytest.raises(ValueError, match="source manifest hash differs"):
        validate_se3_cache(good_path, expected_source_sha256="b" * 64)


def test_atomic_cache_round_trips_full_schema(tmp_path: Path) -> None:
    """Catches publishing an unvalidated or incomplete sidecar instead of the exact v7 schema."""
    output = tmp_path / "condition.npz"
    condition = _condition()
    write_se3_cache_atomic(
        output,
        condition,
        source_episode_sha256=hashlib.sha256(b"episode").hexdigest(),
        source_manifest_sha256=hashlib.sha256(b"manifest").hexdigest(),
    )

    actual = validate_se3_cache(
        output, expected_source_sha256=hashlib.sha256(b"manifest").hexdigest()
    )
    assert set(actual) == {
        "schema",
        "arm_transform",
        "arm_present",
        "anchor_arm",
        "motion_scale",
        "source_episode_sha256",
        "source_manifest_sha256",
        "temporal_contract",
    }
    assert actual["anchor_arm"].item() == condition.anchor_arm
    assert not list(tmp_path.glob("*.partial.npz"))


def test_cache_input_contract_rejects_forbidden_overlap_even_with_1000_rows(
    tmp_path: Path,
) -> None:
    """Catches treating a count of 1,000 as proof of clean-1000 lineage."""
    module = _cache_script()
    manifest = tmp_path / "manifest.jsonl"
    rows = [{"sample": f"safe-{index}", "hdf5": f"safe-{index}.h5"} for index in range(999)]
    rows.append({"sample": "forbidden", "hdf5": "forbidden.h5"})
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    discovery = tmp_path / "discovery.jsonl"
    discovery.write_text('{"sample":"forbidden","hdf5":"forbidden.h5"}\n', encoding="utf-8")
    dev_fast20 = tmp_path / "dev-fast20.jsonl"
    dev_fast20.write_text("", encoding="utf-8")
    official_test = tmp_path / "official-test.jsonl"
    official_test.write_text("", encoding="utf-8")
    receipt = tmp_path / "leakage-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "contract": "wan-action-clean-data-scale-split/1",
                "small_rows": 1000,
                "small_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "evaluation_identity_sha256": {
                    "discovery": hashlib.sha256(
                        b"hdf5:forbidden.h5\nsample:forbidden"
                    ).hexdigest(),
                    "dev-fast20": hashlib.sha256(b"").hexdigest(),
                    "official-test": hashlib.sha256(b"").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset leakage"):
        module.validate_cache_input_contract(
            manifest=manifest,
            canonical_clean1000_manifest=manifest,
            data_leakage_receipt=receipt,
            discovery_manifest=discovery,
            dev_fast20_manifest=dev_fast20,
            official_test_manifest=official_test,
        )


def test_cache_cli_rejects_persistent_output_outside_formal_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches publishing a sidecar beneath a caller-selected local cache directory."""
    module = _cache_script()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_wan_v7_se3_conditions.py",
            "--manifest", "ignored.jsonl",
            "--canonical-clean1000-manifest", "ignored-canonical.jsonl",
            "--data-leakage-receipt", "ignored-receipt.json",
            "--discovery-manifest", "ignored-discovery.jsonl",
            "--dev-fast20-manifest", "ignored-dev.jsonl",
            "--official-test-manifest", "ignored-test.jsonl",
            "--dataset-root", "ignored-dataset",
            "--cache-root", str(tmp_path),
        ],
    )
    with pytest.raises(ValueError, match="project artifact root"):
        module.main()


def test_corrupt_sidecar_quarantine_leaves_neighboring_sample_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a cache repair that deletes or moves a whole cache directory."""
    module = _cache_script()
    monkeypatch.setattr(module, "require_formal_cache_root", lambda path: Path(path))
    monkeypatch.setattr(module, "_formal_output_path", lambda path: Path(path))
    cache_root = tmp_path / "cache"
    corrupt = cache_root / "wan_v7_se3_conditions" / "broken.npz"
    neighbor = cache_root / "wan_v7_se3_conditions" / "healthy.npz"
    corrupt.parent.mkdir(parents=True)
    corrupt.write_bytes(b"not an npz")
    neighbor.write_bytes(b"healthy")

    quarantine = module._quarantine_sidecar(
        corrupt, cache_root=cache_root, sample="broken"
    )

    assert not corrupt.exists()
    assert (quarantine / "broken.npz").read_bytes() == b"not an npz"
    assert neighbor.read_bytes() == b"healthy"
