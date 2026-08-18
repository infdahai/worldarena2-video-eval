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
    sys.modules[spec.name] = module
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


def _write_lineage_authority(module, root: Path):
    manifest = root / "clean-1000.jsonl"
    rows = [{"sample": f"safe-{index}", "hdf5": f"safe-{index}.h5"} for index in range(1000)]
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    dev_fast20 = root / "dev-fast20.jsonl"
    dev_fast20.write_text(
        "".join(
            json.dumps({"sample": f"dev-{index:02d}", "hdf5": f"dev-{index:02d}.h5"}) + "\n"
            for index in reversed(range(20))
        ),
        encoding="utf-8",
    )
    receipt = root / "leakage-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "contract": "wan-action-clean-data-scale-split/1",
                "small_rows": 1000,
                "small_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "evaluation_identity_sha256": {
                    "dev-fast20": hashlib.sha256(
                        "\n".join(
                            sorted(
                                [f"hdf5:dev-{index:02d}.h5" for index in range(20)]
                                + [f"sample:dev-{index:02d}" for index in range(20)]
                            )
                        ).encode()
                    ).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    pins = root / "trusted-lineage-pins.json"
    pins.write_text(
        json.dumps(
            {
                "schema": "wan-action-v7-se3-lineage-pins/3",
                "artifacts": {
                    "clean1000_manifest": {
                        "sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    },
                    "data_leakage_receipt": {
                        "sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
                    },
                    "discovery_manifest": {
                        "contract": "wan-action-v7-discovery-derivation/1",
                        "rows": 8,
                        "selector": "sample-lexicographic-first-8/v1",
                        "source_artifact": "clean1000_manifest",
                        "source_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                    },
                    "dev_fast20_manifest": {
                        "sha256": hashlib.sha256(dev_fast20.read_bytes()).hexdigest(),
                    },
                    "official_test_manifest": {
                        "status": "unavailable",
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return module._LineageAuthority(
        clean1000_manifest=manifest,
        data_leakage_receipt=receipt,
        discovery_manifest=root / "discovery-8.jsonl",
        dev_fast20_manifest=dev_fast20,
        trusted_pins=pins,
        expected_pins_sha256=hashlib.sha256(pins.read_bytes()).hexdigest(),
    )


def test_injected_immutable_lineage_pins_validate_exact_artifacts(tmp_path: Path) -> None:
    """Catches accepting a clean-1000 artifact whose bytes differ from trusted pins."""
    module = _cache_script()
    authority = _write_lineage_authority(module, tmp_path)

    report = module._validate_lineage_for_testing(authority)
    assert report["manifest_sha256"] == hashlib.sha256(
        authority.clean1000_manifest.read_bytes()
    ).hexdigest()

    authority.dev_fast20_manifest.write_text('{"sample":"forged"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="trusted lineage pin"):
        module._validate_lineage_for_testing(authority)


def test_injected_trusted_lineage_still_rejects_discovery_overlap(tmp_path: Path) -> None:
    """Catches trusting hashes alone without replaying the zero-leakage receipt."""
    module = _cache_script()
    authority = _write_lineage_authority(module, tmp_path)
    rows = [
        {"sample": f"safe-{index}", "hdf5": f"safe-{index}.h5"}
        for index in range(999)
    ] + [{"sample": "dev-00", "hdf5": "dev-00.h5"}]
    authority.clean1000_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    receipt = json.loads(authority.data_leakage_receipt.read_text(encoding="utf-8"))
    receipt["small_manifest_sha256"] = hashlib.sha256(
        authority.clean1000_manifest.read_bytes()
    ).hexdigest()
    authority.data_leakage_receipt.write_text(json.dumps(receipt), encoding="utf-8")
    pins = json.loads(authority.trusted_pins.read_text(encoding="utf-8"))
    pins["artifacts"]["clean1000_manifest"]["sha256"] = receipt["small_manifest_sha256"]
    pins["artifacts"]["discovery_manifest"]["source_sha256"] = receipt["small_manifest_sha256"]
    pins["artifacts"]["data_leakage_receipt"]["sha256"] = hashlib.sha256(
        authority.data_leakage_receipt.read_bytes()
    ).hexdigest()
    authority.trusted_pins.write_text(json.dumps(pins, sort_keys=True), encoding="utf-8")
    authority = module._LineageAuthority(
        clean1000_manifest=authority.clean1000_manifest,
        data_leakage_receipt=authority.data_leakage_receipt,
        discovery_manifest=authority.discovery_manifest,
        dev_fast20_manifest=authority.dev_fast20_manifest,
        trusted_pins=authority.trusted_pins,
        expected_pins_sha256=hashlib.sha256(authority.trusted_pins.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="dataset leakage"):
        module._validate_lineage_for_testing(authority)


def test_production_lineage_pins_use_compiled_digest_not_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches letting a caller replace both the tracked pins bytes and an env digest."""
    module = _cache_script()
    tracked = (
        Path(__file__).resolve().parents[1]
        / "source_inputs/trusted-wan-v7-se3-lineage-pins.json"
    )
    assert module.TRUSTED_LINEAGE_PINS_SHA256 == hashlib.sha256(tracked.read_bytes()).hexdigest()

    altered = tmp_path / "altered-pins.json"
    altered.write_text(tracked.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "FORMAL_TRUSTED_LINEAGE_PINS", altered)
    monkeypatch.setenv(
        "WAN_V7_SE3_TRUSTED_PINS_SHA256", hashlib.sha256(altered.read_bytes()).hexdigest()
    )

    authority = module._production_lineage_authority()
    with pytest.raises(ValueError, match="trusted lineage pins differ"):
        module._load_trusted_lineage_pins(authority)


def test_production_stage_a_lineage_allows_frozen_official_test_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches reading formal manifests or HDF5 when a required lineage pin is unavailable."""
    module = _cache_script()
    monkeypatch.setattr(
        module,
        "FORMAL_TRUSTED_LINEAGE_PINS",
        Path(__file__).resolve().parents[1]
        / "source_inputs/trusted-wan-v7-se3-lineage-pins.json",
    )

    hashes, contract = module._load_trusted_lineage_pins(module._production_lineage_authority())
    assert hashes["dev_fast20_manifest"]
    assert contract["rows"] == 8


def test_fabricated_candidate_receipt_and_eval_files_are_not_cli_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches restoring caller-selectable manifest, receipt, or evaluation authority flags."""
    module = _cache_script()
    fabricated = tmp_path / "fabricated.jsonl"
    fabricated.write_text('{"sample":"forged"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cache_wan_v7_se3_conditions.py",
            "--manifest", str(fabricated),
            "--canonical-clean1000-manifest", str(fabricated),
            "--data-leakage-receipt", str(fabricated),
            "--discovery-manifest", str(fabricated),
            "--dev-fast20-manifest", str(fabricated),
            "--official-test-manifest", str(fabricated),
            "--dataset-root", "ignored-dataset",
            "--cache-root", str(tmp_path),
        ],
    )

    with pytest.raises(SystemExit) as error:
        module.parse_args()
    assert error.value.code == 2


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
