from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from worldarena_baseline.wan_v9_transition import (
    LATENT_ENDPOINT_FRAMES,
    TransitionFeatures,
    build_transition_features,
    fit_normalization_statistics,
    physical_states_from_normalized_inverse,
    shift_transition_content,
    so3_log_map,
    validate_split_identities,
    validate_transition_cache,
    write_transition_cache_atomic,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _states() -> np.ndarray:
    values = np.broadcast_to(np.eye(4, dtype=np.float64), (2, 21, 4, 4)).copy()
    values[0, :, 0, 3] = np.arange(21, dtype=np.float64) * 0.002
    values[1, :, 1, 3] = np.arange(21, dtype=np.float64) * 0.003
    return values


def _raster() -> np.ndarray:
    raster = np.zeros((81, 10, 60, 80), dtype=np.float32)
    for frame in range(81):
        left_u, left_v = 10 + frame / 8, 20 + frame / 16
        right_u, right_v = 60 - frame / 10, 40 - frame / 20
        for heatmap_channel, opening_channel, u, v, opening in (
            (1, 4, left_u, left_v, 0.25 + frame / 400),
            (6, 9, right_u, right_v, 0.75 - frame / 400),
        ):
            u0, v0 = int(round(u)), int(round(v))
            raster[frame, heatmap_channel, v0, u0] = 1.0
            raster[frame, opening_channel, v0, u0] = opening
    return raster


def _features() -> TransitionFeatures:
    return build_transition_features(
        _states(),
        np.ones((2, 21), dtype=bool),
        _raster(),
    )


def test_endpoint_mapping_is_exact() -> None:
    assert LATENT_ENDPOINT_FRAMES == (
        0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40,
        44, 48, 52, 56, 60, 64, 68, 72, 76, 80,
    )


@pytest.mark.parametrize("angle", [0.0, 1e-8, 0.7, np.pi - 1e-7])
def test_so3_log_map_is_stable(angle: float) -> None:
    axis = np.array([1.0, 2.0, -3.0], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    rotation = np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)
    vector = so3_log_map(rotation)
    assert vector.dtype == np.float64
    assert np.isfinite(vector).all()
    np.testing.assert_allclose(np.linalg.norm(vector), angle, atol=2e-7, rtol=0)
    if angle > 1e-6:
        np.testing.assert_allclose(vector / np.linalg.norm(vector), axis, atol=2e-6, rtol=0)


def test_transition_features_have_exact_modalities_and_activity() -> None:
    result = _features()
    assert result.translation.shape == (2, 20, 4)
    assert result.rotation.shape == (2, 20, 4)
    assert result.image.shape == (2, 20, 6)
    assert result.gripper.shape == (2, 20, 3)
    assert result.arm_present.shape == (2, 20)
    assert result.motion_active.shape == (2, 20)
    assert result.destination_slot.shape == (20,)
    assert result.destination_slot.tolist() == list(range(20))
    np.testing.assert_allclose(result.translation[0, :, 0], 0.002, atol=1e-12)
    np.testing.assert_allclose(result.translation[1, :, 1], 0.003, atol=1e-12)
    assert result.motion_active.all()


def test_normalized_inverse_cache_recovers_physical_translation_scale() -> None:
    physical = _states()
    scale = 0.04
    normalized = physical.copy()
    normalized[..., :3, 3] /= scale
    inverse = np.linalg.inv(normalized).astype(np.float32)
    recovered = physical_states_from_normalized_inverse(
        inverse, np.ones((2, 21), dtype=bool), scale
    )
    np.testing.assert_allclose(recovered, physical, atol=2e-8, rtol=0)


def test_presence_requires_both_state_and_image_endpoints() -> None:
    state_present = np.ones((2, 21), dtype=bool)
    state_present[0, 4] = False
    raster = _raster()
    raster[LATENT_ENDPOINT_FRAMES[8], 6] = 0
    raster[LATENT_ENDPOINT_FRAMES[8], 9] = 0
    result = build_transition_features(_states(), state_present, raster)
    assert not result.arm_present[0, 3]
    assert not result.arm_present[0, 4]
    assert not result.arm_present[1, 7]
    assert not result.arm_present[1, 8]


def test_cache_roundtrip_binds_payload_and_sources(tmp_path: Path) -> None:
    features = _features()
    stats = fit_normalization_statistics({"sample-a": features})
    path = tmp_path / "sample-a.correct.npz"
    write_transition_cache_atomic(
        path,
        features,
        sample="sample-a",
        variant="correct",
        source_hdf5_sha256=_sha("hdf5"),
        source_action_sha256=_sha("action"),
        source_counterfactual_sha256="",
        urdf_sha256=_sha("urdf"),
        clean_manifest_sha256=_sha("manifest"),
        normalization_receipt_sha256=stats["receipt_sha256"],
    )
    payload = validate_transition_cache(
        path,
        expected_sample="sample-a",
        expected_variant="correct",
        expected_clean_manifest_sha256=_sha("manifest"),
        expected_normalization_receipt_sha256=stats["receipt_sha256"],
    )
    assert payload["translation"].shape == (2, 20, 4)
    assert payload["variant"].item() == "correct"

    with np.load(path, allow_pickle=False) as archive:
        altered = {key: np.asarray(archive[key]) for key in archive.files}
    altered["translation"] = altered["translation"].copy()
    altered["translation"][0, 0, 0] += 1
    np.savez(path, **altered)
    with pytest.raises(ValueError, match="payload hash"):
        validate_transition_cache(
            path,
            expected_sample="sample-a",
            expected_variant="correct",
            expected_clean_manifest_sha256=_sha("manifest"),
            expected_normalization_receipt_sha256=stats["receipt_sha256"],
        )


def test_normalization_statistics_are_order_independent_and_correct_only() -> None:
    first = _features()
    second = replace(_features(), translation=_features().translation + 0.25)
    stats_a = fit_normalization_statistics({"a": first, "b": second})
    stats_b = fit_normalization_statistics({"b": second, "a": first})
    assert stats_a == stats_b
    assert stats_a["samples"] == ["a", "b"]
    assert stats_a["contract"] == "wan-v9-transition-normalization/1"


@pytest.mark.parametrize("direction", [-1, 1])
def test_shift_moves_all_modality_content_without_moving_destination_slot(direction: int) -> None:
    correct = _features()
    shifted = shift_transition_content(correct, direction=direction)
    assert shifted.destination_slot.tolist() == list(range(20))
    for name in ("translation", "rotation", "image", "gripper", "arm_present", "motion_active"):
        original = getattr(correct, name)
        actual = getattr(shifted, name)
        if direction == 1:
            np.testing.assert_array_equal(actual[:, 1:], original[:, :-1])
            np.testing.assert_array_equal(actual[:, 0], original[:, 0])
        else:
            np.testing.assert_array_equal(actual[:, :-1], original[:, 1:])
            np.testing.assert_array_equal(actual[:, -1], original[:, -1])
    assert set(shifted.__dataclass_fields__) == {
        "translation", "rotation", "image", "gripper",
        "arm_present", "motion_active", "destination_slot",
    }


def test_split_contract_excludes_audit_and_rejects_dev_leakage() -> None:
    clean = [{"sample": f"clean-{index:04d}"} for index in range(1785)]
    audit = clean[:20]
    dev = [{"sample": f"dev-{index:04d}"} for index in range(20)]
    optimizer = validate_split_identities(clean, audit, dev)
    assert len(optimizer) == 1765
    assert set(optimizer).isdisjoint(row["sample"] for row in audit + dev)
    leaking_dev = [clean[-1], *dev[1:]]
    with pytest.raises(ValueError, match="dev-fast20 leaks"):
        validate_split_identities(clean, audit, leaking_dev)


def test_cache_cli_dry_run_is_side_effect_free(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    audit = tmp_path / "audit.jsonl"
    dev = tmp_path / "dev.jsonl"
    clean_rows = [{"sample": f"clean-{index:04d}"} for index in range(1785)]
    clean.write_text("\n".join(json.dumps(row) for row in clean_rows) + "\n")
    audit.write_text("\n".join(json.dumps(row) for row in clean_rows[:20]) + "\n")
    dev.write_text("\n".join(json.dumps({"sample": f"dev-{index:04d}"}) for index in range(20)) + "\n")
    script = Path(__file__).parents[1] / "scripts" / "cache_wan_v9_transitions.py"
    spec = importlib.util.spec_from_file_location("cache_wan_v9_transitions_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = tmp_path / "must-not-exist"
    receipt = module.prepare(
        clean_manifest=clean,
        audit_manifest=audit,
        dev_manifest=dev,
        output_root=output,
        dry_run=True,
    )
    assert receipt["optimizer_sample_count"] == 1765
    assert receipt["starts_cache_work"] is False
    assert not output.exists()
