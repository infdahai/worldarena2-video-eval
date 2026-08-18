from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from worldarena_baseline.wan_v10_data import (
    V10RelationFeatures,
    build_v10_data_receipt,
    build_v10_cache_extension,
    build_v10_split,
    fit_v10_relation_normalization,
    relation_features_from_v9_transition,
    validate_v10_data_receipt,
    validate_v10_relation_cache,
    write_v10_relation_cache_atomic,
)


STRATA = ("single_dominant", "bimanual_heavy", "mixed", "quiet")
TAGS = ("single_arm", "bimanual", "sequential", "crossing_or_overlap")


def _rows(count: int = 80):
    rows = []
    metadata = {}
    for index in range(count):
        sample = f"task_{index % 30:02d}__aloha-agilex_clean_50__episode_{index:06d}"
        rows.append({
            "sample": sample,
            "task": f"task_{index % 30:02d}",
            "variant": "aloha-agilex_clean_50",
            "episode_index": index,
            "hdf5": f"task_{index % 30:02d}/aloha-agilex_clean_50/data/episode{index}.hdf5",
            "instruction": f"task_{index % 30:02d}/aloha-agilex_clean_50/instructions/episode{index}.json",
            "seen_prompts": ["prompt"],
            "unseen_prompts": [],
        })
        metadata[sample] = {
            "stratum": STRATA[index % len(STRATA)],
            "probe_observable": index >= 2,
            "position_valid_count": 12,
            "velocity_valid_count": 10,
            "tags": [TAGS[index % len(TAGS)]],
        }
    return rows, metadata


def _write_jsonl(path: Path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_split_retains_every_eligible_non_eval_identity_deterministically() -> None:
    rows, metadata = _rows()
    dev = rows[:20]
    first = build_v10_split(rows, dev, metadata, seed=20260819)
    second = build_v10_split(list(reversed(rows)), list(reversed(dev)), metadata, seed=20260819)
    assert first == second
    assert len(first.audit) == 20
    assert len(first.optimizer) == len(rows) - len(dev) - len(first.audit)
    optimizer = {row["sample"] for row in first.optimizer}
    audit = {row["sample"] for row in first.audit}
    dev_ids = {row["sample"] for row in dev}
    assert not optimizer & (audit | dev_ids)
    assert not audit & dev_ids
    assert len({row["task"] for row in first.audit}) >= 8
    assert set(first.audit_tags) >= set(TAGS)
    assert sum(first.source_counts.values()) == len(rows)


def test_cache_extension_contains_only_uncached_full_action_rows() -> None:
    rows, _ = _rows()
    extension = build_v10_cache_extension(rows, rows[:63])
    assert [row["sample"] for row in extension] == sorted(
        row["sample"] for row in rows[63:]
    )
    with pytest.raises(ValueError, match="foreign"):
        build_v10_cache_extension(rows, [*rows[:2], {**rows[2], "sample": "foreign"}])


def test_split_rejects_robot_only_quarantine_duplicates_and_invalid_audit() -> None:
    rows, metadata = _rows()
    bad = [dict(row) for row in rows]
    bad[0]["hdf5"] = bad[0]["hdf5"].replace(".hdf5", "_robot_only.hdf5")
    with pytest.raises(ValueError, match="robot_only"):
        build_v10_split(bad, rows[:20], metadata)
    bad = [dict(row) for row in rows]
    bad[0]["hdf5"] = f".quarantine/{bad[0]['hdf5']}"
    with pytest.raises(ValueError, match="quarantine"):
        build_v10_split(bad, rows[:20], metadata)
    with pytest.raises(ValueError, match="duplicate"):
        build_v10_split([*rows, rows[0]], rows[:20], metadata)
    invalid = {sample: dict(value) for sample, value in metadata.items()}
    for value in invalid.values():
        value["position_valid_count"] = 0
    with pytest.raises(ValueError, match="20 observable"):
        build_v10_split(rows, rows[:20], invalid)


def test_data_receipt_binds_exact_manifest_bytes_and_zero_leakage(tmp_path: Path) -> None:
    rows, metadata = _rows()
    split = build_v10_split(rows, rows[:20], metadata)
    source = tmp_path / "source.jsonl"
    optimizer = tmp_path / "optimizer.jsonl"
    audit = tmp_path / "audit.jsonl"
    dev = tmp_path / "dev.jsonl"
    _write_jsonl(source, rows)
    _write_jsonl(optimizer, split.optimizer)
    _write_jsonl(audit, split.audit)
    _write_jsonl(dev, rows[:20])
    receipt = build_v10_data_receipt(
        source_manifest=source,
        optimizer_manifest=optimizer,
        audit_manifest=audit,
        dev_manifest=dev,
        split=split,
        official_test_status="unavailable",
    )
    validate_v10_data_receipt(
        receipt,
        source_manifest=source,
        optimizer_manifest=optimizer,
        audit_manifest=audit,
        dev_manifest=dev,
    )
    optimizer.write_text(optimizer.read_text() + "{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_v10_data_receipt(
            receipt,
            source_manifest=source,
            optimizer_manifest=optimizer,
            audit_manifest=audit,
            dev_manifest=dev,
        )


def _features() -> V10RelationFeatures:
    arrays = {
        "anchored_se3": np.zeros((21, 2, 6), dtype=np.float32),
        "velocity": np.zeros((21, 2, 6), dtype=np.float32),
        "uv": np.zeros((21, 2, 2), dtype=np.float32),
        "gripper": np.zeros((21, 2, 2), dtype=np.float32),
        "arm_present": np.ones((21, 2), dtype=bool),
        "motion_active": np.zeros((21, 2), dtype=bool),
    }
    return V10RelationFeatures(**arrays)


def test_relation_cache_roundtrip_is_hash_bound_and_absent_is_exact_zero(tmp_path: Path) -> None:
    correct = _features()
    reverse = _features()
    swap = _features()
    correct.arm_present[:, 1] = False
    path = tmp_path / "sample.npz"
    write_v10_relation_cache_atomic(
        path,
        sample="sample",
        variants={"correct": correct, "reverse": reverse, "swap": swap},
        source_hdf5_sha256=hashlib.sha256(b"hdf5").hexdigest(),
        source_action_sha256=hashlib.sha256(b"action").hexdigest(),
        source_urdf_sha256=hashlib.sha256(b"urdf").hexdigest(),
        normalization_sha256=hashlib.sha256(b"normalization").hexdigest(),
    )
    loaded = validate_v10_relation_cache(
        path,
        expected_sample="sample",
        expected_normalization_sha256=hashlib.sha256(b"normalization").hexdigest(),
    )
    assert loaded["correct_anchored_se3"].shape == (21, 2, 6)
    assert np.count_nonzero(loaded["correct_anchored_se3"][:, 1]) == 0
    assert str(loaded["source_urdf_sha256"].item()) == hashlib.sha256(b"urdf").hexdigest()
    with path.open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="payload|load|hash"):
        validate_v10_relation_cache(
            path,
            expected_sample="sample",
            expected_normalization_sha256=hashlib.sha256(b"normalization").hexdigest(),
        )


def test_relation_normalization_uses_only_explicit_optimizer_samples() -> None:
    first = _features()
    second = _features()
    first.anchored_se3[..., 0] = 1
    second.anchored_se3[..., 0] = 3
    receipt = fit_v10_relation_normalization({"a": first, "b": second})
    assert receipt["samples"] == ["a", "b"]
    assert receipt["mean"][0] == pytest.approx(2.0)
    assert receipt["scale"][0] == pytest.approx(1.0)
    assert receipt["mean"][16:] == [0.0, 0.0, 0.0, 0.0]
    assert receipt["scale"][16:] == [1.0, 1.0, 1.0, 1.0]
    assert len(receipt["receipt_sha256"]) == 64


def test_v9_transition_conversion_reconstructs_anchored_slots() -> None:
    translation = np.zeros((2, 20, 4), dtype=np.float32)
    rotation = np.zeros((2, 20, 4), dtype=np.float32)
    image = np.zeros((2, 20, 6), dtype=np.float32)
    gripper = np.zeros((2, 20, 3), dtype=np.float32)
    present = np.ones((2, 20), dtype=bool)
    active = np.zeros((2, 20), dtype=bool)
    translation[0, :, 0] = 0.1
    translation[0, :, 3] = 0.1
    image[0, :, 2] = np.arange(20)
    image[0, :, 3] = np.arange(20) + 1
    image[0, :, 4] = 1
    converted = relation_features_from_v9_transition({
        "translation": translation,
        "rotation": rotation,
        "image": image,
        "gripper": gripper,
        "arm_present": present,
        "motion_active": active,
    })
    assert converted.anchored_se3[0, 0].tolist() == [0.0] * 6
    assert converted.anchored_se3[20, 0, 0] == pytest.approx(2.0)
    assert converted.velocity[1, 0, 0] == pytest.approx(0.1)
    assert converted.uv[1, 0].tolist() == [0.0, 1.0]
    assert converted.arm_present[:, 0].all()
    present[1] = False
    absent = relation_features_from_v9_transition({
        "translation": translation,
        "rotation": rotation,
        "image": image,
        "gripper": gripper,
        "arm_present": present,
        "motion_active": active,
    })
    assert not absent.arm_present[:, 1].any()
    assert np.count_nonzero(absent.anchored_se3[:, 1]) == 0
    present[0, 5] = False
    with pytest.raises(ValueError, match="contiguous prefix"):
        relation_features_from_v9_transition({
            "translation": translation, "rotation": rotation, "image": image,
            "gripper": gripper, "arm_present": present, "motion_active": active,
        })
    states = np.broadcast_to(np.eye(4), (2, 21, 4, 4)).copy()
    states[0, :, 0, 3] = np.arange(21) * 0.1
    exact = relation_features_from_v9_transition(
        {
            "translation": translation, "rotation": rotation, "image": image,
            "gripper": gripper, "arm_present": present, "motion_active": active,
        },
        exact_states=states,
        exact_state_present=np.ones((2, 21), dtype=bool),
    )
    assert exact.anchored_se3[20, 0, 0] == pytest.approx(2.0)
    assert not exact.arm_present[6, 0]
