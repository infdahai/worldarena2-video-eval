import hashlib

import numpy as np
import pytest

from worldarena_baseline.wan_se3_condition import SE3Condition, write_se3_cache_atomic
from worldarena_baseline.wan_v8_data import (
    V8_AUDIT_SEED,
    build_v8_data_receipt,
    build_v8_replay,
    validate_v8_correct_cache,
)


TARGET = {
    "single_dominant": 0.45,
    "bimanual_heavy": 0.30,
    "mixed": 0.15,
    "quiet": 0.10,
}


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _rows(prefix: str, count: int) -> list[dict[str, str]]:
    return [{"sample": f"{prefix}-{index:03d}", "task": f"task-{index % 5}"} for index in range(count)]


def _roles(rows: list[dict[str, str]]) -> dict[str, str]:
    labels = tuple(TARGET)
    return {row["sample"]: labels[index % len(labels)] for index, row in enumerate(rows)}


def test_audit20_comes_from_probe_heldout_and_is_not_trainable():
    rows = _rows("clean", 64)
    heldout_rows = rows[20:]
    dev_rows = [{"sample": "external-dev-000"}]
    receipt = build_v8_data_receipt(
        cached_rows=rows,
        probe_split={"heldout_rows": heldout_rows},
        dev_rows=dev_rows,
        sample_roles=_roles(rows),
        seed=V8_AUDIT_SEED,
        steps=3,
        world_size=1,
    )
    assert len(receipt.audit_samples) == 20
    assert set(receipt.audit_samples) <= {row["sample"] for row in heldout_rows}
    assert set(receipt.audit_samples).isdisjoint(receipt.optimizer_samples)
    assert set(receipt.optimizer_samples).isdisjoint({row["sample"] for row in dev_rows})
    assert receipt.audit_samples == build_v8_data_receipt(
        cached_rows=rows,
        probe_split={"heldout_rows": heldout_rows},
        dev_rows=dev_rows,
        sample_roles=_roles(rows),
        seed=V8_AUDIT_SEED,
        steps=3,
        world_size=1,
    ).audit_samples


def test_replay_binds_rank_step_noise_and_full_action_negative_cycle():
    rows = _rows("clean", 40)
    replay = build_v8_replay(
        optimizer_samples=[row["sample"] for row in rows],
        sample_roles=_roles(rows),
        steps=5,
        world_size=2,
        seed=7,
    )
    assert len(replay) == 10
    assert {(row["optimizer_step"], row["rank"]) for row in replay} == {
        (step, rank) for step in range(1, 6) for rank in range(2)
    }
    assert {row["negative_family"] for row in replay} == {"reverse", "shift", "swap"}
    assert {row["shift_direction"] for row in replay if row["negative_family"] == "shift"} == {-1, 1}
    assert all(isinstance(row["noise_seed"], int) and isinstance(row["timestep_seed"], int) for row in replay)


def test_correct_se3_cache_requires_all_expected_samples(tmp_path):
    samples = ("a", "b")
    digest = _sha("manifest")
    condition = SE3Condition(
        arm_transform=np.broadcast_to(np.eye(4, dtype=np.float32), (2, 21, 4, 4)).copy(),
        arm_present=np.zeros((2, 21), dtype=bool),
        anchor_arm="identity",
        motion_scale=1.0,
    )
    write_se3_cache_atomic(
        tmp_path / "wan_v8_se3_conditions" / "a.npz",
        condition,
        source_episode_sha256=_sha("a"),
        source_manifest_sha256=digest,
    )
    with pytest.raises(ValueError, match="SE3 coverage 1/2"):
        validate_v8_correct_cache(tmp_path, expected_samples=samples, source_manifest_sha256=digest)


def test_correct_se3_cache_rejects_wrong_source_and_accepts_complete_cache(tmp_path):
    digest = _sha("manifest")
    condition = SE3Condition(
        arm_transform=np.broadcast_to(np.eye(4, dtype=np.float32), (2, 21, 4, 4)).copy(),
        arm_present=np.zeros((2, 21), dtype=bool),
        anchor_arm="identity",
        motion_scale=1.0,
    )
    for sample in ("a", "b"):
        write_se3_cache_atomic(
            tmp_path / "wan_v8_se3_conditions" / f"{sample}.npz",
            condition,
            source_episode_sha256=_sha(sample),
            source_manifest_sha256=digest,
        )
    payload = validate_v8_correct_cache(
        tmp_path, expected_samples=("a", "b"), source_manifest_sha256=digest
    )
    assert payload["valid_samples"] == ("a", "b")
    with pytest.raises(ValueError, match="source manifest hash differs"):
        validate_v8_correct_cache(
            tmp_path, expected_samples=("a", "b"), source_manifest_sha256=_sha("other")
        )
