from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.train_wan_v11_bimanual import (
    FORMAL_ROOT,
    _formal,
    build_v11_replay_rows,
)


def test_formal_path_guard_rejects_escape() -> None:
    assert _formal(FORMAL_ROOT / "runs/v11") == FORMAL_ROOT / "runs/v11"
    with pytest.raises(ValueError, match="escapes formal root"):
        _formal(Path("/tmp/v11"))


def test_replay_is_exact_deterministic_exposure_and_zero_leakage() -> None:
    optimizer = [
        {"sample": f"train-{index}", "v10_stratum": "single_dominant"}
        for index in range(2060)
    ]
    audit = [{"sample": f"audit-{index}"} for index in range(20)]

    first = build_v11_replay_rows(optimizer, audit)
    second = build_v11_replay_rows(list(reversed(optimizer)), audit)

    assert first == second
    assert len(first) == 2060
    assert [row["optimizer_step"] for row in first] == list(range(1, 2061))
    assert {row["sample"] for row in first} == {row["sample"] for row in optimizer}
    assert not ({row["sample"] for row in first} & {row["sample"] for row in audit})
    assert [row["negative_family"] for row in first[:6]] == [
        "wrong-left",
        "wrong-right",
        "active-arm-null",
        "wrong-left",
        "wrong-right",
        "active-arm-null",
    ]


def test_replay_rejects_duplicate_or_leaked_samples() -> None:
    optimizer = [{"sample": f"train-{index}"} for index in range(2060)]
    optimizer[-1] = dict(optimizer[0])
    with pytest.raises(ValueError, match="unique"):
        build_v11_replay_rows(optimizer, [])
    optimizer[-1] = {"sample": "audit-0"}
    with pytest.raises(ValueError, match="leaks audit"):
        build_v11_replay_rows(optimizer, [{"sample": "audit-0"}])


def test_replay_rows_are_json_serializable() -> None:
    optimizer = [{"sample": f"train-{index}"} for index in range(2060)]
    rows = build_v11_replay_rows(optimizer, [])
    assert json.loads(json.dumps(rows[0]))["contract"] == "wan-v11-replay-row/1"
