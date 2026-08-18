from __future__ import annotations

import math

import pytest


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task_index in range(24):
        task = f"task_{task_index:02d}"
        for episode_index in range(3):
            rows.append(
                {
                    "sample": f"{task}__episode_{episode_index:06d}",
                    "task": task,
                }
            )
    return rows


def _observability() -> dict[str, tuple[bool, bool]]:
    result: dict[str, tuple[bool, bool]] = {}
    kinds = ((True, False), (False, True), (True, True))
    for row in _rows():
        sample = str(row["sample"])
        episode_index = int(sample.rsplit("_", 1)[1])
        result[sample] = kinds[episode_index]
    return result


def test_retirement20_is_deterministic_multitask_and_arm_balanced() -> None:
    from worldarena_baseline.wan_v7_retirement_audit import select_retirement20

    selected = select_retirement20(list(reversed(_rows())), _observability())

    assert len(selected) == 20
    assert len({row["task"] for row in selected}) == 20
    assert [row["observability_class"] for row in selected].count("left") == 7
    assert [row["observability_class"] for row in selected].count("right") == 7
    assert [row["observability_class"] for row in selected].count("both") == 6
    assert select_retirement20(_rows(), _observability()) == selected


def test_retirement20_rejects_insufficient_tasks_or_unobservable_selection() -> None:
    from worldarena_baseline.wan_v7_retirement_audit import select_retirement20

    with pytest.raises(ValueError, match="20 distinct observable tasks"):
        select_retirement20(_rows()[:30], _observability())

    observability = _observability()
    for sample in list(observability):
        observability[sample] = (False, False)
    with pytest.raises(ValueError, match="20 distinct observable tasks"):
        select_retirement20(_rows(), observability)


def test_retirement_audit_uses_finite_pairs_and_requires_all_families() -> None:
    from worldarena_baseline.wan_v7_retirement_audit import aggregate_retirement_audit

    episodes = []
    for index in range(20):
        correct = {"position_error": 1.0, "velocity_error": 2.0, "fm_loss": 3.0}
        variants = {
            "reverse": {"position_error": 2.0 if index < 14 else 0.5, "velocity_error": 3.0, "fm_loss": 3.0},
            "shift": {"position_error": 2.0 if index < 13 else 0.5, "velocity_error": 3.0, "fm_loss": 3.0},
            "swap": {"position_error": 2.0 if index < 12 else 0.5, "velocity_error": 3.0, "fm_loss": 3.0},
        }
        episodes.append({"sample": f"sample-{index}", "metrics": {"correct": correct, **variants}})

    report = aggregate_retirement_audit(episodes)

    assert report["stable_separation"] is True
    assert report["counterfactual"]["reverse"]["wins"] == 14
    assert report["counterfactual"]["shift"]["wins"] == 13
    assert report["counterfactual"]["swap"]["wins"] == 12
    assert report["counterfactual"]["swap"]["win_rate"] == pytest.approx(0.6)

    episodes[0]["metrics"]["swap"]["position_error"] = math.inf
    report = aggregate_retirement_audit(episodes)
    assert report["counterfactual"]["swap"]["finite_pairs"] == 19
    assert report["stable_separation"] is False
    assert "not_all_20_pairs_finite" in report["failure_reasons"]
