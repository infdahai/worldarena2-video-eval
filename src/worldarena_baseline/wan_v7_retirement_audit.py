"""Deterministic 20-episode retirement audit for the v7 gate-only probe."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final


RETIREMENT20_CONTRACT: Final = "wan-action-v7-gate-only-retirement-audit/1"
_TARGET_CLASSES: Final = ("left", "right", "both") * 6 + ("left", "right")
_COUNTERFACTUALS: Final = ("reverse", "shift", "swap")


def _task(row: Mapping[str, object]) -> str:
    task = row.get("task")
    if isinstance(task, str) and task:
        return task
    sample = row.get("sample")
    if isinstance(sample, str) and "__" in sample:
        return sample.split("__", 1)[0]
    raise ValueError("retirement audit rows require task identity")


def _observability_class(value: tuple[bool, bool]) -> str | None:
    left, right = value
    if left and right:
        return "both"
    if left:
        return "left"
    if right:
        return "right"
    return None


def select_retirement20(
    clean1000_rows: Sequence[Mapping[str, object]],
    observability: Mapping[str, tuple[bool, bool]],
) -> list[dict[str, object]]:
    """Select one observable episode from each of 20 distinct tasks.

    The class schedule is source controlled (7 left, 7 right, 6 both).  Within
    a class, the lexicographically first still-unused task and sample win.  No
    evaluator split participates in this training-internal mechanism audit.
    """

    candidates: dict[str, dict[str, list[dict[str, object]]]] = {}
    seen_samples: set[str] = set()
    for source in clean1000_rows:
        row = dict(source)
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample or sample in seen_samples:
            raise ValueError("retirement audit rows require unique sample identity")
        seen_samples.add(sample)
        arm_class = _observability_class(observability.get(sample, (False, False)))
        if arm_class is None:
            continue
        task = _task(row)
        row["task"] = task
        row["observability_class"] = arm_class
        candidates.setdefault(arm_class, {}).setdefault(task, []).append(row)

    for task_values in candidates.values():
        for values in task_values.values():
            values.sort(key=lambda row: str(row["sample"]))

    selected: list[dict[str, object]] = []
    used_tasks: set[str] = set()
    for arm_class in _TARGET_CLASSES:
        available_tasks = sorted(set(candidates.get(arm_class, {})) - used_tasks)
        if not available_tasks:
            raise ValueError("retirement audit requires 20 distinct observable tasks")
        task = available_tasks[0]
        selected.append(candidates[arm_class][task][0])
        used_tasks.add(task)
    if len(selected) != 20 or len(used_tasks) != 20:
        raise ValueError("retirement audit requires 20 distinct observable tasks")
    return selected


def _finite(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def aggregate_retirement_audit(
    episodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate paired correct-vs-counterfactual metrics without survivor bias."""

    if len(episodes) != 20:
        raise ValueError("retirement audit requires exactly 20 episodes")
    counterfactual: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    for variant in _COUNTERFACTUALS:
        position_deltas: list[float] = []
        velocity_deltas: list[float] = []
        wins = 0
        for episode in episodes:
            metrics = episode.get("metrics")
            if not isinstance(metrics, Mapping):
                continue
            correct = metrics.get("correct")
            changed = metrics.get(variant)
            if not isinstance(correct, Mapping) or not isinstance(changed, Mapping):
                continue
            correct_position = _finite(correct.get("position_error"))
            changed_position = _finite(changed.get("position_error"))
            if correct_position is None or changed_position is None:
                continue
            position_deltas.append(changed_position - correct_position)
            wins += int(correct_position < changed_position)
            correct_velocity = _finite(correct.get("velocity_error"))
            changed_velocity = _finite(changed.get("velocity_error"))
            if correct_velocity is not None and changed_velocity is not None:
                velocity_deltas.append(changed_velocity - correct_velocity)
        finite_pairs = len(position_deltas)
        if finite_pairs != 20:
            failures.append("not_all_20_pairs_finite")
        if wins < 12:
            failures.append(f"{variant}_wins_below_12_of_20")
        position_improvement = (
            sum(position_deltas) / finite_pairs if finite_pairs else None
        )
        velocity_improvement = (
            sum(velocity_deltas) / len(velocity_deltas) if velocity_deltas else None
        )
        if position_improvement is None or position_improvement <= 0:
            failures.append(f"{variant}_position_not_positive")
        if velocity_improvement is None or velocity_improvement <= 0:
            failures.append(f"{variant}_velocity_not_positive")
        counterfactual[variant] = {
            "wins": wins,
            "finite_pairs": finite_pairs,
            "win_rate": wins / finite_pairs if finite_pairs else None,
            "position_improvement": position_improvement,
            "velocity_improvement": velocity_improvement,
        }
    unique_failures = list(dict.fromkeys(failures))
    return {
        "contract": RETIREMENT20_CONTRACT,
        "counterfactual": counterfactual,
        "stable_separation": not unique_failures,
        "failure_reasons": unique_failures,
    }
