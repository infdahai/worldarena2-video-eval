"""Pure aggregation and gate attribution for the fixed v8 audit20."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .wan_v8_training import step100_gate, step250_gate


def hard_shift_energy(row: Mapping[str, float]) -> float:
    return min(float(row["shift_plus_energy"]), float(row["shift_minus_energy"]))


def aggregate_v8_audit(rows: Sequence[Mapping[str, Any]], *, step: int) -> dict[str, Any]:
    if len(rows) != 20:
        raise ValueError("v8 audit requires exactly 20 episodes")
    families: dict[str, dict[str, float | int]] = {}
    for family in ("reverse", "shift", "swap"):
        margins: list[float] = []
        for row in rows:
            correct = float(row["correct_energy"])
            wrong = hard_shift_energy(row) if family == "shift" else float(row[f"{family}_energy"])
            margins.append(wrong - correct)
        families[family] = {"wins": sum(value > 0 for value in margins), "mean_margin": sum(margins) / len(margins)}
    scalar_keys = ("routing_retention", "fm_regression")
    metrics: dict[str, Any] = {**families}
    for key in scalar_keys:
        metrics[key] = sum(float(row[key]) for row in rows) / len(rows)
    if step == 100:
        for key in ("position_regression", "velocity_regression"):
            metrics[key] = sum(float(row[key]) for row in rows) / len(rows)
        decision = step100_gate(metrics)
    elif step == 250:
        for key in ("position_improvement", "velocity_improvement"):
            metrics[key] = sum(float(row[key]) for row in rows) / len(rows)
        decision = step250_gate(metrics)
    else:
        raise ValueError("v8 formal audit is only step100 or step250")
    return {"contract": "wan-v8-audit/1", "step": step, "metrics": metrics, "decision": decision}
