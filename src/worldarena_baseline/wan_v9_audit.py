"""Pure decision gates for the bounded Wan v9 experiment."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


FAMILIES = ("shift+1", "shift-1", "reverse", "swap")
GRADIENT_FAMILIES = (
    "native_qkvo", "cross_q", "cross_k", "cross_v", "cross_o", "tokenizer", "gates",
)


def _finite(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result


def step25_health_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    history = metrics.get("gradient_history")
    if not isinstance(history, Mapping):
        history = {}
    for family in GRADIENT_FAMILIES:
        value = _finite(history.get(family))
        if not math.isfinite(value) or value <= 0:
            reasons.append(f"{family}_gradient_not_positive_finite")
    if metrics.get("outside_whitelist_gradients") != 0:
        reasons.append("outside_whitelist_gradient_detected")
    for name in ("fm_loss", "ranking_loss"):
        if not math.isfinite(_finite(metrics.get(name))):
            reasons.append(f"{name}_not_finite")
    if metrics.get("optimizer_finite") is not True:
        reasons.append("optimizer_not_finite")
    ratio = _finite(metrics.get("residual_max_ratio"))
    if not math.isfinite(ratio) or ratio > 10:
        reasons.append("residual_ratio_above_10x")
    if metrics.get("scheduler_drift") is not False:
        reasons.append("scheduler_drift")
    if metrics.get("topology_drift") is not False:
        reasons.append("topology_drift")
    return {"step": 25, "pass": not reasons, "continue": not reasons, "reasons": reasons}


def _family_checks(metrics: Mapping[str, Any], *, threshold: int) -> tuple[list[str], dict[str, tuple[int, float]]]:
    reasons: list[str] = []
    values: dict[str, tuple[int, float]] = {}
    families = metrics.get("families")
    if not isinstance(families, Mapping):
        families = {}
    for family in FAMILIES:
        item = families.get(family)
        if not isinstance(item, Mapping):
            item = {}
        try:
            wins = int(item.get("wins", -1))
        except (TypeError, ValueError):
            wins = -1
        margin = _finite(item.get("mean_margin"))
        try:
            eligible = int(item.get("eligible", -1))
        except (TypeError, ValueError):
            eligible = -1
        values[family] = wins, margin
        if eligible != 20:
            reasons.append(f"{family}_coverage_not_20")
        if wins < threshold:
            reasons.append(f"{family}_wins_below_{threshold}")
        if not math.isfinite(margin) or margin <= 0:
            reasons.append(f"{family}_margin_not_positive")
    return reasons, values


def _shared_mechanism_checks(metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    routing = _finite(metrics.get("routing_retention"))
    fm = _finite(metrics.get("fm_regression"))
    if not math.isfinite(routing) or routing < 0.90:
        reasons.append("routing_retention_below_90_percent")
    if not math.isfinite(fm) or fm > 0.02:
        reasons.append("fm_regression_above_2_percent")
    return reasons


def step100_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reasons, families = _family_checks(metrics, threshold=12)
    shared = _shared_mechanism_checks(metrics)
    reasons.extend(shared)
    hard_stop = any(families[name][0] <= 6 for name in ("shift+1", "shift-1"))
    margins_positive = all(math.isfinite(value[1]) and value[1] > 0 for value in families.values())
    reverse_swap_ready = all(families[name][0] >= 12 for name in ("reverse", "swap"))
    shifts_borderline = all(families[name][0] >= 7 for name in ("shift+1", "shift-1"))
    trend = metrics.get("shift_last_half_trend_positive") is True
    passed = not reasons
    continuation = passed or (
        not hard_stop and not shared and margins_positive and reverse_swap_ready
        and shifts_borderline and trend
    )
    if hard_stop:
        reasons.append("shift_hard_stop_at_or_below_6_of_20")
    elif not passed and not continuation:
        reasons.append("borderline_shift_continuation_rule_failed")
    return {
        "step": 100,
        "pass": passed,
        "continue": continuation,
        "hard_stop": hard_stop,
        "reasons": sorted(set(reasons)),
    }


def step250_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reasons, _families = _family_checks(metrics, threshold=14)
    reasons.extend(_shared_mechanism_checks(metrics))
    if _finite(metrics.get("position_improvement")) <= 0.05:
        reasons.append("position_improvement_not_above_5_percent")
    if _finite(metrics.get("velocity_improvement")) <= 0.05:
        reasons.append("velocity_improvement_not_above_5_percent")
    if metrics.get("gate_enabled_no_worse") is not True:
        reasons.append("gate_enabled_worse_than_gate_zero")
    return {"step": 250, "pass": not reasons, "continue": not reasons, "reasons": reasons}


def step500_gate(metrics: Mapping[str, Any], *, step250_receipt: Mapping[str, Any]) -> dict[str, Any]:
    reasons, _families = _family_checks(metrics, threshold=14)
    reasons.extend(_shared_mechanism_checks(metrics))
    if step250_receipt.get("pass") is not True:
        reasons.append("missing_passing_step250_receipt")
    if metrics.get("position_within_noise_band") is not True:
        reasons.append("position_regressed_beyond_noise_band")
    if metrics.get("velocity_within_noise_band") is not True:
        reasons.append("velocity_regressed_beyond_noise_band")
    if metrics.get("visual_guardrail_regression") is not False:
        reasons.append("visual_guardrail_regression")
    return {"step": 500, "pass": not reasons, "continue": False, "reasons": reasons}

