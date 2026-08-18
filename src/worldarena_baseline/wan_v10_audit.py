"""Pure fail-closed gates for the bounded Wan v10 run."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any


FAMILIES = ("reverse", "swap", "phase+1", "phase-1")
GRADIENT_FAMILIES = (
    "native_qkvo",
    "relation_encoders",
    "relation_gates",
    "hidden_eef_heads",
)


def _finite(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _shared(metrics: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not math.isfinite(_finite(metrics.get("routing_retention"))) or _finite(metrics.get("routing_retention")) < 0.90:
        reasons.append("routing_retention_below_90_percent")
    if not math.isfinite(_finite(metrics.get("fm_regression"))) or _finite(metrics.get("fm_regression")) > 0.02:
        reasons.append("fm_regression_above_2_percent")
    return reasons


def _family_reasons(
    metrics: Mapping[str, Any], threshold: int, *, required: tuple[str, ...] = FAMILIES
) -> list[str]:
    reasons: list[str] = []
    families = metrics.get("families")
    if not isinstance(families, Mapping):
        families = {}
    for family in required:
        item = families.get(family)
        if not isinstance(item, Mapping):
            item = {}
        try:
            wins = int(item.get("wins", -1))
            eligible = int(item.get("eligible", -1))
        except (TypeError, ValueError):
            wins, eligible = -1, -1
        margin = _finite(item.get("mean_margin"))
        if eligible != 20:
            reasons.append(f"{family}_coverage_not_20")
        if wins < threshold:
            reasons.append(f"{family}_wins_below_{threshold}")
        if not math.isfinite(margin) or margin <= 0:
            reasons.append(f"{family}_margin_not_positive")
    return reasons


def evaluate_v10_gate(
    step: int,
    metrics: Mapping[str, Any],
    *,
    step300_receipt: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if step == 50:
        gradients = metrics.get("gradient_history")
        if not isinstance(gradients, Mapping):
            gradients = {}
        for family in GRADIENT_FAMILIES:
            value = _finite(gradients.get(family))
            if not math.isfinite(value) or value <= 0:
                reasons.append(f"{family}_gradient_not_positive_finite")
        if metrics.get("optimizer_finite") is not True:
            reasons.append("optimizer_not_finite")
        if metrics.get("outside_whitelist_gradients") != 0:
            reasons.append("outside_whitelist_gradient_detected")
        if metrics.get("scheduler_drift") is not False:
            reasons.append("scheduler_drift")
        if metrics.get("topology_drift") is not False:
            reasons.append("topology_drift")
        reasons.extend(_shared(metrics))
    elif step == 150:
        reasons.extend(
            _family_reasons(metrics, 11, required=("reverse", "swap"))
        )
        reasons.extend(_shared(metrics))
        if metrics.get("hidden_eef_finite") is not True:
            reasons.append("hidden_eef_not_finite")
    elif step in (300, 500):
        threshold = 11 if step == 300 else 14
        reasons.extend(_family_reasons(metrics, threshold))
        reasons.extend(_shared(metrics))
        if step == 500:
            position = _finite(metrics.get("position_improvement"))
            velocity = _finite(metrics.get("velocity_improvement"))
            if not math.isfinite(position) or position <= 0.05:
                reasons.append("position_improvement_not_above_5_percent")
            if not math.isfinite(velocity) or velocity <= 0.05:
                reasons.append("velocity_improvement_not_above_5_percent")
        if metrics.get("relation_enabled_beats_zero") is not True:
            reasons.append("relation_enabled_does_not_beat_zero")
        if step == 500 and (
            not isinstance(step300_receipt, Mapping)
            or step300_receipt.get("pass") is not True
        ):
            reasons.append("missing_passing_step300_receipt")
    else:
        raise ValueError("v10 gate step must be 50, 150, 300, or 500")
    return {"contract": "wan-v10-gate/1", "step": step, "pass": not reasons, "reasons": reasons}
