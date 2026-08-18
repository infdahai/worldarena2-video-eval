from __future__ import annotations

from worldarena_baseline.wan_v9_audit import (
    step25_health_gate,
    step100_gate,
    step250_gate,
    step500_gate,
)


def _families(wins: int = 14, margin: float = 0.1) -> dict[str, dict[str, float]]:
    return {
        family: {"wins": wins, "mean_margin": margin, "eligible": 20}
        for family in ("shift+1", "shift-1", "reverse", "swap")
    }


def test_step25_requires_every_gradient_family_after_step_one() -> None:
    metrics = {
        "gradient_history": {name: 1.0 for name in (
            "native_qkvo", "cross_q", "cross_k", "cross_v", "cross_o", "tokenizer", "gates",
        )},
        "outside_whitelist_gradients": 0,
        "fm_loss": 1.0,
        "ranking_loss": 0.5,
        "optimizer_finite": True,
        "residual_max_ratio": 2.0,
        "scheduler_drift": False,
        "topology_drift": False,
    }
    assert step25_health_gate(metrics)["pass"]
    metrics["gradient_history"]["cross_k"] = 0.0
    result = step25_health_gate(metrics)
    assert not result["pass"]
    assert "cross_k_gradient_not_positive_finite" in result["reasons"]


def test_step100_hard_stops_dead_shift_and_allows_only_positive_borderline_trend() -> None:
    metrics = {
        "families": _families(wins=12),
        "routing_retention": 0.90,
        "fm_regression": 0.02,
        "shift_last_half_trend_positive": True,
    }
    assert step100_gate(metrics)["pass"]
    metrics["families"]["shift+1"]["wins"] = 6
    result = step100_gate(metrics)
    assert result["hard_stop"] and not result["continue"]

    metrics["families"]["shift+1"]["wins"] = 9
    metrics["families"]["shift-1"]["wins"] = 10
    result = step100_gate(metrics)
    assert not result["pass"] and result["continue"]
    metrics["shift_last_half_trend_positive"] = False
    assert not step100_gate(metrics)["continue"]


def test_step250_requires_all_four_families_and_mechanism_improvement() -> None:
    metrics = {
        "families": _families(),
        "routing_retention": 0.90,
        "fm_regression": 0.02,
        "position_improvement": 0.051,
        "velocity_improvement": 0.051,
        "gate_enabled_no_worse": True,
    }
    assert step250_gate(metrics)["pass"]
    metrics["families"]["shift-1"]["wins"] = 13
    assert not step250_gate(metrics)["pass"]
    metrics["families"]["shift-1"]["wins"] = 14
    metrics["position_improvement"] = 0.05
    assert not step250_gate(metrics)["pass"]


def test_step500_retains_mechanism_and_quality_guards() -> None:
    metrics = {
        "families": _families(),
        "routing_retention": 0.92,
        "fm_regression": 0.01,
        "position_within_noise_band": True,
        "velocity_within_noise_band": True,
        "visual_guardrail_regression": False,
    }
    assert step500_gate(metrics, step250_receipt={"pass": True})["pass"]
    assert not step500_gate(metrics, step250_receipt={"pass": False})["pass"]
    metrics["visual_guardrail_regression"] = True
    assert not step500_gate(metrics, step250_receipt={"pass": True})["pass"]
