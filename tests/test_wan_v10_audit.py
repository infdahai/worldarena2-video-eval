from __future__ import annotations

from worldarena_baseline.wan_v10_audit import evaluate_v10_gate


def _families(wins: int):
    return {
        name: {"wins": wins, "eligible": 20, "mean_margin": 0.1}
        for name in ("reverse", "swap", "phase+1", "phase-1")
    }


def test_step150_semantics_gate_checks_reverse_swap_only() -> None:
    metrics = {
        "families": _families(11),
        "hidden_eef_finite": True,
        "routing_retention": 0.90,
        "fm_regression": 0.02,
    }
    metrics["families"]["phase+1"]["wins"] = 0
    assert evaluate_v10_gate(150, metrics)["pass"] is True
    metrics["families"]["swap"]["wins"] = 10
    assert evaluate_v10_gate(150, metrics)["pass"] is False


def test_step300_gate_uses_registered_timing_thresholds() -> None:
    metrics = {
        "families": _families(11),
        "routing_retention": 0.90,
        "fm_regression": 0.02,
        "relation_enabled_beats_zero": True,
    }
    assert evaluate_v10_gate(300, metrics)["pass"] is True
    metrics["families"]["phase+1"]["wins"] = 10
    result = evaluate_v10_gate(300, metrics)
    assert result["pass"] is False
    assert "phase+1_wins_below_11" in result["reasons"]


def test_step500_gate_requires_fourteen_wins_and_five_percent_trajectory() -> None:
    metrics = {
        "families": _families(14),
        "position_improvement": 0.051,
        "velocity_improvement": 0.051,
        "routing_retention": 0.91,
        "fm_regression": 0.019,
        "relation_enabled_beats_zero": True,
    }
    assert evaluate_v10_gate(500, metrics, step300_receipt={"pass": True})["pass"] is True
    metrics["velocity_improvement"] = 0.05
    result = evaluate_v10_gate(500, metrics, step300_receipt={"pass": True})
    assert result["pass"] is False
    assert "velocity_improvement_not_above_5_percent" in result["reasons"]


def test_step50_health_gate_requires_every_trainable_family_update() -> None:
    metrics = {
        "gradient_history": {
            "native_qkvo": 1.0,
            "relation_encoders": 1.0,
            "relation_gates": 1.0,
            "hidden_eef_heads": 1.0,
        },
        "optimizer_finite": True,
        "outside_whitelist_gradients": 0,
        "fm_regression": 0.01,
        "routing_retention": 0.95,
        "scheduler_drift": False,
        "topology_drift": False,
    }
    assert evaluate_v10_gate(50, metrics)["pass"] is True
    metrics["gradient_history"]["hidden_eef_heads"] = 0
    assert evaluate_v10_gate(50, metrics)["pass"] is False
