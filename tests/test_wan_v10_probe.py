from __future__ import annotations

import math

from worldarena_baseline.wan_v10_probe import aggregate_v10_probe


def test_probe_aggregation_is_finite_only_and_failure_aware() -> None:
    report = aggregate_v10_probe(
        [
            {"sample": "a", "valid": True, "error": 2.0},
            {"sample": "b", "valid": True, "error": 4.0},
            {"sample": "c", "valid": False, "reason": "missing_right"},
        ]
    )
    assert report["total_episodes"] == 3
    assert report["finite_episodes"] == 2
    assert report["invalid_episodes"] == 1
    assert report["invalid_reasons"] == {"missing_right": 1}
    assert report["finite_mean_error"] == 3.0
    assert report["finite_median_error"] == 3.0
    assert report["failure_aware_score"] == (1 / 3 + 1 / 5) / 3
    assert math.isfinite(report["failure_aware_score"])


def test_probe_rejects_duplicate_nonfinite_and_silent_invalid_rows() -> None:
    for rows, message in (
        ([{"sample": "a", "valid": True, "error": float("inf")}], "finite"),
        ([{"sample": "a", "valid": False}], "reason"),
        ([{"sample": "a", "valid": True, "error": 1}, {"sample": "a", "valid": True, "error": 2}], "duplicate"),
    ):
        try:
            aggregate_v10_probe(rows)
        except ValueError as exc:
            assert message in str(exc)
        else:
            raise AssertionError("invalid probe rows were accepted")

