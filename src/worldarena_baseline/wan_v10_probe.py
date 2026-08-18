"""Finite-only and failure-aware aggregation for v10 trajectory probes."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any


def aggregate_v10_probe(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("v10 probe rows are empty")
    samples: set[str] = set()
    finite_errors: list[float] = []
    invalid_reasons: Counter[str] = Counter()
    failure_aware_values: list[float] = []
    normalized: list[dict[str, Any]] = []
    for source in rows:
        row = dict(source)
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample:
            raise ValueError("v10 probe sample identity is invalid")
        if sample in samples:
            raise ValueError("v10 probe contains a duplicate sample")
        samples.add(sample)
        if row.get("valid") is True:
            try:
                error = float(row["error"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("v10 valid probe row lacks finite error") from exc
            if not math.isfinite(error) or error < 0:
                raise ValueError("v10 valid probe error must be finite and non-negative")
            finite_errors.append(error)
            failure_aware_values.append(1.0 / (1.0 + error))
            normalized.append({"sample": sample, "valid": True, "error": error})
        elif row.get("valid") is False:
            reason = row.get("reason")
            if not isinstance(reason, str) or not reason:
                raise ValueError("v10 invalid probe row must declare a reason")
            invalid_reasons[reason] += 1
            failure_aware_values.append(0.0)
            normalized.append({"sample": sample, "valid": False, "reason": reason})
        else:
            raise ValueError("v10 probe row must declare boolean valid")
    finite_mean = statistics.fmean(finite_errors) if finite_errors else None
    finite_median = statistics.median(finite_errors) if finite_errors else None
    failure_aware_score = statistics.fmean(failure_aware_values)
    if not math.isfinite(failure_aware_score):
        raise ValueError("v10 failure-aware score is non-finite")
    return {
        "contract": "wan-v10-probe-aggregate/1",
        "total_episodes": len(rows),
        "finite_episodes": len(finite_errors),
        "invalid_episodes": len(rows) - len(finite_errors),
        "invalid_reasons": dict(sorted(invalid_reasons.items())),
        "finite_mean_error": finite_mean,
        "finite_median_error": finite_median,
        "failure_aware_score": failure_aware_score,
        "rows": normalized,
    }

