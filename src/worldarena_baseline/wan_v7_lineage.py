"""Source-controlled Stage-A discovery subset contract for Wan v7.

The eight discovery episodes are a deterministic subset of the already-pinned
``clean-1000`` training source.  They are a *mechanism audit* over cached,
zero-leakage training inputs, not an evaluation split.  ``dev-fast20`` stays a
receipt-only exclusion boundary and is never materialized into this cache or
audit.  Discovery is never a substitute for the frozen official test manifest.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Final


DISCOVERY8_CONTRACT: Final = {
    "contract": "wan-action-v7-discovery-derivation/1",
    "source_artifact": "clean1000_manifest",
    "selector": "sample-lexicographic-first-8/v1",
    "rows": 8,
}


def discovery8_contract_from_pin(
    pin: object, *, clean1000_manifest_sha256: str
) -> dict[str, object]:
    """Validate the source-pinned discovery derivation descriptor."""

    if not isinstance(pin, Mapping):
        raise ValueError("v7 discovery lineage pin must be a mapping")
    expected = {**DISCOVERY8_CONTRACT, "source_sha256": clean1000_manifest_sha256}
    if dict(pin) != expected:
        raise ValueError("v7 discovery lineage pin differs from the pinned clean-1000 contract")
    return dict(DISCOVERY8_CONTRACT)


def _validated_contract(value: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("v7 discovery contract must be a mapping")
    expected = dict(DISCOVERY8_CONTRACT)
    if dict(value) != expected:
        raise ValueError("v7 discovery contract or selector differs")
    return expected


def derive_discovery8(
    clean1000_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object] = DISCOVERY8_CONTRACT,
) -> list[dict[str, object]]:
    """Return the only legal eight-row Stage-A audit subset.

    The full clean-1000 manifest is independently hash-pinned before calling
    this function.  Sorting by stable robot identity makes the selected rows
    reproducible from its exact bytes, rather than a cache-worker ordering.
    """

    descriptor = _validated_contract(contract)
    rows = [dict(row) for row in clean1000_rows]
    samples = [row.get("sample") for row in rows]
    if any(not isinstance(sample, str) or not sample for sample in samples):
        raise ValueError("v7 discovery rows require a unique non-empty sample")
    if len(set(samples)) != len(samples):
        raise ValueError("v7 discovery rows require a unique non-empty sample")
    count = int(descriptor["rows"])
    if len(rows) < count:
        raise ValueError("v7 clean-1000 source manifest has fewer than eight rows")
    return sorted(rows, key=lambda row: str(row["sample"]))[:count]


def render_discovery8_jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    """Render only a fully derived discovery-8 manifest in canonical JSONL."""

    canonical_rows = derive_discovery8(rows)
    if [dict(row) for row in rows] != canonical_rows:
        raise ValueError("v7 discovery manifest is not the canonical derived subset")
    return b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8") + b"\n"
        for row in canonical_rows
    )


def validate_discovery8_jsonl(
    content: bytes,
    clean1000_rows: Sequence[Mapping[str, object]],
    contract: Mapping[str, object] = DISCOVERY8_CONTRACT,
) -> list[dict[str, object]]:
    """Reject missing, hand-selected, or tampered discovery manifests."""

    try:
        rows = [json.loads(line) for line in content.decode("utf-8").splitlines() if line]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("v7 discovery manifest is unreadable") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("v7 discovery manifest rows must be JSON objects")
    expected = derive_discovery8(clean1000_rows, contract)
    if rows != expected:
        raise ValueError("v7 discovery manifest differs from fixed derived subset")
    if content != render_discovery8_jsonl(expected):
        raise ValueError("v7 discovery manifest is not canonical JSONL")
    return expected
