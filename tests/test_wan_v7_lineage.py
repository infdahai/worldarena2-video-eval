from __future__ import annotations

import json

import pytest


def _rows() -> list[dict[str, object]]:
    return [
        {"sample": f"clean-{index:04d}", "hdf5": f"clean-{index:04d}.h5"}
        for index in reversed(range(1000))
    ]


def test_discovery8_is_a_canonical_fixed_subset_of_the_pinned_clean1000() -> None:
    """Stage A cannot choose a caller-supplied audit subset or touch dev-fast20."""
    from worldarena_baseline.wan_v7_lineage import (
        DISCOVERY8_CONTRACT,
        derive_discovery8,
        render_discovery8_jsonl,
    )

    derived = derive_discovery8(_rows(), DISCOVERY8_CONTRACT)

    assert [row["sample"] for row in derived] == [f"clean-{index:04d}" for index in range(8)]
    rendered = render_discovery8_jsonl(derived)
    assert rendered.startswith(b'{"hdf5":"clean-0000.h5","sample":"clean-0000"}\n')
    assert [json.loads(line) for line in rendered.decode().splitlines()] == derived


def test_discovery8_rejects_duplicate_or_missing_sample_identity() -> None:
    from worldarena_baseline.wan_v7_lineage import DISCOVERY8_CONTRACT, derive_discovery8

    rows = _rows()
    rows[0] = {"sample": "clean-0001", "hdf5": "duplicate.h5"}
    with pytest.raises(ValueError, match="unique non-empty sample"):
        derive_discovery8(rows, DISCOVERY8_CONTRACT)


def test_discovery8_contract_rejects_a_different_selector() -> None:
    from worldarena_baseline.wan_v7_lineage import DISCOVERY8_CONTRACT, derive_discovery8

    altered = dict(DISCOVERY8_CONTRACT, selector="caller-chosen/v1")
    with pytest.raises(ValueError, match="selector"):
        derive_discovery8(_rows(), altered)


def test_discovery8_rejects_an_absent_or_hand_selected_manifest() -> None:
    from worldarena_baseline.wan_v7_lineage import validate_discovery8_jsonl

    with pytest.raises(ValueError, match="differs"):
        validate_discovery8_jsonl(b"", _rows())
    with pytest.raises(ValueError, match="differs"):
        validate_discovery8_jsonl(b'{"sample":"clean-0999"}\n', _rows())
