from __future__ import annotations

import importlib.util


def test_v9_objective_module_exists() -> None:
    assert importlib.util.find_spec("worldarena_baseline.wan_v9_objective") is not None
