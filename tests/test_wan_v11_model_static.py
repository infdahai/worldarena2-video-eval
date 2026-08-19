from __future__ import annotations

import os
from pathlib import Path

from worldarena_baseline.wan_v11_model import validate_upstream_wan_forward_source


def test_upstream_wan_forward_structure_is_still_pinned() -> None:
    root = Path(os.environ.get("WAN2_2_SOURCE_ROOT", "/home/huazhi/nlh/Wan2.2"))
    validate_upstream_wan_forward_source((root / "wan/modules/model.py").read_text())
