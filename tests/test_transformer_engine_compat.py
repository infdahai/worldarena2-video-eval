from __future__ import annotations

import ast
from pathlib import Path

from packaging.version import Version


def test_compat_version_is_valid_pep440() -> None:
    init_file = (
        Path(__file__).parents[1] / "compat" / "transformer_engine" / "__init__.py"
    )
    tree = ast.parse(init_file.read_text(encoding="utf-8"))
    version = next(
        node.value.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
    )

    assert Version(version).release >= (2, 11, 0)
