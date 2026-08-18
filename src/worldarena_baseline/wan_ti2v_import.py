"""Load Wan TI2V submodules without importing unrelated optional pipelines."""

from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path


def _find_wan_dir(source_root: Path | None) -> Path:
    if source_root is not None:
        candidate = Path(source_root).expanduser().resolve() / "wan"
        if not candidate.is_dir():
            raise FileNotFoundError(f"Wan package directory does not exist: {candidate}")
        return candidate

    for entry in sys.path:
        root = Path(entry or ".").expanduser()
        candidate = root / "wan"
        if (candidate / "modules" / "model.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        "Wan package directory was not found on sys.path; add the Wan2.2 source "
        "root to PYTHONPATH or pass source_root"
    )


def install_wan_ti2v_package(source_root: Path | str | None = None) -> types.ModuleType:
    """Register a lightweight ``wan`` package without executing ``wan/__init__.py``.

    Upstream Wan2.2 imports every generation pipeline from its package initializer.
    That makes the TI2V training path depend on S2V-only packages such as librosa.
    Registering the package path directly keeps normal relative imports working while
    avoiding initialization of pipelines that Track 1 never uses.
    """

    wan_dir = _find_wan_dir(Path(source_root) if source_root is not None else None)
    existing = sys.modules.get("wan")
    if existing is not None:
        existing_paths = [Path(path).resolve() for path in getattr(existing, "__path__", ())]
        if wan_dir in existing_paths:
            return existing
        raise RuntimeError(
            f"a different wan package is already imported: {existing_paths or existing!r}"
        )

    package = types.ModuleType("wan")
    package.__file__ = str(wan_dir / "__init__.py")
    package.__package__ = "wan"
    package.__path__ = [str(wan_dir)]
    spec = importlib.machinery.ModuleSpec("wan", loader=None, is_package=True)
    spec.submodule_search_locations = package.__path__
    package.__spec__ = spec
    sys.modules["wan"] = package
    return package
