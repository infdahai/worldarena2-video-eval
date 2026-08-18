"""Fail-closed source closure validation for the remote Wan v7 sync.

The repository is intentionally dirty with unrelated historical experiments.
This module names only the code directly imported by the v7 cache, trainer,
and launcher.  A remote sync must validate this closure *before* copying it:
every listed file must be regular, committed, clean, and cover the direct
``worldarena_baseline`` imports of both Python entrypoints.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any


CLOSURE_RELATIVE_PATH = "source_inputs/wan-v7-stagea-sync-closure.json"
_SCHEMA = "wan-action-v7-stagea-sync-closure/1"
_PYTHON_ENTRYPOINTS = (
    "scripts/cache_wan_v7_se3_conditions.py",
    "scripts/train_wan_se3_probe_v7_fsdp.py",
)
_REQUIRED_ENTRYPOINTS = (*_PYTHON_ENTRYPOINTS, "scripts/run_wan_se3_probe_v7.sh")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_file_list(repo_root: Path) -> tuple[str, ...]:
    path = repo_root / CLOSURE_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v7 sync closure manifest is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {"schema", "files"}:
        raise RuntimeError("v7 sync closure manifest schema is invalid")
    if payload["schema"] != _SCHEMA or not isinstance(payload["files"], list):
        raise RuntimeError("v7 sync closure manifest contract differs")
    files = tuple(payload["files"])
    if not files or any(not isinstance(item, str) or not item for item in files):
        raise RuntimeError("v7 sync closure file list is invalid")
    if tuple(sorted(files)) != files or len(set(files)) != len(files):
        raise RuntimeError("v7 sync closure file list must be sorted and unique")
    if CLOSURE_RELATIVE_PATH not in files:
        raise RuntimeError("v7 sync closure must include itself")
    if any(entrypoint not in files for entrypoint in _REQUIRED_ENTRYPOINTS):
        raise RuntimeError("v7 sync closure is missing an entrypoint")
    return files


def _direct_internal_imports(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"v7 sync entrypoint is unreadable: {path}") from exc
    imports: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom) or node.module is None:
            continue
        if node.module == "worldarena_baseline":
            for alias in node.names:
                imports.add(f"src/worldarena_baseline/{alias.name}.py")
        elif node.module.startswith("worldarena_baseline."):
            imports.add(f"src/{node.module.replace('.', '/')}.py")
    return imports


def _run_git(repo_root: Path, *args: str) -> None:
    completed = subprocess.run(
        ["git", "-C", os.fspath(repo_root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"v7 sync closure git validation failed: {detail}")


def validate_sync_closure(repo_root: Path | str) -> dict[str, Any]:
    """Validate and fingerprint the exact source set allowed to reach remote.

    This has no remote side effects.  Call it locally before ``rsync`` and
    again remotely after copying, so an absent, untracked, modified, or
    incomplete source closure fails before cache/preflight/GPU work.
    """

    root = Path(repo_root).resolve()
    files = _load_file_list(root)
    direct_imports: set[str] = set()
    for entrypoint in _PYTHON_ENTRYPOINTS:
        direct_imports.update(_direct_internal_imports(root / entrypoint))
    missing_imports = sorted(direct_imports - set(files))
    if missing_imports:
        raise RuntimeError(
            "v7 sync closure omits direct internal imports: " + ", ".join(missing_imports)
        )
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v7 sync closure source is missing or non-regular: {relative}")
        _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        _run_git(root, "diff", "--quiet", "--", relative)
        _run_git(root, "diff", "--cached", "--quiet", "--", relative)
    entries = [{"path": relative, "sha256": _sha256(root / relative)} for relative in files]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "contract": _SCHEMA,
        "files": entries,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }
