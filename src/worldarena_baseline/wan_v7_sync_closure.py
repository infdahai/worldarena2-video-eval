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
    "scripts/build_wan_v7_replay.py",
    "scripts/cache_wan_v7_se3_conditions.py",
    "scripts/train_wan_se3_probe_v7_fsdp.py",
)
_STATIC_REQUIRED_FILES = (
    *_PYTHON_ENTRYPOINTS,
    "scripts/run_wan_se3_probe_v7.sh",
    "scripts/run_wan_se3_probe_v7_single_gpu.sh",
    "scripts/run_wan_se3_geometry_lora_v71_single_gpu.sh",
    "scripts/validate_wan_v7_sync_closure.py",
    "source_inputs/trusted-wan-v7-se3-lineage-pins.json",
    CLOSURE_RELATIVE_PATH,
    "src/worldarena_baseline/wan_v7_sync_closure.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_file_list(repo_root: Path) -> tuple[tuple[str, ...], dict[str, str]]:
    path = repo_root / CLOSURE_RELATIVE_PATH
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("v7 sync closure manifest is unreadable") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "schema",
        "files",
        "legacy_dirty_exact_sha256",
    }:
        raise RuntimeError("v7 sync closure manifest schema is invalid")
    if (
        payload["schema"] != _SCHEMA
        or not isinstance(payload["files"], list)
        or not isinstance(payload["legacy_dirty_exact_sha256"], dict)
    ):
        raise RuntimeError("v7 sync closure manifest contract differs")
    files = tuple(payload["files"])
    if not files or any(not isinstance(item, str) or not item for item in files):
        raise RuntimeError("v7 sync closure file list is invalid")
    if tuple(sorted(files)) != files or len(set(files)) != len(files):
        raise RuntimeError("v7 sync closure file list must be sorted and unique")
    if CLOSURE_RELATIVE_PATH not in files:
        raise RuntimeError("v7 sync closure must include itself")
    if any(entrypoint not in files for entrypoint in _STATIC_REQUIRED_FILES):
        raise RuntimeError("v7 sync closure is missing an entrypoint")
    legacy = payload["legacy_dirty_exact_sha256"]
    if set(legacy) != {"src/worldarena_baseline/skeleton.py"}:
        raise RuntimeError("v7 sync closure legacy dirty allowlist differs")
    if any(item not in files for item in legacy):
        raise RuntimeError("v7 sync closure legacy dirty dependency is not a closure file")
    for relative, digest in legacy.items():
        if not isinstance(digest, str) or len(digest) != 64 or digest.lower() != digest:
            raise RuntimeError("v7 sync closure legacy dirty dependency SHA is invalid")
    return files, dict(legacy)


def _module_relative(module: str) -> str:
    if not module.startswith("worldarena_baseline"):
        raise RuntimeError(f"not a local v7 module: {module}")
    suffix = module.split(".")[1:]
    if not suffix:
        return "src/worldarena_baseline/__init__.py"
    return "src/worldarena_baseline/" + "/".join(suffix) + ".py"


def _current_module(relative: str) -> str:
    prefix = "src/worldarena_baseline/"
    if not relative.startswith(prefix) or not relative.endswith(".py"):
        raise RuntimeError(f"cannot resolve local module identity: {relative}")
    stem = relative[len("src/") : -3].replace("/", ".")
    if stem.endswith(".__init__"):
        return stem[: -len(".__init__")]
    return stem


def _resolve_relative_import(*, current_module: str, level: int, module: str | None) -> str:
    package = current_module.split(".")[:-1]
    if level > len(package):
        raise RuntimeError(f"v7 local import escapes package: {current_module}")
    base = package[: len(package) - level + 1]
    if module:
        base.extend(module.split("."))
    return ".".join(base)


def _local_imports(path: Path, *, relative: str) -> set[str]:
    """Resolve static local imports and reject unresolved/dynamic variants."""

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"v7 sync entrypoint is unreadable: {path}") from exc
    imports: set[str] = set()
    current_module = _current_module(relative) if relative.startswith("src/") else ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else ""
            attr = node.func.attr if isinstance(node.func, ast.Attribute) else ""
            if name == "__import__" or attr == "import_module":
                first = node.args[0] if node.args else None
                if isinstance(first, ast.Constant) and isinstance(first.value, str) and first.value.startswith("worldarena_baseline"):
                    raise RuntimeError(f"v7 sync closure rejects dynamic local import: {path}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("worldarena_baseline"):
                    imports.add(_module_relative(alias.name))
            continue
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            if not current_module:
                raise RuntimeError(f"v7 sync entrypoint has ambiguous relative import: {path}")
            module = _resolve_relative_import(
                current_module=current_module, level=node.level, module=node.module
            )
            if module.startswith("worldarena_baseline"):
                imports.add(_module_relative(module))
            continue
        if node.module == "worldarena_baseline":
            for alias in node.names:
                imports.add(_module_relative(f"worldarena_baseline.{alias.name}"))
        elif node.module and node.module.startswith("worldarena_baseline."):
            imports.add(_module_relative(node.module))
    return imports


def _transitive_local_imports(repo_root: Path) -> set[str]:
    pending = list(_PYTHON_ENTRYPOINTS)
    resolved: set[str] = set()
    while pending:
        relative = pending.pop()
        if relative in resolved:
            continue
        path = repo_root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v7 sync closure source is missing or non-regular: {relative}")
        resolved.add(relative)
        for dependency in _local_imports(path, relative=relative):
            dependency_path = repo_root / dependency
            if not dependency_path.is_file() or dependency_path.is_symlink():
                raise RuntimeError(f"v7 sync closure local import is unresolved: {dependency}")
            pending.append(dependency)
    return resolved


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
    files, legacy_dirty_exact_sha256 = _load_file_list(root)
    discovered = _transitive_local_imports(root)
    expected_files = tuple(sorted(set(_STATIC_REQUIRED_FILES) | discovered))
    if files != expected_files:
        missing_imports = sorted(set(expected_files) - set(files))
        extra_imports = sorted(set(files) - set(expected_files))
        raise RuntimeError(
            "v7 sync closure does not match transitive local imports: "
            + ", ".join([*(f"missing={item}" for item in missing_imports), *(f"extra={item}" for item in extra_imports)])
        )
    for relative in files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v7 sync closure source is missing or non-regular: {relative}")
        _run_git(root, "ls-files", "--error-unmatch", "--", relative)
        _run_git(root, "diff", "--cached", "--quiet", "--", relative)
        if relative in legacy_dirty_exact_sha256:
            if _sha256(path) != legacy_dirty_exact_sha256[relative]:
                raise RuntimeError(
                    f"v7 sync closure legacy dirty dependency hash mismatch: {relative}"
                )
        else:
            _run_git(root, "diff", "--quiet", "--", relative)
    entries = [{"path": relative, "sha256": _sha256(root / relative)} for relative in files]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "contract": _SCHEMA,
        "files": entries,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }
