"""Byte-bound source closure for the Wan v9 production run."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess

from .wan_v7_sync_closure import validate_sync_closure as validate_v7_sync_closure


V9_FILES = (
    "scripts/build_wan_v9_replay.py",
    "scripts/cache_wan_v9_transitions.py",
    "scripts/run_wan_v9_phase_locked_gpu6.sh",
    "scripts/train_wan_v9_phase_locked.py",
    "scripts/validate_wan_v9_sync_closure.py",
    "src/worldarena_baseline/wan_v9_attention.py",
    "src/worldarena_baseline/wan_v9_audit.py",
    "src/worldarena_baseline/wan_v9_model.py",
    "src/worldarena_baseline/wan_v9_objective.py",
    "src/worldarena_baseline/wan_v9_sync_closure.py",
    "src/worldarena_baseline/wan_v9_tokens.py",
    "src/worldarena_baseline/wan_v9_training.py",
    "src/worldarena_baseline/wan_v9_transition.py",
)


def _module_file(root: Path, module: str) -> str | None:
    if module == "scripts" or module.startswith("scripts."):
        relative = Path(*module.split(".")).with_suffix(".py")
    elif module == "worldarena_baseline" or module.startswith("worldarena_baseline."):
        relative = Path("src", *module.split(".")).with_suffix(".py")
    else:
        return None
    path = root / relative
    return relative.as_posix() if path.is_file() else None


def _package_for(relative: str) -> tuple[str, ...]:
    path = Path(relative)
    if path.parts[0] == "src":
        return tuple(path.with_suffix("").parts[1:-1])
    if path.parts[0] == "scripts":
        return tuple(path.with_suffix("").parts[:-1])
    return ()


def _static_local_imports(root: Path, relative: str) -> set[str]:
    path = root / relative
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
    except (OSError, SyntaxError) as exc:
        raise RuntimeError(f"cannot parse v9 source closure member: {relative}") from exc
    result: set[str] = set()
    package = _package_for(relative)
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                trim = node.level - 1
                if trim > len(package):
                    raise RuntimeError(f"v9 relative import escapes package: {relative}")
                prefix = package[: len(package) - trim]
                base = ".".join((*prefix, *(node.module.split(".") if node.module else ())))
                modules.extend(
                    f"{base}.{alias.name}" if base and node.module is None else base
                    for alias in node.names
                )
            elif node.module:
                modules.append(node.module)
                if node.module in {"scripts", "worldarena_baseline"}:
                    modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        for module in modules:
            candidate = _module_file(root, module)
            if candidate is not None:
                result.add(candidate)
    return result


def discover_v9_source_files(source_root: Path | str, v7_files: set[str]) -> tuple[str, ...]:
    root = Path(source_root).resolve(strict=True)
    pending = list(V9_FILES)
    discovered = set(V9_FILES)
    while pending:
        relative = pending.pop()
        if not relative.endswith(".py"):
            continue
        for dependency in _static_local_imports(root, relative):
            if dependency in v7_files or dependency in discovered:
                continue
            discovered.add(dependency)
            pending.append(dependency)
    return tuple(sorted(discovered))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_clean(root: Path, relative: str) -> None:
    for arguments in (
        ("ls-files", "--error-unmatch", "--", relative),
        ("diff", "--cached", "--quiet", "--", relative),
        ("diff", "--quiet", "--", relative),
    ):
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True, check=False
        )
        if completed.returncode:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(f"v9 source is untracked or dirty: {relative}: {detail}")


def build_v9_source_receipt(source_root: Path | str) -> dict[str, object]:
    root = Path(source_root).resolve(strict=True)
    v7 = validate_v7_sync_closure(root)
    entries_by_path = {str(entry["path"]): dict(entry) for entry in v7["files"]}
    v9_files = discover_v9_source_files(root, set(entries_by_path))
    for relative in v9_files:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v9 source is missing or non-regular: {relative}")
        _git_clean(root, relative)
        entries_by_path[relative] = {"path": relative, "sha256": _sha(path)}
    entries = [entries_by_path[name] for name in sorted(entries_by_path)]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "contract": "wan-v9-runtime-source-closure/1",
        "files": entries,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
        "v7_base_closure_sha256": v7["closure_sha256"],
        "v9_files": list(v9_files),
    }
