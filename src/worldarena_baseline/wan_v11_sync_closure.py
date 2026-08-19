from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


V11_ROOTS = (
    "scripts/run_wan_v11_gpu6.sh",
    "scripts/train_wan_v11_bimanual.py",
    "scripts/validate_wan_v11_sync_closure.py",
    "src/worldarena_baseline/wan_v11_audit.py",
    "src/worldarena_baseline/wan_v11_controller.py",
    "src/worldarena_baseline/wan_v11_model.py",
    "src/worldarena_baseline/wan_v11_objective.py",
    "src/worldarena_baseline/wan_v11_sparse.py",
    "src/worldarena_baseline/wan_v11_state.py",
    "src/worldarena_baseline/wan_v11_sync_closure.py",
    "src/worldarena_baseline/wan_v11_training.py",
)

LEGACY_BYTE_PINS = {
    "src/worldarena_baseline/skeleton.py": (
        "93cb2aa340a725e7b0a7acd37450334bf46c6483c7baf3646bd14ca3e09a8143"
    ),
}


def _module_file(root: Path, module: str) -> str | None:
    if module == "scripts" or module.startswith("scripts."):
        relative = Path(*module.split(".")).with_suffix(".py")
    elif module == "worldarena_baseline" or module.startswith("worldarena_baseline."):
        relative = Path("src", *module.split(".")).with_suffix(".py")
    else:
        return None
    path = root / relative
    return relative.as_posix() if path.is_file() else None


def _package(relative: str) -> tuple[str, ...]:
    parts = Path(relative).with_suffix("").parts
    return tuple(parts[1:-1] if parts[0] == "src" else parts[:-1])


def _imports(root: Path, relative: str) -> set[str]:
    if not relative.endswith(".py"):
        return set()
    tree = ast.parse((root / relative).read_text(encoding="utf-8"), filename=relative)
    package = _package(relative)
    result: set[str] = set()
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                trim = node.level - 1
                if trim > len(package):
                    raise RuntimeError(f"v11 relative import escapes package: {relative}")
                prefix = package[: len(package) - trim]
                base = ".".join((*prefix, *(node.module.split(".") if node.module else ())))
                modules.append(base)
                if node.module is None:
                    modules.extend(f"{base}.{alias.name}" for alias in node.names)
            elif node.module:
                modules.append(node.module)
                if node.module in {"scripts", "worldarena_baseline"}:
                    modules.extend(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            function = node.func
            dynamic = (
                isinstance(function, ast.Name) and function.id == "__import__"
            ) or (
                isinstance(function, ast.Attribute) and function.attr == "import_module"
            )
            if dynamic and node.args:
                argument = node.args[0]
                if isinstance(argument, ast.Constant) and isinstance(argument.value, str):
                    if argument.value.startswith(("scripts", "worldarena_baseline")):
                        modules.append(argument.value)
                elif any(
                    token in ast.unparse(node)
                    for token in ("scripts", "worldarena_baseline")
                ):
                    raise RuntimeError(f"v11 dynamic local import is unresolved: {relative}")
        for module in modules:
            candidate = _module_file(root, module)
            if candidate is not None:
                result.add(candidate)
    return result


def discover_v11_source_files(source_root: Path | str) -> tuple[str, ...]:
    root = Path(source_root).resolve(strict=True)
    pending = list(V11_ROOTS)
    found = set(V11_ROOTS)
    while pending:
        relative = pending.pop()
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise RuntimeError(f"v11 source closure member unavailable: {relative}")
        for dependency in _imports(root, relative):
            if dependency not in found:
                found.add(dependency)
                pending.append(dependency)
    return tuple(sorted(found))


def _git_clean(root: Path, relative: str) -> None:
    checks = [
        ("ls-files", "--error-unmatch", "--", relative),
        ("diff", "--cached", "--quiet", "--", relative),
    ]
    if relative not in LEGACY_BYTE_PINS:
        checks.append(("diff", "--quiet", "--", relative))
    for arguments in checks:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments], capture_output=True, text=True
        )
        if completed.returncode:
            raise RuntimeError(f"v11 source is untracked or dirty: {relative}")
    expected = LEGACY_BYTE_PINS.get(relative)
    if expected is not None and hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"v11 legacy byte pin differs: {relative}")


def build_v11_source_receipt(
    source_root: Path | str,
    *,
    require_git_clean: bool,
) -> dict[str, object]:
    root = Path(source_root).resolve(strict=True)
    entries = []
    for relative in discover_v11_source_files(root):
        path = root / relative
        if require_git_clean:
            _git_clean(root, relative)
        entries.append({"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "contract": "wan-v11-runtime-source-closure/1",
        "files": entries,
        "closure_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def validate_v11_source_receipt(
    source_root: Path | str,
    expected: dict[str, object],
) -> dict[str, object]:
    actual = build_v11_source_receipt(source_root, require_git_clean=False)
    if actual != expected:
        raise RuntimeError("v11 remote source closure differs from reviewed receipt")
    return actual
