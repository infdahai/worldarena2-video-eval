from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path


ARTIFACT_ROOT = Path("/data/di/worldarena2_track1_20260815")


def _beneath_root(
    path: Path | str, *, artifact_root: Path | str, label: str
) -> Path:
    root = Path(artifact_root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must resolve beneath artifact root {root}") from exc
    if resolved == root:
        raise ValueError(f"{label} must be below artifact root {root}")
    return resolved


def require_artifact_output(
    output: Path | str, *, artifact_root: Path | str = ARTIFACT_ROOT
) -> Path:
    return _beneath_root(
        output, artifact_root=artifact_root, label="output"
    )


def require_artifact_input(
    input_path: Path | str,
    *,
    artifact_root: Path | str = ARTIFACT_ROOT,
    kind: str = "file",
) -> Path:
    """Guard an input path before checking or reading its contents."""

    path = _beneath_root(
        input_path, artifact_root=artifact_root, label="input"
    )
    if kind == "file":
        if not path.is_file():
            raise FileNotFoundError(path)
    elif kind == "directory":
        if not path.is_dir():
            raise FileNotFoundError(path)
    else:
        raise ValueError("input kind must be file or directory")
    return path


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def environment_manifest_binding_sha256(payload: Mapping[str, object]) -> str:
    """Hash reproducible environment provenance while excluding collection time."""

    binding = dict(payload)
    binding.pop("created_at", None)
    return canonical_sha256(binding)


def _source_file_hashes(
    source_root: Path, listed_paths: bytes
) -> dict[str, dict[str, object]]:
    files: dict[str, dict[str, object]] = {}
    for raw_relative in listed_paths.split(b"\0"):
        if not raw_relative:
            continue
        relative = raw_relative.decode("utf-8")
        path = (source_root / relative).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as exc:
            raise ValueError("git source path escapes source root") from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return files


CommandRunner = Callable[[list[str], Path | None], bytes]


def collect_environment_manifest_payload(
    *,
    source_root: Path | str,
    expected_source_root: Path | str,
    named_inputs: Mapping[str, Path | str],
    artifact_root: Path | str,
    interpreter: Path | str,
    expected_interpreter: Path | str,
    base_interpreter: Path | str,
    expected_base_interpreter: Path | str,
    python_version: str,
    platform_string: str,
    torch_info: Mapping[str, object],
    run_command: CommandRunner,
    created_at: str,
) -> dict[str, object]:
    """Collect a fully bound manifest with all process calls injectable for tests."""

    source = Path(source_root).resolve()
    if source != Path(expected_source_root).resolve():
        raise ValueError("source root does not match the approved checkout")
    if not (source / ".git").exists():
        raise FileNotFoundError(f"source root is not a git checkout: {source}")
    executable = Path(interpreter).resolve()
    if executable != Path(expected_interpreter).resolve():
        raise ValueError("interpreter does not match the formal interpreter")
    base = Path(base_interpreter).resolve()
    if base != Path(expected_base_interpreter).resolve():
        raise ValueError("base interpreter does not match the approved interpreter")

    input_hashes = {}
    for name, raw_path in sorted(named_inputs.items()):
        if not name:
            raise ValueError("named input cannot have an empty name")
        path = require_artifact_input(
            raw_path, artifact_root=artifact_root, kind="file"
        )
        input_hashes[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    def run(command: list[str], cwd: Path | None = None) -> bytes:
        result = run_command(command, cwd)
        if not isinstance(result, bytes):
            raise TypeError("environment command runner must return bytes")
        return result

    requirements = run(
        [str(executable), "-m", "pip", "freeze"]
    ).decode("utf-8").strip()
    gpu_inventory = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    ).decode("utf-8").strip()
    git_head = run(["git", "rev-parse", "HEAD"], source).decode(
        "utf-8"
    ).strip()
    git_status = run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], source
    ).decode("utf-8").strip()
    git_diff = run(["git", "diff", "--binary", "HEAD"], source)
    listed_paths = run(
        ["git", "ls-files", "-co", "--exclude-standard", "-z"], source
    )
    source_files = _source_file_hashes(source, listed_paths)
    source_tree_sha256 = canonical_sha256(source_files)
    return {
        "schema_version": 3,
        "contract_version": "wan-action-lite-v3-environment/1",
        "created_at": created_at,
        "python": {
            "executable": str(executable),
            "base_executable": str(base),
            "version": python_version,
            "platform": platform_string,
        },
        "torch": dict(torch_info),
        "requirements": {
            "command": "pip freeze",
            "lines": requirements.splitlines(),
        },
        "gpu_inventory": {
            "command": "nvidia-smi",
            "lines": gpu_inventory.splitlines(),
        },
        "git": {
            "root": str(source),
            "head": git_head,
            "status_porcelain": git_status.splitlines(),
            "diff_sha256": hashlib.sha256(git_diff).hexdigest(),
            "diff_bytes": len(git_diff),
            "source_tree_sha256": source_tree_sha256,
            "source_files": source_files,
        },
        "input_hashes": input_hashes,
    }


def atomic_write_environment_manifest(
    output: Path | str,
    payload: Mapping[str, object],
    *,
    artifact_root: Path | str = ARTIFACT_ROOT,
) -> Path:
    path = require_artifact_output(output, artifact_root=artifact_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
    return path
