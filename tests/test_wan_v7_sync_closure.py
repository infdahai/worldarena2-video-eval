from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _closure_files(root: Path) -> list[str]:
    payload = json.loads((root / "source_inputs/wan-v7-stagea-sync-closure.json").read_text())
    return list(payload["files"])


def _committed_closure_clone(tmp_path: Path) -> Path:
    clone = tmp_path / "baseline"
    clone.mkdir()
    for relative in _closure_files(ROOT):
        source = ROOT / relative
        target = clone / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    subprocess.run(["git", "init"], cwd=clone, check=True, stdout=subprocess.PIPE)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=clone, check=True)
    subprocess.run(["git", "config", "user.name", "v7 test"], cwd=clone, check=True)
    subprocess.run(["git", "add", "."], cwd=clone, check=True)
    subprocess.run(["git", "commit", "-m", "closure"], cwd=clone, check=True, stdout=subprocess.PIPE)
    return clone


def test_sync_closure_is_sorted_and_covers_transitive_cache_and_trainer_imports() -> None:
    from worldarena_baseline.wan_v7_sync_closure import _STATIC_REQUIRED_FILES, _transitive_local_imports

    files = _closure_files(ROOT)
    assert files == sorted(files)
    expected = set(_STATIC_REQUIRED_FILES) | _transitive_local_imports(ROOT)
    assert set(files) == expected
    # These were absent from the first direct-only closure but are imported at
    # runtime by cached-dataset/probe/replay dependencies.
    assert {
        "scripts/build_wan_v7_replay.py",
        "src/worldarena_baseline/robotwin_action_cache.py",
        "src/worldarena_baseline/action_audit.py",
        "src/worldarena_baseline/wan_v6_training.py",
        "src/worldarena_baseline/environment_manifest.py",
    } <= set(files)


def test_sync_closure_rejects_missing_or_untracked_required_source_before_sync(tmp_path: Path) -> None:
    from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure

    clone = _committed_closure_clone(tmp_path)
    receipt = validate_sync_closure(clone)
    assert receipt["contract"] == "wan-action-v7-stagea-sync-closure/1"
    assert len(receipt["files"]) == len(_closure_files(ROOT))

    (clone / "src/worldarena_baseline/wan_cached_dataset.py").unlink()
    with pytest.raises(RuntimeError, match="unresolved|missing or non-regular"):
        validate_sync_closure(clone)

    restored = clone / "src/worldarena_baseline/wan_cached_dataset.py"
    shutil.copyfile(ROOT / "src/worldarena_baseline/wan_cached_dataset.py", restored)
    subprocess.run(["git", "rm", "--cached", "src/worldarena_baseline/wan_cached_dataset.py"], cwd=clone, check=True)
    with pytest.raises(RuntimeError, match="git validation failed"):
        validate_sync_closure(clone)


def test_sync_closure_rejects_dynamic_or_unresolved_local_imports(tmp_path: Path) -> None:
    from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure

    clone = _committed_closure_clone(tmp_path)
    entrypoint = clone / "scripts/cache_wan_v7_se3_conditions.py"
    entrypoint.write_text(
        entrypoint.read_text(encoding="utf-8")
        + '\n__import__("worldarena_baseline.unknown_runtime_module")\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="dynamic local import"):
        validate_sync_closure(clone)


def test_sync_closure_allows_only_the_source_pinned_legacy_skeleton_bytes(tmp_path: Path) -> None:
    from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure

    clone = _committed_closure_clone(tmp_path)
    skeleton = clone / "src/worldarena_baseline/skeleton.py"
    pinned_bytes = skeleton.read_bytes()
    # Make the index represent an older legacy dependency, then restore the
    # exact source-pinned working-tree bytes as the authorized dirty exception.
    skeleton.write_text("# older tracked legacy dependency\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/worldarena_baseline/skeleton.py"], cwd=clone, check=True)
    subprocess.run(["git", "commit", "-m", "older skeleton"], cwd=clone, check=True, stdout=subprocess.PIPE)
    skeleton.write_bytes(pinned_bytes)
    receipt = validate_sync_closure(clone)
    skeleton_entry = next(item for item in receipt["files"] if item["path"].endswith("skeleton.py"))
    assert skeleton_entry["sha256"] == "93cb2aa340a725e7b0a7acd37450334bf46c6483c7baf3646bd14ca3e09a8143"

    skeleton.write_bytes(pinned_bytes + b"# tampered\n")
    with pytest.raises(RuntimeError, match="legacy dirty dependency hash mismatch"):
        validate_sync_closure(clone)
