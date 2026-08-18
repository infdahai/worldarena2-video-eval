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


def test_sync_closure_is_sorted_and_covers_direct_cache_and_trainer_imports() -> None:
    from worldarena_baseline.wan_v7_sync_closure import _direct_internal_imports

    files = _closure_files(ROOT)
    assert files == sorted(files)
    direct = set()
    for entrypoint in (
        "scripts/cache_wan_v7_se3_conditions.py",
        "scripts/train_wan_se3_probe_v7_fsdp.py",
    ):
        direct.update(_direct_internal_imports(ROOT / entrypoint))
    assert direct <= set(files)


def test_sync_closure_rejects_missing_or_untracked_required_source_before_sync(tmp_path: Path) -> None:
    from worldarena_baseline.wan_v7_sync_closure import validate_sync_closure

    clone = _committed_closure_clone(tmp_path)
    receipt = validate_sync_closure(clone)
    assert receipt["contract"] == "wan-action-v7-stagea-sync-closure/1"
    assert len(receipt["files"]) == len(_closure_files(ROOT))

    (clone / "src/worldarena_baseline/wan_cached_dataset.py").unlink()
    with pytest.raises(RuntimeError, match="missing or non-regular"):
        validate_sync_closure(clone)

    restored = clone / "src/worldarena_baseline/wan_cached_dataset.py"
    shutil.copyfile(ROOT / "src/worldarena_baseline/wan_cached_dataset.py", restored)
    subprocess.run(["git", "rm", "--cached", "src/worldarena_baseline/wan_cached_dataset.py"], cwd=clone, check=True)
    with pytest.raises(RuntimeError, match="git validation failed"):
        validate_sync_closure(clone)
