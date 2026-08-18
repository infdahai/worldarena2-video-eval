"""Explicit, byte-bound runtime source closure for the bounded Wan v8 run."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json


V8_CLOSURE_ANCHORS = (
    "src/worldarena_baseline/wan_v8_data.py",
    "src/worldarena_baseline/wan_v8_sync_closure.py",
    "scripts/prepare_wan_v8_data.py",
    "scripts/cache_wan_v8_counterfactuals.py",
    "scripts/train_wan_v8_direct_action_band.py",
    "scripts/run_wan_v8_direct_action_band.sh",
    "src/worldarena_baseline/wan_v8_counterfactual.py",
    "src/worldarena_baseline/wan_v8_attention.py",
    "src/worldarena_baseline/wan_v8_model.py",
    "src/worldarena_baseline/wan_v8_objective.py",
    "src/worldarena_baseline/wan_v8_training.py",
    "src/worldarena_baseline/wan_v8_audit.py",
    "src/worldarena_baseline/wan_v7_model.py",
    "src/worldarena_baseline/wan_se3_attention.py",
    "src/worldarena_baseline/wan_action_adapter.py",
    "src/worldarena_baseline/wan_action_loss.py",
    "src/worldarena_baseline/wan_cached_dataset.py",
    "src/worldarena_baseline/wan_v71_cf.py",
    "src/worldarena_baseline/wan_gripper_probe.py",
    "src/worldarena_baseline/wan_gripper_trajectory_loss.py",
    "src/worldarena_baseline/wan_v6_checkpoint.py",
    "scripts/train_wan_se3_probe_v7_fsdp.py",
)


def validate_v8_source_anchors(source_root: Path | str) -> tuple[Path, ...]:
    """Fail before sync when a committed v8 entrypoint is missing or a symlink."""
    root = Path(source_root)
    resolved: list[Path] = []
    for relative in V8_CLOSURE_ANCHORS:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"v8 closure anchor is not a regular file: {relative}")
        resolved.append(path)
    return tuple(resolved)


def build_v8_source_receipt(source_root: Path | str) -> dict[str, object]:
    root=Path(source_root).resolve(strict=True); files=[]; digest=hashlib.sha256()
    for path in validate_v8_source_anchors(root):
        relative=path.relative_to(root).as_posix(); value=hashlib.sha256(path.read_bytes()).hexdigest()
        files.append({"path":relative,"sha256":value}); digest.update(relative.encode()); digest.update(b"\0"); digest.update(value.encode()); digest.update(b"\0")
    return {"contract":"wan-v8-runtime-source-closure/1","files":files,"closure_sha256":digest.hexdigest()}
