from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .environment_manifest import (
    canonical_sha256,
    require_artifact_input,
    sha256_file,
)


WAN_BACKBONE_SHARD_SIZES = {
    "diffusion_pytorch_model-00001-of-00003.safetensors": 9_825_014_472,
    "diffusion_pytorch_model-00002-of-00003.safetensors": 9_995_661_736,
    "diffusion_pytorch_model-00003-of-00003.safetensors": 178_558_176,
}
WAN_BACKBONE_PARAMETER_BYTES = 19_999_150_848
WAN_VAE_BYTES = 2_818_839_170

WAN_TI2V_CONFIG = {
    "dim": 3072,
    "in_dim": 48,
    "out_dim": 48,
    "model_type": "ti2v",
    "num_heads": 24,
    "num_layers": 30,
    "text_len": 512,
}


@dataclass(frozen=True)
class WanProbeShape:
    latent: tuple[int, int, int, int, int]
    action_raster: tuple[int, int, int, int, int]
    left_pose: tuple[int, int, int]
    right_pose: tuple[int, int, int]
    condition_support: tuple[int, int, int, int, int]
    loss_weight: tuple[int, int, int, int, int]
    action_present: tuple[int]
    sequence_length: int


def require_wan_backbone_checkpoint(checkpoint_dir: Path | str) -> None:
    """Validate the independently usable Wan TI2V transformer component."""

    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "config.json"
    index_path = checkpoint_dir / "diffusion_pytorch_model.safetensors.index.json"
    if not config_path.is_file() or not index_path.is_file():
        raise FileNotFoundError("Wan backbone config or safetensors index is missing")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    wrong_config = {
        key: (config.get(key), expected)
        for key, expected in WAN_TI2V_CONFIG.items()
        if config.get(key) != expected
    }
    if wrong_config:
        raise ValueError(f"unexpected Wan TI2V config: {wrong_config}")

    index = json.loads(index_path.read_text(encoding="utf-8"))
    expected_names = set(WAN_BACKBONE_SHARD_SIZES)
    indexed_names = set(index.get("weight_map", {}).values())
    if indexed_names != expected_names:
        raise ValueError(
            f"unexpected Wan backbone shard index: {sorted(indexed_names)}"
        )
    if index.get("metadata", {}).get("total_size") != WAN_BACKBONE_PARAMETER_BYTES:
        raise ValueError("unexpected Wan backbone total_size")

    for name, expected_size in WAN_BACKBONE_SHARD_SIZES.items():
        path = checkpoint_dir / name
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_size != expected_size:
            raise ValueError(
                f"shard size mismatch for {name}: "
                f"expected {expected_size}, got {actual_size}"
            )


def require_wan_vae_checkpoint(checkpoint_dir: Path | str) -> None:
    """Validate the Wan2.2 VAE independently from the T5 text component."""

    path = Path(checkpoint_dir) / "Wan2.2_VAE.pth"
    actual_size = path.stat().st_size if path.is_file() else None
    if actual_size != WAN_VAE_BYTES:
        raise ValueError(
            f"VAE size mismatch: expected {WAN_VAE_BYTES}, got {actual_size}"
        )


def build_probe_shape(
    *, batch_size: int, frames: int, height: int, width: int
) -> WanProbeShape:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if (frames, height, width) != (81, 480, 640):
        raise ValueError("v3 probe requires exactly 81 frames at 480x640")
    return WanProbeShape(
        latent=(batch_size, 48, 21, 30, 40),
        action_raster=(batch_size, 10, 81, 60, 80),
        left_pose=(batch_size, 81, 11),
        right_pose=(batch_size, 81, 11),
        condition_support=(batch_size, 2, 21, 15, 20),
        loss_weight=(batch_size, 1, 21, 30, 40),
        action_present=(batch_size,),
        sequence_length=6300,
    )


_PROBE_GRADIENT_FAMILIES = (
    "left_raster",
    "right_raster",
    "left_pose",
    "right_pose",
)
_PROBE_RESIDUAL_POINTS = ("0", "8", "16", "24")
_DEFAULT_WORLD_SIZE = 8
_DEFAULT_MEMORY_LIMIT_BYTES = 22 * 1024**3
_SMOKE_CONTRACT_VERSION = "wan-action-lite-v3-smoke/1"
_PRODUCTION_OPTIMIZER_STEPS = 3
_FORBIDDEN_TRAINING_MODULES = ("wan.modules.t5", "wan.modules.vae2_2")
_POST_STEP_MEMORY_CREEP_LIMIT_BYTES = 512 * 1024**2


def validate_training_hot_path_components(
    live_objects: Sequence[object],
) -> tuple[str, ...]:
    forbidden = tuple(sorted({
        type(obj).__module__
        for obj in live_objects
        if type(obj).__module__ in _FORBIDDEN_TRAINING_MODULES
    }))
    if forbidden:
        raise ValueError(
            f"training hot path instantiated forbidden T5/VAE components: {forbidden}"
        )
    return forbidden


def _file_provenance(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_smoke_input_provenance(
    *,
    checkpoint_dir: Path | str,
    adapter_checkpoint: Path | str | None,
    action_condition: Path | str | None,
    source_files: Mapping[str, Path | str],
    artifact_root: Path | str,
) -> dict[str, object]:
    """Hash every model asset and optional probe input before a formal smoke."""

    model_root = require_artifact_input(
        checkpoint_dir, artifact_root=artifact_root, kind="directory"
    )
    adapter = (
        None
        if adapter_checkpoint is None
        else require_artifact_input(
            adapter_checkpoint, artifact_root=artifact_root, kind="file"
        )
    )
    condition = (
        None
        if action_condition is None
        else require_artifact_input(
            action_condition, artifact_root=artifact_root, kind="file"
        )
    )

    manifest_path = require_artifact_input(
        model_root / ".worldarena_manifest.json",
        artifact_root=artifact_root,
        kind="file",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_assets = manifest.get("assets")
    if not isinstance(raw_assets, list) or not raw_assets:
        raise ValueError("model manifest must contain at least one asset")
    assets: dict[str, dict[str, object]] = {}
    for raw_asset in raw_assets:
        if not isinstance(raw_asset, Mapping):
            raise ValueError("model manifest asset must be an object")
        relative = raw_asset.get("path")
        expected_size = raw_asset.get("size")
        if not isinstance(relative, str) or not relative:
            raise ValueError("model manifest asset path is invalid")
        if not isinstance(expected_size, int) or expected_size < 0:
            raise ValueError("model manifest asset size is invalid")
        if relative in assets:
            raise ValueError(f"duplicate model manifest asset: {relative}")
        asset_path = require_artifact_input(
            model_root / relative,
            artifact_root=artifact_root,
            kind="file",
        )
        try:
            asset_path.relative_to(model_root)
        except ValueError as exc:
            raise ValueError("model manifest asset escapes checkpoint directory") from exc
        if asset_path.stat().st_size != expected_size:
            raise ValueError(f"model asset size mismatch: {relative}")
        assets[relative] = _file_provenance(asset_path)
    payload_bytes = manifest.get("payload_bytes")
    if payload_bytes is not None and payload_bytes != sum(
        asset["bytes"] for asset in assets.values()
    ):
        raise ValueError("model manifest payload_bytes does not match assets")

    source: dict[str, dict[str, object]] = {}
    for label, raw_path in sorted(source_files.items()):
        if not label:
            raise ValueError("source provenance label cannot be empty")
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        source[label] = _file_provenance(path)
    if not source:
        raise ValueError("source provenance cannot be empty")

    return {
        "contract_version": _SMOKE_CONTRACT_VERSION,
        "model": {
            "root": str(model_root),
            "manifest": _file_provenance(manifest_path),
            "repo_id": manifest.get("repo_id"),
            "revision": manifest.get("revision"),
            "assets": assets,
        },
        "adapter": None if adapter is None else _file_provenance(adapter),
        "action_condition": (
            None if condition is None else _file_provenance(condition)
        ),
        "source": source,
    }


def smoke_provenance_binding(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if provenance.get("contract_version") != _SMOKE_CONTRACT_VERSION:
        raise ValueError("unexpected smoke provenance contract version")
    return {
        "input_provenance": dict(provenance),
        "input_provenance_sha256": canonical_sha256(provenance),
    }


def validate_smoke_report_binding(
    report: Mapping[str, object], expected_provenance: Mapping[str, object]
) -> None:
    """Reject stale smoke evidence after any model, input, or source change."""

    if report.get("schema_version") != 3:
        raise ValueError("smoke report schema must be version 3")
    expected_binding = smoke_provenance_binding(expected_provenance)
    if report.get("input_provenance") != expected_binding["input_provenance"]:
        raise ValueError("smoke report input provenance does not match")
    if (
        report.get("input_provenance_sha256")
        != expected_binding["input_provenance_sha256"]
    ):
        raise ValueError("smoke report provenance hash does not match")


def _positive_finite(value: object, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def validate_probe_telemetry(
    records: Sequence[Mapping[str, object]],
    *,
    world_size: int = _DEFAULT_WORLD_SIZE,
    use_pose: bool = True,
    memory_limit_bytes: int = _DEFAULT_MEMORY_LIMIT_BYTES,
    residual_ratio_limit: float = 10.0,
) -> dict[str, object]:
    """Validate gathered fixed-topology smoke evidence, failing closed."""

    if world_size not in (7, 8):
        raise ValueError("formal v3 probe requires exactly seven or eight ranks")
    if memory_limit_bytes <= 0 or residual_ratio_limit <= 1:
        raise ValueError("probe limits must be positive")
    if len(records) != world_size:
        raise ValueError(f"expected {world_size} rank telemetry records")
    ranks = [record.get("rank") for record in records]
    if sorted(ranks) != list(range(world_size)):
        raise ValueError("probe telemetry must contain each rank exactly once")

    max_allocated = 0
    max_reserved = 0
    max_step_time = 0.0
    for record in records:
        rank = int(record["rank"])
        if record.get("optimizer_steps") != _PRODUCTION_OPTIMIZER_STEPS:
            raise ValueError(f"rank {rank} must complete exactly three optimizer steps")
        raw_step_times = record.get("step_times_seconds")
        if (
            not isinstance(raw_step_times, Sequence)
            or isinstance(raw_step_times, (str, bytes))
            or len(raw_step_times) != _PRODUCTION_OPTIMIZER_STEPS
        ):
            raise ValueError(f"rank {rank} step-time telemetry is incomplete")
        step_times = [
            _positive_finite(value, label=f"rank {rank} step time")
            for value in raw_step_times
        ]
        max_step_time = max(max_step_time, *step_times)
        forbidden = record.get("forbidden_training_modules")
        if forbidden != []:
            raise ValueError(f"rank {rank} loaded forbidden T5/VAE modules")
        per_step_memory: dict[str, list[int]] = {}
        for field in (
            "step_peak_allocated_bytes",
            "step_peak_reserved_bytes",
            "post_step_allocated_bytes",
        ):
            raw_values = record.get(field)
            if (
                not isinstance(raw_values, Sequence)
                or isinstance(raw_values, (str, bytes))
                or len(raw_values) != _PRODUCTION_OPTIMIZER_STEPS
            ):
                raise ValueError(f"rank {rank} per-step memory telemetry is incomplete")
            try:
                values = [int(value) for value in raw_values]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"rank {rank} per-step memory telemetry is invalid"
                ) from exc
            if any(value < 0 or value >= memory_limit_bytes for value in values):
                raise ValueError(f"rank {rank} reached the 22 GiB memory limit")
            per_step_memory[field] = values
        if any(
            reserved < allocated
            for allocated, reserved in zip(
                per_step_memory["step_peak_allocated_bytes"],
                per_step_memory["step_peak_reserved_bytes"],
            )
        ):
            raise ValueError(f"rank {rank} per-step reserved memory is below allocated")
        post_step = per_step_memory["post_step_allocated_bytes"]
        if post_step[-1] - post_step[0] >= _POST_STEP_MEMORY_CREEP_LIMIT_BYTES:
            raise ValueError(f"rank {rank} shows abnormal post-step memory creep")
        try:
            allocated = int(record["peak_allocated_bytes"])
            reserved = int(record["peak_reserved_bytes"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"rank {rank} lacks valid memory telemetry") from exc
        if (
            allocated < 0
            or reserved < 0
            or allocated >= memory_limit_bytes
            or reserved >= memory_limit_bytes
        ):
            raise ValueError(f"rank {rank} reached the 22 GiB memory limit")
        if reserved < allocated:
            raise ValueError(f"rank {rank} reserved memory is below allocated memory")
        if allocated != max(per_step_memory["step_peak_allocated_bytes"]):
            raise ValueError(f"rank {rank} allocated peak does not match per-step data")
        if reserved != max(per_step_memory["step_peak_reserved_bytes"]):
            raise ValueError(f"rank {rank} reserved peak does not match per-step data")
        max_allocated = max(max_allocated, allocated)
        max_reserved = max(max_reserved, reserved)

        gradients = record.get("gradient_norms")
        if not isinstance(gradients, Mapping) or set(gradients) != set(
            _PROBE_GRADIENT_FAMILIES
        ):
            raise ValueError(f"rank {rank} gradient telemetry is incomplete")
        for family in ("left_raster", "right_raster"):
            _positive_finite(
                gradients[family], label=f"rank {rank} {family} gradient"
            )
        for family in ("left_pose", "right_pose"):
            if use_pose:
                _positive_finite(
                    gradients[family], label=f"rank {rank} {family} gradient"
                )
                continue
            try:
                pose_gradient = float(gradients[family])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"rank {rank} raster-only pose gradient must be numeric"
                ) from exc
            if not math.isfinite(pose_gradient) or pose_gradient != 0.0:
                raise ValueError(
                    f"rank {rank} raster-only probe produced a pose gradient"
                )

        residuals = record.get("residual_norms")
        if not isinstance(residuals, Mapping) or set(residuals) != set(
            _PROBE_RESIDUAL_POINTS
        ):
            raise ValueError(f"rank {rank} residual telemetry is incomplete")
        values = [
            _positive_finite(
                residuals[point], label=f"rank {rank} block {point} residual"
            )
            for point in _PROBE_RESIDUAL_POINTS
        ]
        if max(values) / min(values) >= residual_ratio_limit:
            raise ValueError(f"rank {rank} residual norms differ by an order of magnitude")

    return {
        "world_size": world_size,
        "memory_limit_bytes": memory_limit_bytes,
        "optimizer_steps": _PRODUCTION_OPTIMIZER_STEPS,
        "max_step_time_seconds": max_step_time,
        "max_peak_allocated_bytes": max_allocated,
        "max_peak_reserved_bytes": max_reserved,
        "ranks": [
            dict(record)
            for record in sorted(records, key=lambda row: int(row["rank"]))
        ],
    }
