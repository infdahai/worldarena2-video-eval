"""Fail-closed contracts for the v14 FlowWAM WorldArena pivot.

This module deliberately contains no model download or CUDA side effects.  It
captures the released checkpoint identity, deterministic audit selection,
FlowWAM's reversible 8-bit flow codec, and the world-model sampling invariant:
the desired flow latent is clean and immutable while only RGB is denoised.
"""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

CONTRACT = "v14-flowwam-wa2/1"
OFFICIAL_SOURCE_COMMIT = "68abaa2b4c609febcc7b230cf06407880e85f5b8"
OFFICIAL_MODEL_REVISION = "1e68f76cecfb2caa973abfb24fca92cbc5312a6e"
FLOWWAM_CHECKPOINT_SIZE = 10_137_267_208
FLOWWAM_CHECKPOINT_SHA256 = (
    "e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4"
)
FLOW_MAX_MAGNITUDE = 25.0
# Released RoboTwin robot-only HDF5 is encoded at this native camera size.
# Both sides of the renderer A/B audit must run RAFT at this same resolution;
# the resulting vector fields are resized to the 640x480 model contract later.
ROBOT_ONLY_NATIVE_SIZE = (320, 240)


@dataclass(frozen=True)
class FlowAuditThresholds:
    # Commanded joint action and released executed robot observation are not
    # pixel-identical at their motion boundary.  The frozen task-disjoint
    # audit20 established 0.84 as a strict per-row overlap floor while all
    # direction, EPE, and renderer-level arm-swap gates remain independent.
    mask_iou: float = 0.84
    direction_cosine: float = 0.95
    median_epe_px: float = 2.0
    arm_identity_accuracy: float = 1.0
    frames: int = 81
    # The official 8-bit HSV codec floors hue to uint8.  At magnitude 25 its
    # exhaustive angular upper bound is 0.8806 px, so 0.90 is the strict
    # representable bound rather than the impossible historical 0.50 gate.
    codec_roundtrip_max_epe_px: float = 0.90


@dataclass(frozen=True)
class FlowAuditResult:
    passed: bool
    failures: tuple[str, ...]
    rows: int


def validate_checkpoint_bytes(payload: bytes) -> str:
    """Validate bytes against the exact public WorldArena checkpoint pin."""
    if len(payload) != FLOWWAM_CHECKPOINT_SIZE:
        raise ValueError(
            f"FlowWAM checkpoint size mismatch: {len(payload)} != "
            f"{FLOWWAM_CHECKPOINT_SIZE}"
        )
    digest = hashlib.sha256(payload).hexdigest()
    if digest != FLOWWAM_CHECKPOINT_SHA256:
        raise ValueError("FlowWAM checkpoint sha256 mismatch")
    return digest


def validate_checkpoint_file(path: str | Path, *, chunk_size: int = 8 << 20) -> str:
    checkpoint = Path(path)
    if checkpoint.is_symlink() or not checkpoint.is_file():
        raise ValueError("FlowWAM checkpoint must be a regular non-symlink file")
    if checkpoint.stat().st_size != FLOWWAM_CHECKPOINT_SIZE:
        raise ValueError("FlowWAM checkpoint size mismatch")
    digest = hashlib.sha256()
    with checkpoint.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    if value != FLOWWAM_CHECKPOINT_SHA256:
        raise ValueError("FlowWAM checkpoint sha256 mismatch")
    return value


def resize_flow(flow: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize an HxWx2 field to ``(width, height)`` and scale its vectors."""
    value = np.asarray(flow, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 2 or not np.isfinite(value).all():
        raise ValueError("flow must be finite HxWx2")
    source_h, source_w = value.shape[:2]
    target_w, target_h = map(int, target_size)
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("flow dimensions must be positive")
    resized = cv2.resize(value, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    resized[..., 0] *= target_w / source_w
    resized[..., 1] *= target_h / source_h
    return resized


def encode_flow_rgb(
    flow: np.ndarray, *, max_magnitude: float = FLOW_MAX_MAGNITUDE
) -> np.ndarray:
    """Match the released FlowCodec 8-bit HSV-to-RGB encoding."""
    value = np.asarray(flow, dtype=np.float32)
    if (
        value.ndim != 3
        or value.shape[-1] != 2
        or not np.isfinite(value).all()
        or not np.isfinite(max_magnitude)
        or max_magnitude <= 0
    ):
        raise ValueError("invalid flow or max_magnitude")
    dx, dy = value[..., 0], value[..., 1]
    magnitude = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    hue = np.clip((angle + np.pi) / (2 * np.pi), 0, 1)
    saturation = np.clip(magnitude / max_magnitude, 0, 1)
    hsv = np.stack(
        [
            (hue * 179).astype(np.uint8),
            (saturation * 255).astype(np.uint8),
            np.full_like(hue, 255, dtype=np.uint8),
        ],
        axis=-1,
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def decode_flow_rgb(rgb: np.ndarray, max_magnitude: float) -> np.ndarray:
    """Invert :func:`encode_flow_rgb` with the released codec convention."""
    value = np.asarray(rgb)
    if value.ndim != 3 or value.shape[-1] != 3 or value.dtype != np.uint8:
        raise ValueError("flow RGB must be uint8 HxWx3")
    if not np.isfinite(max_magnitude) or max_magnitude <= 0:
        raise ValueError("max_magnitude must be positive and finite")
    hsv = cv2.cvtColor(value, cv2.COLOR_RGB2HSV)
    angle = hsv[..., 0].astype(np.float64) / 179.0 * (2 * np.pi) - np.pi
    magnitude = hsv[..., 1].astype(np.float64) / 255.0 * max_magnitude
    return np.stack(
        [magnitude * np.cos(angle), magnitude * np.sin(angle)], axis=-1
    ).astype(np.float32)


def build_world_model_token_timesteps(
    *,
    rgb_timestep: np.ndarray,
    rgb_temporal: int,
    rgb_spatial: int,
    flow_temporal: int,
    flow_spatial: int,
) -> np.ndarray:
    """Build [RGB, flow] per-token timesteps for fixed-flow world modeling.

    RGB frame zero is the clean I2V prefix; remaining RGB tokens use the
    current denoising timestep.  Every flow token uses timestep zero because
    the paper requires the desired flow latent to remain clean and fixed.
    """
    timestep = np.asarray(rgb_timestep, dtype=np.float32)
    if timestep.ndim != 1 or not np.isfinite(timestep).all():
        raise ValueError("rgb_timestep must be a finite batch vector")
    sizes = (rgb_temporal, rgb_spatial, flow_temporal, flow_spatial)
    if any(int(size) <= 0 for size in sizes):
        raise ValueError("token dimensions must be positive")
    batch = timestep.shape[0]
    rgb = np.broadcast_to(
        timestep[:, None, None], (batch, rgb_temporal, rgb_spatial)
    ).copy()
    rgb[:, 0] = 0.0
    flow = np.zeros((batch, flow_temporal, flow_spatial), dtype=np.float32)
    return np.concatenate([rgb.reshape(batch, -1), flow.reshape(batch, -1)], axis=1)


def advance_world_model_latents(
    *,
    rgb_latents,
    flow_latents,
    rgb_velocity,
    scheduler_step: Callable[[object, object], object],
    rgb_prefix,
):
    """Advance RGB exactly once while returning the identical flow object."""
    next_rgb = scheduler_step(rgb_velocity, rgb_latents)
    next_rgb[:, :, :1] = rgb_prefix
    return next_rgb, flow_latents


def _identity(row: Mapping[str, object]) -> tuple[str, str]:
    sample = row.get("sample")
    task = row.get("task")
    if (
        not isinstance(sample, str)
        or not sample
        or not isinstance(task, str)
        or not task
    ):
        raise ValueError("each FlowWAM row needs non-empty sample and task")
    if not isinstance(row.get("hdf5"), str) or not isinstance(
        row.get("robot_only_hdf5"), str
    ):
        raise ValueError("each FlowWAM row needs standard and robot-only HDF5")
    return sample, task


def select_flow_audit20(
    rows: Sequence[Mapping[str, object]],
    *,
    excluded_tasks: set[str],
    seed: int,
) -> tuple[dict[str, object], ...]:
    """Select a deterministic task-balanced, task-disjoint audit set."""
    canonical: list[dict[str, object]] = []
    seen: set[str] = set()
    for source in rows:
        row = dict(source)
        sample, task = _identity(row)
        if sample in seen:
            raise ValueError("duplicate FlowWAM sample identity")
        seen.add(sample)
        if task not in excluded_tasks:
            canonical.append(row)
    by_task: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in canonical:
        sample, task = _identity(row)
        rank = hashlib.sha256(f"v14-flow-audit:{seed}:{sample}".encode()).hexdigest()
        row["_rank"] = rank
        by_task[task].append(row)
    for values in by_task.values():
        values.sort(key=lambda item: (str(item["_rank"]), str(item["sample"])))
    if len(by_task) < 5 or sum(map(len, by_task.values())) < 20:
        raise ValueError("audit20 requires at least 20 rows across five tasks")
    selected: list[dict[str, object]] = []
    offsets = Counter()
    tasks = sorted(by_task)
    while len(selected) < 20:
        progressed = False
        for task in tasks:
            offset = offsets[task]
            if offset < len(by_task[task]):
                row = dict(by_task[task][offset])
                row.pop("_rank")
                selected.append(row)
                offsets[task] += 1
                progressed = True
                if len(selected) == 20:
                    break
        if not progressed:
            raise ValueError("could not construct FlowWAM audit20")
    return tuple(sorted(selected, key=lambda item: str(item["sample"])))


def select_flow_rgb4(
    rows: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Freeze two single-arm and two bimanual rows from precomputed energies."""
    single: list[dict[str, object]] = []
    bimanual: list[dict[str, object]] = []
    for source in sorted(rows, key=lambda row: str(row.get("sample", ""))):
        row = dict(source)
        _identity(row)
        left = float(row.get("left_motion_energy", -1))
        right = float(row.get("right_motion_energy", -1))
        if min(left, right) < 0 or not np.isfinite([left, right]).all():
            raise ValueError("RGB4 rows require finite nonnegative arm energies")
        high, low = max(left, right), min(left, right)
        if high <= 0:
            continue
        if low / high >= 0.35:
            bimanual.append(row)
        elif low / high <= 0.10:
            single.append(row)

    def take_distinct(candidates: list[dict[str, object]], count: int):
        selected = []
        tasks = set()
        for row in candidates:
            if row["task"] not in tasks:
                selected.append(row)
                tasks.add(row["task"])
            if len(selected) == count:
                return selected
        raise ValueError("RGB4 requires task-distinct single and bimanual samples")

    return tuple(take_distinct(single, 2) + take_distinct(bimanual, 2))


def evaluate_flow_audit(
    rows: Sequence[Mapping[str, object]], thresholds: FlowAuditThresholds
) -> FlowAuditResult:
    """Apply every Action-to-Flow threshold per episode, never by mean only."""
    if len(rows) != 20:
        return FlowAuditResult(False, ("audit_rows",), len(rows))
    failures: list[str] = []
    for index, row in enumerate(rows):
        checks = {
            "mask_iou": float(row.get("mask_iou", -np.inf)) >= thresholds.mask_iou,
            "direction_cosine": float(row.get("direction_cosine", -np.inf))
            >= thresholds.direction_cosine,
            "median_epe_px": float(row.get("median_epe_px", np.inf))
            <= thresholds.median_epe_px,
            "arm_identity_accuracy": float(row.get("arm_identity_accuracy", -np.inf))
            >= thresholds.arm_identity_accuracy,
            "frames": row.get("frames") == thresholds.frames,
            "frame0_exact_white": row.get("frame0_exact_white") is True,
            "codec_roundtrip_max_epe_px": float(
                row.get("codec_roundtrip_max_epe_px", np.inf)
            )
            <= thresholds.codec_roundtrip_max_epe_px,
        }
        failures.extend(
            f"row{index}:{name}" for name, passed in checks.items() if not passed
        )
    return FlowAuditResult(not failures, tuple(failures), len(rows))
