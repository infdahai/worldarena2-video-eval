"""Immutable data, replay, and correct-SE(3) cache contracts for Wan v8."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .wan_balanced_sampling import build_balanced_history
from .wan_se3_condition import validate_se3_cache


V8_AUDIT_SEED = 20260818
V8_TARGET = {
    "single_dominant": 0.45,
    "bimanual_heavy": 0.30,
    "mixed": 0.15,
    "quiet": 0.10,
}
_NEGATIVE_CYCLE = ("reverse", "shift", "swap")


@dataclass(frozen=True)
class V8DataReceipt:
    """The immutable split used by all v8 phases."""

    audit_samples: tuple[str, ...]
    optimizer_samples: tuple[str, ...]
    replay: tuple[dict[str, object], ...]


def _require_samples(rows: Iterable[Mapping[str, object]], *, label: str) -> tuple[str, ...]:
    samples: list[str] = []
    for row in rows:
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample:
            raise ValueError(f"{label} row has invalid sample")
        samples.append(sample)
    if len(samples) != len(set(samples)):
        raise ValueError(f"{label} contains duplicate samples")
    return tuple(samples)


def deterministic_task_stratum_take(
    rows: Sequence[Mapping[str, object]], *, sample_roles: Mapping[str, str], count: int, seed: int
) -> tuple[str, ...]:
    """Choose a reproducible task-then-stratum balanced held-out set."""
    if count <= 0:
        raise ValueError("audit count must be positive")
    by_lane: dict[str, list[tuple[str, str]]] = {lane: [] for lane in V8_TARGET}
    for row in rows:
        sample = row.get("sample")
        task = row.get("task", "")
        if not isinstance(sample, str) or not sample or not isinstance(task, str):
            raise ValueError("audit candidate has invalid sample or task")
        role = sample_roles.get(sample)
        if role not in by_lane:
            raise ValueError(f"audit candidate lacks a registered role: {sample}")
        by_lane[role].append((sample, task))
    quotas = {lane: int(count * fraction) for lane, fraction in V8_TARGET.items()}
    remaining = count - sum(quotas.values())
    for lane in sorted(V8_TARGET, key=lambda item: (-(count * V8_TARGET[item] - quotas[item]), item))[:remaining]:
        quotas[lane] += 1
    result: list[str] = []
    for lane, quota in quotas.items():
        candidates = by_lane[lane]
        if len(candidates) < quota:
            raise ValueError(f"v8 audit20 cannot satisfy held-out coverage: {lane}")
        candidates = sorted(
            candidates,
            key=lambda item: hashlib.sha256(f"{seed}:{lane}:{item[1]}:{item[0]}".encode()).digest(),
        )
        result.extend(sample for sample, _ in candidates[:quota])
    return tuple(sorted(result, key=lambda sample: hashlib.sha256(f"{seed}:audit:{sample}".encode()).digest()))


def select_v8_audit20(
    *,
    cached_rows: Sequence[Mapping[str, object]],
    heldout_rows: Sequence[Mapping[str, object]],
    dev_samples: Iterable[str],
    sample_roles: Mapping[str, str],
    seed: int = V8_AUDIT_SEED,
) -> tuple[str, ...]:
    cached = set(_require_samples(cached_rows, label="cached"))
    heldout = _require_samples(heldout_rows, label="probe heldout")
    dev = set(dev_samples)
    candidates = [row for row in heldout_rows if row["sample"] in cached and row["sample"] not in dev]
    selected = deterministic_task_stratum_take(candidates, sample_roles=sample_roles, count=20, seed=seed)
    if len(selected) != 20 or not set(selected) <= set(heldout):
        raise ValueError("v8 audit20 cannot satisfy held-out coverage")
    return selected


def build_v8_replay(
    *, optimizer_samples: Sequence[str], sample_roles: Mapping[str, str], steps: int, world_size: int, seed: int
) -> tuple[dict[str, object], ...]:
    """Bind every optimizer input and complete-action wrong family before CUDA."""
    if steps <= 0 or world_size <= 0:
        raise ValueError("steps and world_size must be positive")
    samples = tuple(optimizer_samples)
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("optimizer samples must be unique and non-empty")
    roles = {sample: sample_roles[sample] for sample in samples}
    history = build_balanced_history(
        roles, target=V8_TARGET, seed=seed, samples_seen=steps * world_size
    )
    records: list[dict[str, object]] = []
    for index, sample in enumerate(history):
        step = index // world_size + 1
        rank = index % world_size
        family = _NEGATIVE_CYCLE[(step - 1) % len(_NEGATIVE_CYCLE)]
        records.append({
            "optimizer_step": step,
            "rank": rank,
            "sample": sample,
            "sample_role": roles[sample],
            "noise_seed": int.from_bytes(hashlib.sha256(f"{seed}:noise:{step}:{rank}:{sample}".encode()).digest()[:8], "big"),
            "timestep_seed": int.from_bytes(hashlib.sha256(f"{seed}:time:{step}:{rank}:{sample}".encode()).digest()[:8], "big"),
            "negative_family": family,
            "shift_direction": 0 if family != "shift" else (1 if ((step // len(_NEGATIVE_CYCLE)) % 2) else -1),
        })
    return tuple(records)


def build_v8_data_receipt(
    *,
    cached_rows: Sequence[Mapping[str, object]],
    probe_split: Mapping[str, object],
    dev_rows: Sequence[Mapping[str, object]],
    sample_roles: Mapping[str, str],
    seed: int = V8_AUDIT_SEED,
    steps: int = 250,
    world_size: int = 1,
) -> V8DataReceipt:
    heldout_rows = probe_split.get("heldout_rows")
    if not isinstance(heldout_rows, list):
        raise ValueError("v6 probe split lacks heldout_rows")
    dev_samples = set(_require_samples(dev_rows, label="dev"))
    cached_samples = _require_samples(cached_rows, label="cached")
    audit = select_v8_audit20(
        cached_rows=cached_rows,
        heldout_rows=heldout_rows,
        dev_samples=dev_samples,
        sample_roles=sample_roles,
        seed=seed,
    )
    optimizer = tuple(sample for sample in cached_samples if sample not in set(audit) and sample not in dev_samples)
    if set(optimizer) & set(audit) or set(optimizer) & dev_samples:
        raise ValueError("v8 optimizer identities leak into audit or dev")
    replay = build_v8_replay(
        optimizer_samples=optimizer, sample_roles=sample_roles, steps=steps, world_size=world_size, seed=seed
    )
    return V8DataReceipt(audit_samples=audit, optimizer_samples=optimizer, replay=replay)


def validate_v8_correct_cache(
    cache_root: Path | str,
    *,
    expected_samples: Sequence[str],
    source_manifest_sha256: str,
    condition_directory: str = "wan_v8_se3_conditions",
) -> dict[str, object]:
    """Require every expected v8 row to have a provenance-bound correct SE(3) sidecar."""
    root = Path(cache_root)
    if not condition_directory or Path(condition_directory).name != condition_directory:
        raise ValueError("v8 condition directory must be one relative path component")
    samples = tuple(expected_samples)
    if not samples or len(samples) != len(set(samples)):
        raise ValueError("expected samples must be unique and non-empty")
    valid: list[str] = []
    first_error: ValueError | None = None
    for sample in samples:
        try:
            validate_se3_cache(root / condition_directory / f"{sample}.npz", source_manifest_sha256)
            valid.append(sample)
        except ValueError as exc:
            first_error = first_error or exc
    if len(valid) != len(samples):
        if first_error and "source manifest hash differs" in str(first_error):
            raise first_error
        raise ValueError(f"SE3 coverage {len(valid)}/{len(samples)}")
    return {
        "valid_samples": tuple(valid),
        "source_manifest_sha256": source_manifest_sha256,
        "condition_directory": condition_directory,
    }
