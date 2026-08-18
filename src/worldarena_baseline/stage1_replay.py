"""Versioned, exact Stage-1 replay schedules and checkpoint provenance."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any, Mapping


REPLAY_SCHEMA_VERSION = 1
WORLD_SIZE = 8
DATASET_ROWS = 200
GLOBAL_STEPS = 200
LATENT_SHAPE = (48, 21, 30, 40)
_GENERATOR = {
    "algorithm": "torch.Generator.manual_seed+torch.randn",
    "device_rule": "cpu_then_to_cuda_rank",
}
_NOISE_CANARY_SEED = 2026081501
_NOISE_CANARY_SHAPE = (7, 11, 13)
_NOISE_CANARY_DTYPE = "float32"


class ReplayCompatibilityError(ValueError):
    """Raised when a replay or checkpoint cannot belong to this experiment."""


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: object, field: str, *, nullable: bool = False) -> str | None:
    if nullable and value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise ReplayCompatibilityError(f"{field} must be a SHA-256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ReplayCompatibilityError(f"{field} must be a SHA-256") from exc
    if value != value.lower():
        raise ReplayCompatibilityError(f"{field} must be a lowercase SHA-256")
    return value


def _read_source_rows(source_manifest: Path | str) -> list[dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in Path(source_manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != DATASET_ROWS:
        raise ValueError("Stage-1 replay requires exactly 200 rows")
    tasks = {str(row.get("task", "")) for row in rows}
    if "" in tasks or len(tasks) != 4:
        raise ValueError("Stage-1 replay requires exactly four tasks")
    return rows


def _noise_canary_sha256() -> str:
    """Hash fixed actual torch CPU noise bytes for the recorded runtime contract."""
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_NOISE_CANARY_SEED)
    tensor = torch.randn(
        _NOISE_CANARY_SHAPE,
        generator=generator,
        device="cpu",
        dtype=torch.float32,
    )
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()


def _generator_descriptor(torch_version: str, noise_check_sha256: str) -> dict[str, Any]:
    _require_sha256(noise_check_sha256, "deterministic_check_sha256")
    descriptor: dict[str, Any] = {
        **_GENERATOR,
        "torch_version": str(torch_version),
        "noise_canary_seed": _NOISE_CANARY_SEED,
        "noise_canary_shape": list(_NOISE_CANARY_SHAPE),
        "noise_canary_dtype": _NOISE_CANARY_DTYPE,
        "deterministic_check_sha256": noise_check_sha256,
    }
    return descriptor


def build_stage1_replay_manifest(
    source_manifest: Path | str,
    output: Path | str,
    *,
    seed: int = 20260815,
    action_dropout_probability: float = 0.1,
    text_dropout_probability: float = 0.1,
    torch_version: str = "runtime-validated",
    deterministic_check_sha256: str | None = None,
    world_size: int = WORLD_SIZE,
) -> dict[str, Any]:
    """Atomically write a balanced exact replay for a fixed seven/eight-rank run."""
    if not 0 <= action_dropout_probability <= 1:
        raise ValueError("action dropout probability must be in [0, 1]")
    if not 0 <= text_dropout_probability <= 1:
        raise ValueError("text dropout probability must be in [0, 1]")
    if world_size not in (7, 8):
        raise ValueError("Stage-1 replay world_size must be seven or eight")
    _read_source_rows(source_manifest)
    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    record_count = GLOBAL_STEPS * world_size
    if record_count % DATASET_ROWS:
        raise ValueError("Stage-1 replay records must form complete dataset epochs")
    for _epoch in range(record_count // DATASET_ROWS):
        permutation = list(range(DATASET_ROWS))
        rng.shuffle(permutation)
        for sample_index in permutation:
            record_index = len(records)
            records.append(
                {
                    "step": record_index // world_size + 1,
                    "rank": record_index % world_size,
                    "sample_index": sample_index,
                    "timestep": rng.randint(1, 999),
                    "noise_seed": rng.randrange(0, 2**63),
                    "action_dropout": rng.random() < action_dropout_probability,
                    "text_dropout": rng.random() < text_dropout_probability,
                }
            )
    generator = _generator_descriptor(
        torch_version,
        _noise_canary_sha256() if deterministic_check_sha256 is None else deterministic_check_sha256,
    )
    payload: dict[str, Any] = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_manifest_sha256": sha256_file(source_manifest),
        "source_rows": DATASET_ROWS,
        "task_count": 4,
        "world_size": world_size,
        "global_steps": GLOBAL_STEPS,
        "latent_shape": list(LATENT_SHAPE),
        "seed": seed,
        "generator": generator,
        "records": records,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(partial, output)
    return payload


def _validate_replay_payload(
    payload: Mapping[str, Any],
    *,
    source_manifest: Path | str,
    torch_version: str | None,
    expected_world_size: int | None,
) -> None:
    if payload.get("schema_version") != REPLAY_SCHEMA_VERSION:
        raise ReplayCompatibilityError("replay schema_version mismatch")
    if payload.get("source_manifest_sha256") != sha256_file(source_manifest):
        raise ReplayCompatibilityError("replay source manifest SHA-256 mismatch")
    if payload.get("source_rows") != DATASET_ROWS or payload.get("task_count") != 4:
        raise ReplayCompatibilityError("replay dataset contract mismatch")
    world_size = payload.get("world_size")
    if world_size not in (7, 8) or payload.get("global_steps") != GLOBAL_STEPS:
        raise ReplayCompatibilityError("replay world-size or global-step contract mismatch")
    if expected_world_size is not None and world_size != expected_world_size:
        raise ReplayCompatibilityError("replay world-size differs from the active run")
    if payload.get("latent_shape") != list(LATENT_SHAPE):
        raise ReplayCompatibilityError("replay latent shape mismatch")
    generator = payload.get("generator")
    if not isinstance(generator, dict):
        raise ReplayCompatibilityError("replay generator is missing")
    if (
        generator.get("algorithm") != _GENERATOR["algorithm"]
        or generator.get("device_rule") != _GENERATOR["device_rule"]
        or generator.get("noise_canary_seed") != _NOISE_CANARY_SEED
        or generator.get("noise_canary_shape") != list(_NOISE_CANARY_SHAPE)
        or generator.get("noise_canary_dtype") != _NOISE_CANARY_DTYPE
    ):
        raise ReplayCompatibilityError("replay generator contract mismatch")
    _require_sha256(generator.get("deterministic_check_sha256"), "deterministic_check_sha256")
    if torch_version is not None and generator["torch_version"] != torch_version:
        raise ReplayCompatibilityError("replay torch version mismatch")
    if torch_version is not None and generator["deterministic_check_sha256"] != _noise_canary_sha256():
        raise ReplayCompatibilityError("replay deterministic noise check mismatch")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != GLOBAL_STEPS * world_size:
        raise ReplayCompatibilityError("replay record count mismatch")
    for index, record in enumerate(records):
        if record.get("step") != index // world_size + 1 or record.get("rank") != index % world_size:
            raise ReplayCompatibilityError("replay record ordering mismatch")
        sample = record.get("sample_index")
        if not isinstance(sample, int) or not 0 <= sample < DATASET_ROWS:
            raise ReplayCompatibilityError("replay sample index mismatch")
        if not isinstance(record.get("timestep"), int) or not 1 <= record["timestep"] <= 999:
            raise ReplayCompatibilityError("replay timestep mismatch")
        if not isinstance(record.get("noise_seed"), int) or not 0 <= record["noise_seed"] < 2**63:
            raise ReplayCompatibilityError("replay noise seed mismatch")
        if not isinstance(record.get("action_dropout"), bool) or not isinstance(record.get("text_dropout"), bool):
            raise ReplayCompatibilityError("replay dropout flag mismatch")
    for offset in range(0, len(records), DATASET_ROWS):
        samples = [record["sample_index"] for record in records[offset : offset + DATASET_ROWS]]
        if sorted(samples) != list(range(DATASET_ROWS)):
            raise ReplayCompatibilityError("replay epoch is not balanced")


def load_stage1_replay_manifest(
    path: Path | str,
    *,
    source_manifest: Path | str,
    torch_version: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReplayCompatibilityError("replay root must be an object")
    _read_source_rows(source_manifest)
    _validate_replay_payload(
        payload,
        source_manifest=source_manifest,
        torch_version=torch_version,
        expected_world_size=expected_world_size,
    )
    return payload


def replay_record(replay: Mapping[str, Any], *, step: int, rank: int) -> dict[str, Any]:
    if not 1 <= step <= GLOBAL_STEPS:
        raise ReplayCompatibilityError("replay step must be in [1, 200]")
    world_size = replay.get("world_size")
    if world_size not in (7, 8):
        raise ReplayCompatibilityError("replay world-size contract mismatch")
    if not 0 <= rank < world_size:
        raise ReplayCompatibilityError("replay rank is outside the recorded world-size")
    records = replay["records"]
    return dict(records[(step - 1) * world_size + rank])


def replay_noise(
    record: Mapping[str, Any],
    *,
    shape: tuple[int, ...],
    device: object,
    dtype: object,
):
    """Generate replay noise from an isolated CPU generator, then transfer it."""
    import torch

    noise_seed = record.get("noise_seed")
    if not isinstance(noise_seed, int) or not 0 <= noise_seed < 2**63:
        raise ReplayCompatibilityError("replay noise seed mismatch")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(noise_seed)
    return torch.randn(shape, generator=generator, device="cpu", dtype=torch.float32).to(
        device=device, dtype=dtype, non_blocking=True
    )


def checkpoint_provenance(
    *,
    replay_sha256: str,
    source_manifest_sha256: str,
    branch: str,
    parent_checkpoint_sha256: str | None,
    global_completed_step: int,
    world_size: int,
    rank_mapping: tuple[int, ...],
    optimizer_updates_since_parent: int,
    cumulative_path_updates: int,
    experiment_compute_updates: int,
) -> dict[str, Any]:
    if branch not in {"prefix", "A", "B"}:
        raise ValueError("branch must be prefix, A, or B")
    _require_sha256(replay_sha256, "replay_sha256")
    _require_sha256(source_manifest_sha256, "source_manifest_sha256")
    _require_sha256(parent_checkpoint_sha256, "parent_checkpoint_sha256", nullable=True)
    if branch == "prefix" and parent_checkpoint_sha256 is not None:
        raise ValueError("prefix checkpoint cannot have a parent checkpoint")
    if branch != "prefix" and parent_checkpoint_sha256 is None:
        raise ValueError("branch checkpoint requires a parent checkpoint")
    if not 0 <= global_completed_step <= GLOBAL_STEPS:
        raise ValueError("global completed step must be in [0, 200]")
    if world_size not in (7, 8) or rank_mapping != tuple(range(world_size)):
        raise ValueError("checkpoint rank mapping must exactly cover the fixed world-size")
    expected_updates = global_completed_step if branch == "prefix" else global_completed_step - 50
    if optimizer_updates_since_parent != expected_updates:
        raise ValueError("optimizer updates since parent mismatch")
    if cumulative_path_updates != global_completed_step:
        raise ValueError("cumulative path updates mismatch")
    if not cumulative_path_updates <= experiment_compute_updates <= GLOBAL_STEPS:
        raise ValueError("experiment compute updates mismatch")
    training_lineage = {
        "prefix": f"s1_prefix_{global_completed_step:04d}",
        "A": f"s1A_branch_{global_completed_step:04d}",
        "B": f"s1B_branch_{global_completed_step:04d}",
    }[branch]
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "replay_sha256": replay_sha256,
        "source_manifest_sha256": source_manifest_sha256,
        "branch": branch,
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
        "global_completed_step": global_completed_step,
        "world_size": world_size,
        "rank_mapping": list(rank_mapping),
        "training_lineage": training_lineage,
        "optimizer_updates_since_parent": optimizer_updates_since_parent,
        "cumulative_path_updates": cumulative_path_updates,
        "experiment_compute_updates": experiment_compute_updates,
    }


def validate_checkpoint_provenance(
    actual: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in (
        "schema_version",
        "replay_sha256",
        "source_manifest_sha256",
        "branch",
        "parent_checkpoint_sha256",
        "global_completed_step",
        "world_size",
        "rank_mapping",
        "training_lineage",
        "optimizer_updates_since_parent",
        "cumulative_path_updates",
        "experiment_compute_updates",
    ):
        if actual.get(field) != expected.get(field):
            raise ReplayCompatibilityError(f"checkpoint provenance {field} mismatch")
