from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .robotwin_action_cache import validate_v3_condition_payload


_V3_CONDITION_SHAPES = {
    "raster": (81, 10, 60, 80),
    "left_pose": (81, 11),
    "right_pose": (81, 11),
    "condition_support": (2, 21, 15, 20),
    "loss_weight": (1, 21, 30, 40),
}


def _validate_pose_statistics_hash(value: object) -> str:
    statistics_hash = str(value)
    if re.fullmatch(r"[0-9a-f]{64}", statistics_hash) is None:
        raise ValueError("pose statistics hash must be a lowercase SHA-256")
    return statistics_hash


def _validated_payload_hash(path: Path) -> str:
    with np.load(path) as payload:
        try:
            statistics_hash = _validate_pose_statistics_hash(
                np.asarray(payload["pose_statistics_sha256"]).item()
            )
        except KeyError as exc:
            raise ValueError("cached action condition lacks pose statistics hash") from exc
        validate_v3_condition_payload(payload, statistics_hash)
    return statistics_hash


def build_cached_training_manifest(
    source_manifest: Path | str,
    cache_root: Path | str,
    output: Path | str,
) -> int:
    cache_root = Path(cache_root)
    rows = []
    statistics_hashes: set[str] = set()
    for line in Path(source_manifest).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        sample = row["sample"]
        cached = {
            **row,
            "latent": f"latents/{sample}.pt",
            "context": f"contexts/{sample}.pt",
            "action_raster": row.get("action_raster", f"action_rasters_v3/{sample}.npz"),
        }
        missing = [
            relative
            for relative in (cached["latent"], cached["context"], cached["action_raster"])
            if not (cache_root / relative).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"cached assets missing for {sample}: {missing}")
        statistics_hash = _validated_payload_hash(cache_root / cached["action_raster"])
        statistics_hashes.add(statistics_hash)
        cached["pose_statistics_sha256"] = statistics_hash
        rows.append(cached)
    if not rows:
        raise ValueError("cached training manifest would be empty")
    if len(statistics_hashes) != 1:
        raise ValueError("cached training manifest has mixed pose statistics hashes")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, output)
    return len(rows)


def _load_v3_condition(
    path: Path, *, sample: str, expected_statistics_hash: str
) -> dict[str, torch.Tensor]:
    with np.load(path) as payload:
        try:
            validate_v3_condition_payload(payload, expected_statistics_hash)
        except ValueError as exc:
            raise ValueError(f"invalid v3 action condition for {sample}: {exc}") from exc
        arrays = {}
        for key, shape in _V3_CONDITION_SHAPES.items():
            try:
                value = np.asarray(payload[key])
            except KeyError as exc:
                raise ValueError(f"cached action condition lacks {key} for {sample}") from exc
            if value.shape != shape or value.dtype != np.dtype(np.float16):
                raise ValueError(f"invalid {key} shape or dtype for {sample}")
            if not np.isfinite(value).all():
                raise ValueError(f"non-finite {key} for {sample}")
            arrays[key] = torch.from_numpy(value.astype(np.float32))
    return arrays


class WanActionCachedDataset(Dataset):
    def __init__(self, manifest: Path | str, cache_root: Path | str) -> None:
        self.cache_root = Path(cache_root)
        self.rows = [
            json.loads(line)
            for line in Path(manifest).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError("training manifest is empty")
        try:
            hashes = {
                _validate_pose_statistics_hash(row["pose_statistics_sha256"])
                for row in self.rows
            }
        except KeyError as exc:
            raise ValueError("training manifest lacks pose statistics hash") from exc
        if len(hashes) != 1:
            raise ValueError("training manifest has mixed pose statistics hashes")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        latent = torch.load(self.cache_root / row["latent"], map_location="cpu", weights_only=True)
        context = torch.load(self.cache_root / row["context"], map_location="cpu", weights_only=True)
        if latent.ndim != 4 or tuple(latent.shape) != (48, 21, 30, 40):
            raise ValueError(f"invalid v3 latent shape for {row['sample']}: {latent.shape}")
        if context.ndim != 2 or context.shape[1] != 4096:
            raise ValueError(f"invalid context shape for {row['sample']}: {context.shape}")
        statistics_hash = _validate_pose_statistics_hash(row["pose_statistics_sha256"])
        condition = _load_v3_condition(
            self.cache_root / row["action_raster"],
            sample=row["sample"],
            expected_statistics_hash=statistics_hash,
        )
        return {
            "latent": latent,
            "context": context,
            "action_raster": condition["raster"].permute(1, 0, 2, 3).contiguous(),
            "left_pose": condition["left_pose"],
            "right_pose": condition["right_pose"],
            "condition_support": condition["condition_support"],
            "loss_weight": condition["loss_weight"],
            "action_present": torch.tensor(1.0, dtype=torch.float32),
            "pose_statistics_sha256": statistics_hash,
            "sample": row["sample"],
            "task": row.get("task", ""),
        }
