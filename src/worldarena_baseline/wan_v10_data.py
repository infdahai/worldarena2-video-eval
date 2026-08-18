"""Leakage-free full-action split and relation-cache contracts for Wan v10."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any

import numpy as np

from .wan_v9_transition import build_transition_features, so3_log_map


DATA_CONTRACT = "wan-v10-full-action-data/1"
RELATION_CACHE_CONTRACT = "wan-v10-relation-cache/1"
STRATA = ("single_dominant", "bimanual_heavy", "mixed", "quiet")
AUDIT_TAGS = ("single_arm", "bimanual", "sequential", "crossing_or_overlap")
VARIANTS = ("correct", "reverse", "swap")
_SHA = re.compile(r"[0-9a-f]{64}")
_FEATURE_SHAPES = {
    "anchored_se3": (21, 2, 6),
    "velocity": (21, 2, 6),
    "uv": (21, 2, 2),
    "gripper": (21, 2, 2),
    "arm_present": (21, 2),
    "motion_active": (21, 2),
}


@dataclass(frozen=True)
class V10Split:
    optimizer: tuple[dict[str, Any], ...]
    audit: tuple[dict[str, Any], ...]
    source_counts: dict[str, int]
    audit_tags: tuple[str, ...]


@dataclass(frozen=True)
class V10RelationFeatures:
    anchored_se3: np.ndarray
    velocity: np.ndarray
    uv: np.ndarray
    gripper: np.ndarray
    arm_present: np.ndarray
    motion_active: np.ndarray


def build_v10_cache_extension(
    source_rows: Sequence[Mapping[str, Any]],
    cached_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Return the exact source rows missing from an independently valid cache."""

    source = _validate_rows(source_rows, label="v10 source")
    cached = _validate_rows(cached_rows, label="existing cache")
    source_by_sample = {str(row["sample"]): row for row in source}
    cached_samples = {str(row["sample"]) for row in cached}
    if not cached_samples <= set(source_by_sample):
        raise ValueError("existing cache contains a foreign v10 source identity")
    return tuple(
        source_by_sample[sample]
        for sample in sorted(set(source_by_sample) - cached_samples)
    )


def build_v10_relation_features(
    states: np.ndarray,
    state_present: np.ndarray,
    raster: np.ndarray,
) -> V10RelationFeatures:
    """Build exact 21-slot relation state from physical anchored SE(3) poses."""

    transforms = np.asarray(states, dtype=np.float64)
    present = np.asarray(state_present)
    if transforms.shape != (2, 21, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("v10 relation states must have finite shape (2,21,4,4)")
    if present.shape != (2, 21) or present.dtype != np.dtype(bool):
        raise ValueError("v10 relation state presence must have boolean shape (2,21)")
    transitions = build_transition_features(transforms, present, raster)
    anchored = np.zeros((21, 2, 6), dtype=np.float32)
    velocity = np.zeros_like(anchored)
    uv = np.zeros((21, 2, 2), dtype=np.float32)
    gripper = np.zeros((21, 2, 2), dtype=np.float32)
    slot_present = np.zeros((21, 2), dtype=bool)
    motion_active = np.zeros((21, 2), dtype=bool)
    for arm in range(2):
        for time in range(21):
            if present[arm, time]:
                anchored[time, arm, :3] = transforms[arm, time, :3, 3]
                anchored[time, arm, 3:] = so3_log_map(transforms[arm, time, :3, :3])
        interval_present = np.asarray(transitions.arm_present[arm], dtype=bool)
        if interval_present[0]:
            uv[0, arm] = transitions.image[arm, 0, :2]
            gripper[0, arm, 0] = transitions.gripper[arm, 0, 0]
            slot_present[0, arm] = True
        for interval in range(20):
            time = interval + 1
            if not interval_present[interval]:
                continue
            velocity[time, arm, :3] = transitions.translation[arm, interval, :3]
            velocity[time, arm, 3:] = transitions.rotation[arm, interval, :3]
            uv[time, arm] = transitions.image[arm, interval, 2:4]
            gripper[time, arm] = (
                transitions.gripper[arm, interval, 1],
                transitions.gripper[arm, interval, 2],
            )
            slot_present[time, arm] = True
            motion_active[time, arm] = transitions.motion_active[arm, interval]
    slot_present &= present.T
    absent = ~slot_present
    for values in (anchored, velocity, uv, gripper):
        values[absent] = 0
    return V10RelationFeatures(
        anchored_se3=anchored,
        velocity=velocity,
        uv=uv,
        gripper=gripper,
        arm_present=slot_present,
        motion_active=motion_active & slot_present,
    )


def _so3_exp_map(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError("v10 rotation vector must have finite shape (3,)")
    angle = float(np.linalg.norm(value))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = value / angle
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


def relation_features_from_v9_transition(
    payload: Mapping[str, np.ndarray],
    *,
    exact_states: np.ndarray | None = None,
    exact_state_present: np.ndarray | None = None,
) -> V10RelationFeatures:
    """Lift a contiguous v9 transition cache into v10 slot features."""

    required = {"translation", "rotation", "image", "gripper", "arm_present", "motion_active"}
    if not required <= set(payload):
        raise ValueError("v9 transition payload lacks relation feature arrays")
    translation = np.asarray(payload["translation"], dtype=np.float64)
    rotation = np.asarray(payload["rotation"], dtype=np.float64)
    image = np.asarray(payload["image"], dtype=np.float64)
    gripper_delta = np.asarray(payload["gripper"], dtype=np.float64)
    interval_present = np.asarray(payload["arm_present"])
    active = np.asarray(payload["motion_active"])
    if (
        translation.shape != (2, 20, 4)
        or rotation.shape != (2, 20, 4)
        or image.shape != (2, 20, 6)
        or gripper_delta.shape != (2, 20, 3)
        or interval_present.shape != (2, 20)
        or interval_present.dtype != np.dtype(bool)
        or active.shape != (2, 20)
        or active.dtype != np.dtype(bool)
        or not all(np.isfinite(value).all() for value in (translation, rotation, image, gripper_delta))
    ):
        raise ValueError("v9 transition payload has invalid relation arrays")
    if np.any(active & ~interval_present):
        raise ValueError("v9 transition activity exists for an absent arm")
    if exact_states is not None or exact_state_present is not None:
        if exact_states is None or exact_state_present is None:
            raise ValueError("exact v10 states and presence must be supplied together")
        states = np.asarray(exact_states, dtype=np.float64)
        state_present = np.asarray(exact_state_present)
        if states.shape != (2, 21, 4, 4) or state_present.shape != (2, 21):
            raise ValueError("exact v10 states/presence shape differs")
        if state_present.dtype != np.dtype(bool) or not np.isfinite(states).all():
            raise ValueError("exact v10 states/presence are invalid")
    else:
        states = np.broadcast_to(np.eye(4, dtype=np.float64), (2, 21, 4, 4)).copy()
        state_present = np.zeros((2, 21), dtype=bool)
        for arm in range(2):
            present_indices = np.flatnonzero(interval_present[arm])
            if len(present_indices) and not np.array_equal(
                present_indices, np.arange(present_indices[-1] + 1)
            ):
                raise ValueError("v9 transition presence must be a contiguous prefix without exact states")
            if not len(present_indices):
                continue
            state_present[arm, : present_indices[-1] + 2] = True
            for interval in present_indices:
                current = states[arm, interval]
                states[arm, interval + 1, :3, :3] = (
                    current[:3, :3] @ _so3_exp_map(rotation[arm, interval, :3])
                )
                states[arm, interval + 1, :3, 3] = (
                    current[:3, 3] + current[:3, :3] @ translation[arm, interval, :3]
                )
    anchored = np.zeros((21, 2, 6), dtype=np.float32)
    velocity = np.zeros_like(anchored)
    uv = np.zeros((21, 2, 2), dtype=np.float32)
    gripper = np.zeros((21, 2, 2), dtype=np.float32)
    slot_present = np.zeros((21, 2), dtype=bool)
    slot_present[0] = interval_present[:, 0]
    slot_present[1:] = interval_present.T
    slot_present &= state_present.T
    motion_active = np.zeros((21, 2), dtype=bool)
    for arm in range(2):
        for time in np.flatnonzero(state_present[arm]):
            anchored[time, arm, :3] = states[arm, time, :3, 3]
            anchored[time, arm, 3:] = so3_log_map(states[arm, time, :3, :3])
        for interval in np.flatnonzero(interval_present[arm]):
            time = interval + 1
            velocity[time, arm, :3] = translation[arm, interval, :3]
            velocity[time, arm, 3:] = rotation[arm, interval, :3]
            if interval == 0:
                uv[0, arm] = image[arm, interval, :2]
                gripper[0, arm, 0] = gripper_delta[arm, interval, 0]
            uv[time, arm] = image[arm, interval, 2:4]
            gripper[time, arm] = gripper_delta[arm, interval, 1:3]
            motion_active[time, arm] = active[arm, interval]
    absent = ~slot_present
    for values in (anchored, velocity, uv, gripper):
        values[absent] = 0
    return V10RelationFeatures(
        anchored_se3=anchored,
        velocity=velocity,
        uv=uv,
        gripper=gripper,
        arm_present=slot_present,
        motion_active=motion_active & slot_present,
    )


def fit_v10_relation_normalization(
    features_by_sample: Mapping[str, V10RelationFeatures],
) -> dict[str, Any]:
    """Fit correct-only state statistics from explicit optimizer identities."""

    if not features_by_sample:
        raise ValueError("v10 relation normalization requires optimizer samples")
    samples = sorted(features_by_sample)
    rows: list[np.ndarray] = []
    for sample in samples:
        arrays = _validate_features(features_by_sample[sample])
        state = np.concatenate(
            (
                arrays["anchored_se3"], arrays["velocity"],
                arrays["uv"], arrays["gripper"],
            ),
            axis=-1,
        )
        if arrays["arm_present"].any():
            rows.append(state[arrays["arm_present"]].astype(np.float64))
    if not rows:
        raise ValueError("v10 relation normalization has no present arm states")
    combined = np.concatenate(rows, axis=0)
    mean = np.concatenate((combined.mean(axis=0), np.zeros(4, dtype=np.float64)))
    scale = np.concatenate(
        (np.maximum(combined.std(axis=0), 1e-6), np.ones(4, dtype=np.float64))
    )
    payload: dict[str, Any] = {
        "contract": "wan-v10-relation-normalization/1",
        "samples": samples,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable(seed: int, *values: object) -> bytes:
    return hashlib.sha256(":".join((str(seed), *map(str, values))).encode()).digest()


def _validate_rows(rows: Sequence[Mapping[str, Any]], *, label: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    samples: set[str] = set()
    for raw in rows:
        row = dict(raw)
        sample = row.get("sample")
        task = row.get("task")
        hdf5 = row.get("hdf5")
        if not all(isinstance(value, str) and value for value in (sample, task, hdf5)):
            raise ValueError(f"{label} row identity/path is invalid")
        if sample in samples:
            raise ValueError(f"{label} contains duplicate sample identities")
        samples.add(sample)
        hdf5_path = PurePosixPath(hdf5)
        if hdf5_path.is_absolute() or ".." in hdf5_path.parts:
            raise ValueError(f"{label} HDF5 path escapes its source root")
        if "robot_only" in hdf5_path.stem:
            raise ValueError("robot_only RGB derivatives are not v10 action episodes")
        if ".quarantine" in hdf5_path.parts:
            raise ValueError("quarantine episodes are not v10 optimizer eligible")
        result.append(row)
    return result


def _validate_metadata(
    source: Sequence[Mapping[str, Any]], metadata: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    samples = {str(row["sample"]) for row in source}
    if set(metadata) != samples:
        raise ValueError("v10 role/audit metadata must cover the exact source pool")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        item = dict(metadata[sample])
        if item.get("stratum") not in STRATA:
            raise ValueError("v10 source metadata has an unknown stratum")
        tags = item.get("tags")
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise ValueError("v10 audit tags must be a list of strings")
        for count_name in ("position_valid_count", "velocity_valid_count"):
            if type(item.get(count_name)) is not int or item[count_name] < 0:
                raise ValueError("v10 audit valid counts must be non-negative integers")
        if type(item.get("probe_observable")) is not bool:
            raise ValueError("v10 probe observability must be boolean")
        result[sample] = item
    return result


def _choose_audit(
    candidates: Sequence[dict[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    seed: int,
    size: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in candidates
        if metadata[str(row["sample"])]["probe_observable"]
        and metadata[str(row["sample"])]["position_valid_count"] >= 8
        and metadata[str(row["sample"])]["velocity_valid_count"] >= 4
    ]
    if len(eligible) < size:
        raise ValueError(f"v10 requires {size} observable audit candidates")
    ordered = sorted(eligible, key=lambda row: _stable(seed, "audit", row["sample"]))
    selected: list[dict[str, Any]] = []
    selected_samples: set[str] = set()
    selected_tasks: set[str] = set()
    for tag in AUDIT_TAGS:
        options = [
            row for row in ordered
            if tag in metadata[str(row["sample"])]["tags"]
            and str(row["sample"]) not in selected_samples
        ]
        if not options:
            raise ValueError(f"v10 audit has no eligible {tag} sample")
        row = next(
            (item for item in options if str(item["task"]) not in selected_tasks),
            options[0],
        )
        selected.append(row)
        selected_samples.add(str(row["sample"]))
        selected_tasks.add(str(row["task"]))
    while len(selected) < size:
        remaining = [row for row in ordered if str(row["sample"]) not in selected_samples]
        if not remaining:
            raise ValueError(f"v10 requires {size} observable audit candidates")
        row = next(
            (item for item in remaining if str(item["task"]) not in selected_tasks),
            remaining[0],
        )
        selected.append(row)
        selected_samples.add(str(row["sample"]))
        selected_tasks.add(str(row["task"]))
    if len(selected_tasks) < 8:
        raise ValueError("v10 audit must cover at least eight tasks")
    return selected


def _annotate(row: Mapping[str, Any], metadata: Mapping[str, Any], *, audit: bool) -> dict[str, Any]:
    result = {**dict(row), "v10_stratum": str(metadata["stratum"])}
    if audit:
        result["v10_audit"] = {
            "position_valid_count": int(metadata["position_valid_count"]),
            "velocity_valid_count": int(metadata["velocity_valid_count"]),
            "tags": sorted(set(metadata["tags"])),
        }
    return result


def build_v10_split(
    source_rows: Sequence[Mapping[str, Any]],
    dev_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Mapping[str, Any]],
    *,
    seed: int = 20260819,
    audit_size: int = 20,
) -> V10Split:
    if type(seed) is not int or seed < 0 or audit_size != 20:
        raise ValueError("v10 split requires a non-negative seed and exact audit20")
    source = _validate_rows(source_rows, label="v10 source")
    dev = _validate_rows(dev_rows, label="dev-fast20")
    source_by_sample = {str(row["sample"]): row for row in source}
    dev_samples = {str(row["sample"]) for row in dev}
    if len(dev) != 20 or not dev_samples <= set(source_by_sample):
        raise ValueError("v10 requires dev-fast20 to be an exact source subset")
    details = _validate_metadata(source, metadata)
    candidates = [row for row in source if str(row["sample"]) not in dev_samples]
    audit_source = _choose_audit(candidates, details, seed=seed, size=audit_size)
    audit_samples = {str(row["sample"]) for row in audit_source}
    optimizer_source = [row for row in source if str(row["sample"]) not in dev_samples | audit_samples]
    optimizer = tuple(
        _annotate(row, details[str(row["sample"])], audit=False)
        for row in sorted(optimizer_source, key=lambda item: str(item["sample"]))
    )
    audit = tuple(
        _annotate(row, details[str(row["sample"])], audit=True)
        for row in sorted(audit_source, key=lambda item: str(item["sample"]))
    )
    counts = Counter(str(item["stratum"]) for item in details.values())
    tags = tuple(sorted({tag for row in audit for tag in row["v10_audit"]["tags"]}))
    return V10Split(
        optimizer=optimizer,
        audit=audit,
        source_counts={name: int(counts[name]) for name in STRATA},
        audit_tags=tags,
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read v10 manifest: {path}") from exc
    if not rows:
        raise ValueError(f"v10 manifest is empty: {path}")
    return rows


def build_v10_data_receipt(
    *,
    source_manifest: Path,
    optimizer_manifest: Path,
    audit_manifest: Path,
    dev_manifest: Path,
    split: V10Split,
    official_test_status: str,
) -> dict[str, Any]:
    if official_test_status != "unavailable":
        raise ValueError("official test must remain unavailable during v10 training")
    payload = {
        "contract": DATA_CONTRACT,
        "source_manifest_sha256": _sha256_file(source_manifest),
        "optimizer_manifest_sha256": _sha256_file(optimizer_manifest),
        "audit_manifest_sha256": _sha256_file(audit_manifest),
        "dev_manifest_sha256": _sha256_file(dev_manifest),
        "source_count": len(_read_jsonl(source_manifest)),
        "optimizer_count": len(split.optimizer),
        "audit_count": len(split.audit),
        "dev_count": len(_read_jsonl(dev_manifest)),
        "source_counts": dict(split.source_counts),
        "audit_tags": list(split.audit_tags),
        "official_test_status": official_test_status,
        "selection_seed": 20260819,
    }
    validate_v10_data_receipt(
        payload,
        source_manifest=source_manifest,
        optimizer_manifest=optimizer_manifest,
        audit_manifest=audit_manifest,
        dev_manifest=dev_manifest,
    )
    return payload


def validate_v10_data_receipt(
    receipt: Mapping[str, Any],
    *,
    source_manifest: Path,
    optimizer_manifest: Path,
    audit_manifest: Path,
    dev_manifest: Path,
) -> dict[str, Any]:
    if receipt.get("contract") != DATA_CONTRACT or receipt.get("official_test_status") != "unavailable":
        raise ValueError("v10 data receipt contract/status differs")
    paths = {
        "source": source_manifest,
        "optimizer": optimizer_manifest,
        "audit": audit_manifest,
        "dev": dev_manifest,
    }
    rows = {name: _read_jsonl(path) for name, path in paths.items()}
    for name, path in paths.items():
        if receipt.get(f"{name}_manifest_sha256") != _sha256_file(path):
            raise ValueError(f"v10 {name} manifest hash differs")
        if receipt.get(f"{name}_count") != len(rows[name]):
            raise ValueError(f"v10 {name} manifest count differs")
    identities = {
        name: {str(row.get("sample")) for row in values}
        for name, values in rows.items()
    }
    if any(len(identities[name]) != len(rows[name]) for name in rows):
        raise ValueError("v10 manifest contains duplicate identities")
    if not (identities["optimizer"] | identities["audit"] | identities["dev"]) <= identities["source"]:
        raise ValueError("v10 split contains a foreign source identity")
    if (
        identities["optimizer"] & identities["audit"]
        or identities["optimizer"] & identities["dev"]
        or identities["audit"] & identities["dev"]
    ):
        raise ValueError("v10 optimizer/audit/dev leakage detected")
    if identities["optimizer"] | identities["audit"] | identities["dev"] != identities["source"]:
        raise ValueError("v10 source identities were silently dropped")
    if len(rows["audit"]) != 20 or len(rows["dev"]) != 20:
        raise ValueError("v10 requires exact audit20 and dev-fast20")
    return dict(receipt)


def _validate_features(features: V10RelationFeatures) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for name, shape in _FEATURE_SHAPES.items():
        value = np.asarray(getattr(features, name))
        dtype = np.dtype(bool) if name in {"arm_present", "motion_active"} else np.dtype(np.float32)
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(f"v10 relation {name} shape/dtype differs")
        if dtype != np.dtype(bool) and not np.isfinite(value).all():
            raise ValueError(f"v10 relation {name} contains non-finite values")
        result[name] = value
    if np.any(result["motion_active"] & ~result["arm_present"]):
        raise ValueError("v10 relation activity exists for an absent arm")
    absent = ~result["arm_present"]
    for name in ("anchored_se3", "velocity", "uv", "gripper"):
        if np.count_nonzero(result[name][absent]):
            raise ValueError("v10 absent-arm relation features must be exact zero")
    return result


def _payload_hash(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in payload if key != "payload_sha256"):
        value = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _require_sha(value: str, *, label: str) -> None:
    if _SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def write_v10_relation_cache_atomic(
    path: Path,
    *,
    sample: str,
    variants: Mapping[str, V10RelationFeatures],
    source_hdf5_sha256: str,
    source_action_sha256: str,
    source_urdf_sha256: str,
    normalization_sha256: str,
) -> Path:
    if set(variants) != set(VARIANTS) or not sample:
        raise ValueError("v10 relation cache requires correct/reverse/swap and one sample")
    for label, value in (
        ("source HDF5", source_hdf5_sha256),
        ("source action", source_action_sha256),
        ("source URDF", source_urdf_sha256),
        ("normalization", normalization_sha256),
    ):
        _require_sha(value, label=label)
    payload: dict[str, np.ndarray] = {
        "contract": np.asarray(RELATION_CACHE_CONTRACT),
        "sample": np.asarray(sample),
        "source_hdf5_sha256": np.asarray(source_hdf5_sha256),
        "source_action_sha256": np.asarray(source_action_sha256),
        "source_urdf_sha256": np.asarray(source_urdf_sha256),
        "normalization_sha256": np.asarray(normalization_sha256),
    }
    for variant in VARIANTS:
        for name, value in _validate_features(variants[variant]).items():
            payload[f"{variant}_{name}"] = value
    payload["payload_sha256"] = np.asarray(_payload_hash(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.{os.getpid()}.partial")
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    sidecar_partial = sidecar.with_name(f".{sidecar.name}.{os.getpid()}.partial")
    try:
        with partial.open("xb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        file_sha = _sha256_file(partial)
        sidecar_partial.write_text(
            json.dumps(
                {"contract": RELATION_CACHE_CONTRACT, "file_sha256": file_sha,
                 "payload_sha256": str(payload["payload_sha256"].item())},
                sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
        os.replace(sidecar_partial, sidecar)
    finally:
        partial.unlink(missing_ok=True)
        sidecar_partial.unlink(missing_ok=True)
    validate_v10_relation_cache(
        path, expected_sample=sample, expected_normalization_sha256=normalization_sha256
    )
    return path


def validate_v10_relation_cache(
    path: Path,
    *,
    expected_sample: str,
    expected_normalization_sha256: str,
) -> dict[str, np.ndarray]:
    _require_sha(expected_normalization_sha256, label="expected normalization")
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not path.is_file() or path.is_symlink() or not sidecar.is_file() or sidecar.is_symlink():
        raise ValueError("cannot load v10 relation cache or sidecar")
    try:
        metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise ValueError("cannot load v10 relation cache") from exc
    if metadata.get("contract") != RELATION_CACHE_CONTRACT or metadata.get("file_sha256") != _sha256_file(path):
        raise ValueError("v10 relation cache file hash differs")
    scalar = lambda key: str(np.asarray(payload[key]).item())
    if scalar("contract") != RELATION_CACHE_CONTRACT or scalar("sample") != expected_sample:
        raise ValueError("v10 relation cache contract/sample differs")
    if scalar("normalization_sha256") != expected_normalization_sha256:
        raise ValueError("v10 relation cache normalization differs")
    for key in (
        "source_hdf5_sha256", "source_action_sha256", "source_urdf_sha256",
        "normalization_sha256",
    ):
        _require_sha(scalar(key), label=f"cached {key}")
    expected_keys = {
        "contract", "sample", "source_hdf5_sha256", "source_action_sha256",
        "source_urdf_sha256",
        "normalization_sha256", "payload_sha256",
        *(f"{variant}_{name}" for variant in VARIANTS for name in _FEATURE_SHAPES),
    }
    if set(payload) != expected_keys:
        raise ValueError("v10 relation cache schema keys differ")
    actual_payload_hash = _payload_hash(payload)
    if scalar("payload_sha256") != actual_payload_hash or metadata.get("payload_sha256") != actual_payload_hash:
        raise ValueError("v10 relation cache payload hash differs")
    for variant in VARIANTS:
        _validate_features(
            V10RelationFeatures(**{
                name: payload[f"{variant}_{name}"] for name in _FEATURE_SHAPES
            })
        )
    return payload
