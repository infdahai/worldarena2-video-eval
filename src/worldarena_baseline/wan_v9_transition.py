"""Compact phase-locked transition features for Wan v9.

The destination slot is structural metadata.  It is deliberately not exposed
as action-token content, which prevents shifted actions from carrying a source
phase identifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Final, Mapping, Sequence

import numpy as np


TRANSITION_CACHE_CONTRACT: Final = "wan-v9-transition-cache/1"
NORMALIZATION_CONTRACT: Final = "wan-v9-transition-normalization/1"
LATENT_ENDPOINT_FRAMES: Final = tuple(range(0, 81, 4))
_SHA256: Final = re.compile(r"[0-9a-f]{64}")
_MODALITY_SHAPES: Final = {
    "translation": (2, 20, 4),
    "rotation": (2, 20, 4),
    "image": (2, 20, 6),
    "gripper": (2, 20, 3),
}
_CACHE_KEYS: Final = {
    "contract",
    "sample",
    "variant",
    "translation",
    "rotation",
    "image",
    "gripper",
    "arm_present",
    "motion_active",
    "destination_slot",
    "source_hdf5_sha256",
    "source_action_sha256",
    "source_counterfactual_sha256",
    "urdf_sha256",
    "clean_manifest_sha256",
    "normalization_receipt_sha256",
    "payload_sha256",
}
_ALL_VARIANTS: Final = ("correct", "reverse", "shift+1", "shift-1", "swap")


@dataclass(frozen=True)
class TransitionFeatures:
    translation: np.ndarray
    rotation: np.ndarray
    image: np.ndarray
    gripper: np.ndarray
    arm_present: np.ndarray
    motion_active: np.ndarray
    destination_slot: np.ndarray


def _sample_identities(rows: Sequence[Mapping[str, object]], *, label: str) -> tuple[str, ...]:
    samples: list[str] = []
    for row in rows:
        sample = row.get("sample")
        if not isinstance(sample, str) or not sample:
            raise ValueError(f"{label} row has invalid sample")
        samples.append(sample)
    if len(samples) != len(set(samples)):
        raise ValueError(f"{label} contains duplicate samples")
    return tuple(samples)


def validate_split_identities(
    clean_rows: Sequence[Mapping[str, object]],
    audit_rows: Sequence[Mapping[str, object]],
    dev_rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    """Return the exact optimizer pool after enforcing the v9 zero-leakage split."""
    clean = _sample_identities(clean_rows, label="clean-1785")
    audit = _sample_identities(audit_rows, label="audit20")
    dev = _sample_identities(dev_rows, label="dev-fast20")
    if len(clean) != 1785:
        raise ValueError(f"v9 requires exactly clean-1785, got {len(clean)}")
    if len(audit) != 20:
        raise ValueError(f"v9 requires exactly audit20, got {len(audit)}")
    if len(dev) != 20:
        raise ValueError(f"v9 requires exactly dev-fast20, got {len(dev)}")
    clean_set, audit_set, dev_set = set(clean), set(audit), set(dev)
    if not audit_set <= clean_set:
        raise ValueError("audit20 is not a subset of clean-1785")
    if clean_set & dev_set:
        raise ValueError("dev-fast20 leaks into clean-1785")
    optimizer = tuple(sample for sample in clean if sample not in audit_set)
    if len(optimizer) != 1765 or set(optimizer) & (audit_set | dev_set):
        raise ValueError("v9 optimizer split is not clean-1785 minus audit20")
    return optimizer


def transition_variant_requirements(
    optimizer_samples: Sequence[str], audit_samples: Sequence[str]
) -> dict[str, tuple[str, ...]]:
    optimizer = tuple(optimizer_samples)
    audit = tuple(audit_samples)
    if not optimizer or not audit or len(set(optimizer)) != len(optimizer) or len(set(audit)) != len(audit):
        raise ValueError("v9 variant identities must be unique and non-empty")
    if set(optimizer) & set(audit):
        raise ValueError("v9 optimizer and audit identities overlap")
    return {sample: _ALL_VARIANTS for sample in (*optimizer, *audit)}


def _require_sha256(value: str, *, label: str, allow_empty: bool = False) -> None:
    if allow_empty and value == "":
        return
    if _SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")


def _validate_rotations(rotation: np.ndarray) -> np.ndarray:
    value = np.asarray(rotation, dtype=np.float64)
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError("rotation must contain finite 3x3 matrices")
    gram = value.swapaxes(-1, -2) @ value
    if not np.allclose(gram, np.eye(3), atol=1e-7, rtol=0):
        raise ValueError("rotation is not orthogonal")
    if not np.allclose(np.linalg.det(value), 1.0, atol=1e-7, rtol=0):
        raise ValueError("rotation determinant differs from one")
    return value


def _vee(matrix: np.ndarray) -> np.ndarray:
    return np.array(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=np.float64,
    ) * 0.5


def so3_log_map(rotation: np.ndarray) -> np.ndarray:
    """Return the canonical FP64 SO(3) rotation vector.

    The near-pi branch derives the axis from the symmetric part and fixes its
    sign using the skew part when available.  This avoids quaternion q/-q
    ambiguity and remains stable at the exact identity.
    """
    value = _validate_rotations(rotation)
    if value.shape != (3, 3):
        raise ValueError("so3_log_map accepts exactly one 3x3 rotation")
    cosine = float(np.clip((np.trace(value) - 1.0) * 0.5, -1.0, 1.0))
    angle = float(np.arccos(cosine))
    skew = _vee(value)
    if angle < 1e-7:
        return skew.copy()
    if np.pi - angle < 1e-5:
        diagonal = np.maximum((np.diag(value) + 1.0) * 0.5, 0.0)
        axis = np.sqrt(diagonal)
        pivot = int(np.argmax(axis))
        if axis[pivot] < 1e-10:
            raise ValueError("near-pi rotation has no stable axis")
        if pivot == 0:
            axis[1] = (value[0, 1] + value[1, 0]) / (4.0 * axis[0])
            axis[2] = (value[0, 2] + value[2, 0]) / (4.0 * axis[0])
        elif pivot == 1:
            axis[0] = (value[0, 1] + value[1, 0]) / (4.0 * axis[1])
            axis[2] = (value[1, 2] + value[2, 1]) / (4.0 * axis[1])
        else:
            axis[0] = (value[0, 2] + value[2, 0]) / (4.0 * axis[2])
            axis[1] = (value[1, 2] + value[2, 1]) / (4.0 * axis[2])
        norm = float(np.linalg.norm(axis))
        if norm <= 0 or not np.isfinite(norm):
            raise ValueError("near-pi rotation axis is invalid")
        axis /= norm
        if np.linalg.norm(skew) > 1e-10 and float(axis @ skew) < 0:
            axis *= -1.0
        return axis * angle
    return skew * (angle / np.sin(angle))


def _validate_states(states: np.ndarray, present: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transforms = np.asarray(states, dtype=np.float64)
    valid = np.asarray(present)
    if transforms.shape != (2, 21, 4, 4) or not np.isfinite(transforms).all():
        raise ValueError("states must have finite shape (2,21,4,4)")
    if valid.shape != (2, 21) or valid.dtype != np.dtype(bool):
        raise ValueError("state presence must have boolean shape (2,21)")
    if not np.allclose(transforms[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-8, rtol=0):
        raise ValueError("state homogeneous row is invalid")
    _validate_rotations(transforms[..., :3, :3])
    return transforms, valid


def physical_states_from_normalized_inverse(
    arm_transform: np.ndarray,
    arm_present: np.ndarray,
    motion_scale: float,
) -> np.ndarray:
    """Recover physical anchored poses from the v7 normalized inverse cache."""
    inverse = np.asarray(arm_transform, dtype=np.float64)
    present = np.asarray(arm_present)
    if inverse.shape != (2, 21, 4, 4):
        raise ValueError("normalized inverse transforms must have shape (2,21,4,4)")
    if present.shape != (2, 21) or present.dtype != np.dtype(bool):
        raise ValueError("normalized inverse presence must have boolean shape (2,21)")
    if not np.isfinite(motion_scale) or motion_scale <= 0:
        raise ValueError("motion_scale must be positive finite")
    if not np.isfinite(inverse).all() or not np.allclose(
        inverse[..., 3, :], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0
    ):
        raise ValueError("normalized inverse transforms are non-finite or non-homogeneous")
    raw_rotation = inverse[..., :3, :3]
    gram = raw_rotation.swapaxes(-1, -2) @ raw_rotation
    determinant = np.linalg.det(raw_rotation)
    if not np.allclose(gram, np.eye(3), atol=1e-4, rtol=0) or not np.allclose(
        determinant, 1.0, atol=1e-4, rtol=0
    ):
        raise ValueError("normalized inverse rotation exceeds cache tolerance")
    left, _singular, right = np.linalg.svd(raw_rotation)
    projected = left @ right
    negative = np.linalg.det(projected) < 0
    if np.any(negative):
        left = left.copy()
        left[negative, :, -1] *= -1
        projected = left @ right
    inverse = inverse.copy()
    inverse[..., :3, :3] = projected
    states = np.broadcast_to(np.eye(4, dtype=np.float64), inverse.shape).copy()
    states[..., :3, :3] = inverse[..., :3, :3].swapaxes(-1, -2)
    states[..., :3, 3] = -np.einsum(
        "...ij,...j->...i", states[..., :3, :3], inverse[..., :3, 3]
    )
    states[..., :3, 3] *= float(motion_scale)
    states[~present] = np.eye(4, dtype=np.float64)
    _validate_states(states, present)
    return states


def _endpoint_image_state(raster: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    value = np.asarray(raster, dtype=np.float64)
    if value.shape != (81, 10, 60, 80) or not np.isfinite(value).all():
        raise ValueError("raster must have finite shape (81,10,60,80)")
    uv = np.zeros((2, 21, 2), dtype=np.float64)
    opening = np.zeros((2, 21), dtype=np.float64)
    present = np.zeros((2, 21), dtype=bool)
    yy, xx = np.mgrid[:60, :80]
    for arm, (heatmap_channel, opening_channel) in enumerate(((1, 4), (6, 9))):
        for latent_index, frame_index in enumerate(LATENT_ENDPOINT_FRAMES):
            heatmap = value[frame_index, heatmap_channel]
            mass = float(heatmap.sum())
            if mass <= 1e-12:
                continue
            if np.any(heatmap < 0):
                raise ValueError("EEF heatmap cannot contain negative values")
            uv[arm, latent_index] = [float((heatmap * xx).sum() / mass), float((heatmap * yy).sum() / mass)]
            opening[arm, latent_index] = float(value[frame_index, opening_channel].sum() / mass)
            present[arm, latent_index] = True
    if np.any((opening[present] < -1e-6) | (opening[present] > 1.0 + 1e-6)):
        raise ValueError("gripper opening must be in [0,1]")
    return uv, np.clip(opening, 0.0, 1.0), present


def build_transition_features(
    states: np.ndarray,
    state_present: np.ndarray,
    raster: np.ndarray,
) -> TransitionFeatures:
    """Derive 20 per-arm transitions from 21 SE(3) states and 81 raster frames."""
    transforms, state_valid = _validate_states(states, state_present)
    uv, opening, image_valid = _endpoint_image_state(raster)
    present = state_valid[:, :-1] & state_valid[:, 1:] & image_valid[:, :-1] & image_valid[:, 1:]
    translation = np.zeros((2, 20, 4), dtype=np.float64)
    rotation = np.zeros((2, 20, 4), dtype=np.float64)
    image = np.zeros((2, 20, 6), dtype=np.float64)
    gripper = np.zeros((2, 20, 3), dtype=np.float64)
    for arm in range(2):
        for interval in range(20):
            if not present[arm, interval]:
                continue
            left = transforms[arm, interval]
            right = transforms[arm, interval + 1]
            relative = np.eye(4, dtype=np.float64)
            relative[:3, :3] = left[:3, :3].T @ right[:3, :3]
            relative[:3, 3] = left[:3, :3].T @ (right[:3, 3] - left[:3, 3])
            delta = relative[:3, 3]
            rotation_vector = so3_log_map(relative[:3, :3])
            delta_uv = uv[arm, interval + 1] - uv[arm, interval]
            delta_opening = opening[arm, interval + 1] - opening[arm, interval]
            translation[arm, interval] = (*delta, float(np.linalg.norm(delta)))
            rotation[arm, interval] = (*rotation_vector, float(np.linalg.norm(rotation_vector)))
            image[arm, interval] = (*uv[arm, interval], *uv[arm, interval + 1], *delta_uv)
            gripper[arm, interval] = (opening[arm, interval], opening[arm, interval + 1], delta_opening)
    active = present & (
        (translation[..., 3] > 0.001)
        | (rotation[..., 3] > np.deg2rad(0.5))
        | (np.linalg.norm(image[..., 4:6], axis=-1) > 0.25)
        | (np.abs(gripper[..., 2]) > 0.01)
    )
    return TransitionFeatures(
        translation=translation,
        rotation=rotation,
        image=image,
        gripper=gripper,
        arm_present=present,
        motion_active=active,
        destination_slot=np.arange(20, dtype=np.int16),
    )


def shift_transition_content(features: TransitionFeatures, *, direction: int) -> TransitionFeatures:
    """Shift only action content while preserving the destination-slot contract.

    Positive shift delays content by one destination interval; negative shift
    advances it.  The exposed boundary holds the nearest legal content rather
    than wrapping cyclically.  No source interval identifier is retained.
    """
    if direction not in (-1, 1):
        raise ValueError("transition shift direction must be +1 or -1")
    payload = {
        "translation": np.asarray(features.translation),
        "rotation": np.asarray(features.rotation),
        "image": np.asarray(features.image),
        "gripper": np.asarray(features.gripper),
        "arm_present": np.asarray(features.arm_present),
        "motion_active": np.asarray(features.motion_active),
        "destination_slot": np.asarray(features.destination_slot),
    }
    _validate_feature_arrays(payload)

    def shifted(value: np.ndarray) -> np.ndarray:
        result = np.empty_like(value)
        if direction == 1:
            result[:, 0] = value[:, 0]
            result[:, 1:] = value[:, :-1]
        else:
            result[:, :-1] = value[:, 1:]
            result[:, -1] = value[:, -1]
        return result

    return TransitionFeatures(
        translation=shifted(payload["translation"]),
        rotation=shifted(payload["rotation"]),
        image=shifted(payload["image"]),
        gripper=shifted(payload["gripper"]),
        arm_present=shifted(payload["arm_present"]),
        motion_active=shifted(payload["motion_active"]),
        destination_slot=np.arange(20, dtype=np.int16),
    )


def _feature_payload(features: TransitionFeatures) -> dict[str, np.ndarray]:
    payload = {
        "translation": np.asarray(features.translation, dtype=np.float32),
        "rotation": np.asarray(features.rotation, dtype=np.float32),
        "image": np.asarray(features.image, dtype=np.float32),
        "gripper": np.asarray(features.gripper, dtype=np.float32),
        "arm_present": np.asarray(features.arm_present, dtype=bool),
        "motion_active": np.asarray(features.motion_active, dtype=bool),
        "destination_slot": np.asarray(features.destination_slot, dtype=np.int16),
    }
    _validate_feature_arrays(payload)
    return payload


def _validate_feature_arrays(payload: Mapping[str, np.ndarray]) -> None:
    for name, shape in _MODALITY_SHAPES.items():
        value = np.asarray(payload[name])
        if value.shape != shape or value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError(f"{name} has invalid shape or dtype")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values")
    for name in ("arm_present", "motion_active"):
        value = np.asarray(payload[name])
        if value.shape != (2, 20) or value.dtype != np.dtype(bool):
            raise ValueError(f"{name} must have boolean shape (2,20)")
    slots = np.asarray(payload["destination_slot"])
    if slots.shape != (20,) or slots.dtype != np.dtype(np.int16) or not np.array_equal(slots, np.arange(20, dtype=np.int16)):
        raise ValueError("destination slots must be exact 0..19 int16")
    if np.any(np.asarray(payload["motion_active"]) & ~np.asarray(payload["arm_present"])):
        raise ValueError("motion activity cannot exist for an absent arm")


def _canonical_feature_hash(payload: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(key for key in payload if key != "payload_sha256"):
        value = np.ascontiguousarray(payload[key])
        digest.update(key.encode())
        digest.update(value.dtype.str.encode())
        digest.update(json.dumps(value.shape, separators=(",", ":")).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def fit_normalization_statistics(features_by_sample: Mapping[str, TransitionFeatures]) -> dict[str, object]:
    """Fit deterministic correct-only statistics from explicit sample identities."""
    if not features_by_sample:
        raise ValueError("normalization requires at least one correct sample")
    samples = sorted(features_by_sample)
    result: dict[str, object] = {"contract": NORMALIZATION_CONTRACT, "samples": samples}
    for modality in _MODALITY_SHAPES:
        rows: list[np.ndarray] = []
        for sample in samples:
            features = features_by_sample[sample]
            values = np.asarray(getattr(features, modality), dtype=np.float64)
            present = np.asarray(features.arm_present, dtype=bool)
            if present.any():
                rows.append(values[present])
        if not rows:
            raise ValueError(f"normalization has no present values for {modality}")
        combined = np.concatenate(rows, axis=0)
        mean = combined.mean(axis=0)
        std = np.maximum(combined.std(axis=0), 1e-6)
        result[modality] = {"mean": mean.tolist(), "std": std.tolist()}
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    result["receipt_sha256"] = hashlib.sha256(encoded).hexdigest()
    return result


def _scalar(value: str) -> np.ndarray:
    return np.asarray(value, dtype=np.str_)


def write_transition_cache_atomic(
    path: Path | str,
    features: TransitionFeatures,
    *,
    sample: str,
    variant: str,
    source_hdf5_sha256: str,
    source_action_sha256: str,
    source_counterfactual_sha256: str,
    urdf_sha256: str,
    clean_manifest_sha256: str,
    normalization_receipt_sha256: str,
) -> Path:
    if not sample or "/" in sample or sample in {".", ".."}:
        raise ValueError("sample is invalid")
    if variant not in {"correct", "reverse", "shift+1", "shift-1", "swap"}:
        raise ValueError("transition variant is invalid")
    for label, value, allow_empty in (
        ("source HDF5 hash", source_hdf5_sha256, False),
        ("source action hash", source_action_sha256, False),
        ("source counterfactual hash", source_counterfactual_sha256, variant == "correct"),
        ("URDF hash", urdf_sha256, False),
        ("clean manifest hash", clean_manifest_sha256, False),
        ("normalization receipt hash", normalization_receipt_sha256, False),
    ):
        _require_sha256(value, label=label, allow_empty=allow_empty)
    payload: dict[str, np.ndarray] = {
        "contract": _scalar(TRANSITION_CACHE_CONTRACT),
        "sample": _scalar(sample),
        "variant": _scalar(variant),
        **_feature_payload(features),
        "source_hdf5_sha256": _scalar(source_hdf5_sha256),
        "source_action_sha256": _scalar(source_action_sha256),
        "source_counterfactual_sha256": _scalar(source_counterfactual_sha256),
        "urdf_sha256": _scalar(urdf_sha256),
        "clean_manifest_sha256": _scalar(clean_manifest_sha256),
        "normalization_receipt_sha256": _scalar(normalization_receipt_sha256),
    }
    payload["payload_sha256"] = _scalar(_canonical_feature_hash(payload))
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.partial")
    try:
        with temporary.open("xb") as handle:
            np.savez(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    validate_transition_cache(
        destination,
        expected_sample=sample,
        expected_variant=variant,
        expected_clean_manifest_sha256=clean_manifest_sha256,
        expected_normalization_receipt_sha256=normalization_receipt_sha256,
    )
    return destination


def _read_scalar(payload: Mapping[str, np.ndarray], key: str) -> str:
    value = np.asarray(payload[key])
    if value.shape != () or value.dtype.kind != "U":
        raise ValueError(f"cached {key} must be scalar unicode")
    return str(value.item())


def validate_transition_cache(
    path: Path | str,
    *,
    expected_sample: str,
    expected_variant: str,
    expected_clean_manifest_sha256: str,
    expected_normalization_receipt_sha256: str,
) -> dict[str, np.ndarray]:
    cache_path = Path(path)
    if not cache_path.is_file() or cache_path.is_symlink():
        raise ValueError("transition cache must be a regular file")
    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            payload = {key: np.asarray(archive[key]) for key in archive.files}
    except (OSError, KeyError, ValueError) as exc:
        raise ValueError("cannot load transition cache") from exc
    if set(payload) != _CACHE_KEYS:
        raise ValueError("transition cache schema keys differ")
    if _read_scalar(payload, "contract") != TRANSITION_CACHE_CONTRACT:
        raise ValueError("transition cache contract differs")
    if _read_scalar(payload, "sample") != expected_sample:
        raise ValueError("transition cache sample differs")
    if _read_scalar(payload, "variant") != expected_variant:
        raise ValueError("transition cache variant differs")
    if _read_scalar(payload, "clean_manifest_sha256") != expected_clean_manifest_sha256:
        raise ValueError("transition cache clean manifest differs")
    if _read_scalar(payload, "normalization_receipt_sha256") != expected_normalization_receipt_sha256:
        raise ValueError("transition cache normalization receipt differs")
    for key in (
        "source_hdf5_sha256", "source_action_sha256", "urdf_sha256",
        "clean_manifest_sha256", "normalization_receipt_sha256", "payload_sha256",
    ):
        _require_sha256(_read_scalar(payload, key), label=f"cached {key}")
    counterfactual = _read_scalar(payload, "source_counterfactual_sha256")
    _require_sha256(counterfactual, label="cached source_counterfactual_sha256", allow_empty=expected_variant == "correct")
    _validate_feature_arrays(payload)
    expected_hash = _canonical_feature_hash(payload)
    if _read_scalar(payload, "payload_sha256") != expected_hash:
        raise ValueError("transition cache payload hash differs")
    return payload
