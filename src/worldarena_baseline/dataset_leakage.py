"""Fail-closed dataset identity and clean scaling contracts."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


class DatasetLeakageError(ValueError):
    """Raised when a training identity appears in an evaluation split."""


def _identities(row: Mapping[str, object]) -> set[str]:
    identities: set[str] = set()
    sample = row.get("sample")
    if isinstance(sample, str) and sample:
        identities.add(f"sample:{sample}")
    task, variant, episode = row.get("task"), row.get("variant"), row.get("episode_index")
    if isinstance(task, str) and task and isinstance(variant, str) and variant and isinstance(episode, int):
        identities.add(f"episode:{task}/{variant}/{episode}")
    raw_hdf5 = row.get("source_hdf5", row.get("hdf5"))
    if isinstance(raw_hdf5, str) and raw_hdf5:
        parts = PurePosixPath(raw_hdf5).parts
        suffix = "/".join(parts[-4:]) if len(parts) >= 4 else "/".join(parts)
        identities.add(f"hdf5:{suffix}")
    if not identities:
        raise DatasetLeakageError("dataset row has no stable episode identity")
    return identities


def assert_zero_dataset_leakage(
    train_rows: Iterable[Mapping[str, object]],
    evaluation_rows: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, object]:
    train = list(train_rows)
    evaluation = {name: list(rows) for name, rows in evaluation_rows.items()}
    eval_index: dict[str, set[str]] = {}
    for split, rows in evaluation.items():
        for row in rows:
            for identity in _identities(row):
                eval_index.setdefault(identity, set()).add(split)
    collisions: list[dict[str, object]] = []
    for row in train:
        for identity in sorted(_identities(row)):
            if identity in eval_index:
                collisions.append({
                    "sample": row.get("sample"),
                    "identity": identity,
                    "evaluation_splits": sorted(eval_index[identity]),
                })
    if collisions:
        first = collisions[0]
        raise DatasetLeakageError(
            f"dataset leakage: {first['identity']} sample={first['sample']} "
            f"appears in {','.join(first['evaluation_splits'])}; collisions={len(collisions)}"
        )
    return {
        "contract": "wan-action-dataset-zero-leakage/1",
        "passed": True,
        "train_rows": len(train),
        "evaluation_rows": {name: len(rows) for name, rows in sorted(evaluation.items())},
        "identity_fields": ["sample", "task+variant+episode_index", "hdf5"],
        "collision_count": 0,
    }


def _rank(sample: str, seed: int) -> bytes:
    return hashlib.sha256(f"{seed}:{sample}".encode()).digest()


def _evaluation_identity_sha256(rows: Iterable[Mapping[str, object]]) -> str:
    identities = sorted({identity for row in rows for identity in _identities(row)})
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def build_clean_scale_manifests(
    *,
    source_rows: Iterable[Mapping[str, object]],
    eligible_samples: set[str],
    old_small_rows: Iterable[Mapping[str, object]],
    evaluation_rows: Mapping[str, Iterable[Mapping[str, object]]],
    small_size: int,
    seed: int,
) -> dict[str, object]:
    source = [dict(row) for row in source_rows]
    by_sample = {str(row.get("sample")): row for row in source}
    if len(by_sample) != len(source) or "None" in by_sample:
        raise DatasetLeakageError("source manifest samples must be unique non-empty strings")
    if not eligible_samples <= set(by_sample):
        raise DatasetLeakageError("eligible sample is absent from source manifest")
    evaluation = {name: [dict(row) for row in rows] for name, rows in evaluation_rows.items()}
    eval_identities = {
        identity
        for rows in evaluation.values()
        for row in rows
        for identity in _identities(row)
    }
    large = [
        by_sample[sample]
        for sample in sorted(eligible_samples)
        if _identities(by_sample[sample]).isdisjoint(eval_identities)
    ]
    old_small = [dict(row) for row in old_small_rows]
    kept = []
    seen: set[str] = set()
    large_samples = {str(row["sample"]) for row in large}
    for row in old_small:
        sample = str(row.get("sample"))
        if sample in large_samples and sample not in seen:
            kept.append(by_sample[sample])
            seen.add(sample)
    if small_size <= 0 or small_size > len(large):
        raise DatasetLeakageError("small_size must fit inside the clean eligible dataset")
    fill = sorted(
        (row for row in large if str(row["sample"]) not in seen),
        key=lambda row: (_rank(str(row["sample"]), seed), str(row["sample"])),
    )[: small_size - len(kept)]
    small = sorted([*kept, *fill], key=lambda row: str(row["sample"]))
    if len(small) != small_size:
        raise DatasetLeakageError("could not fill clean small manifest")
    small_check = assert_zero_dataset_leakage(small, evaluation)
    large_check = assert_zero_dataset_leakage(large, evaluation)
    if not {str(row["sample"]) for row in small} <= large_samples:
        raise DatasetLeakageError("clean small manifest is not nested in clean large manifest")
    return {
        "small_rows": small,
        "large_rows": large,
        "receipt": {
            "contract": "wan-action-clean-data-scale-split/1",
            "seed": seed,
            "small_rows": len(small),
            "large_rows": len(large),
            "eligible_rows_before_exclusion": len(eligible_samples),
            "excluded_evaluation_rows": len(eligible_samples) - len(large),
            "preserved_from_old_small": len(kept),
            "replaced_from_old_small": len(old_small) - len(kept),
            "nested": True,
            "evaluation_identity_sha256": {
                name: _evaluation_identity_sha256(rows)
                for name, rows in sorted(evaluation.items())
            },
            "leakage_check": {
                "passed": bool(small_check["passed"] and large_check["passed"]),
                "small": small_check,
                "large": large_check,
            },
        },
    }


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(text, encoding="utf-8")
    os.replace(partial, path)


def _jsonl(rows: list[Mapping[str, object]]) -> str:
    return "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_clean_scale_artifacts(output_root: Path | str, result: Mapping[str, object]) -> dict[str, Path]:
    root = Path(output_root)
    small = list(result["small_rows"])
    large = list(result["large_rows"])
    small_path = root / f"clean-{len(small)}.jsonl"
    large_path = root / f"clean-{len(large)}.jsonl"
    if small_path == large_path:
        raise DatasetLeakageError("clean scale manifests must have different sizes")
    _atomic_text(small_path, _jsonl(small))
    _atomic_text(large_path, _jsonl(large))
    receipt = dict(result["receipt"])
    receipt.update({
        "small_manifest": str(small_path),
        "small_manifest_sha256": _sha256(small_path),
        "large_manifest": str(large_path),
        "large_manifest_sha256": _sha256(large_path),
    })
    receipt_path = root / "clean-data-scale-receipt.json"
    _atomic_text(receipt_path, json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return {"small_manifest": small_path, "large_manifest": large_path, "receipt": receipt_path}


def validate_training_manifest_receipt(
    manifest: Path | str,
    receipt_path: Path | str,
    evaluation_rows: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, object]:
    manifest_path = Path(manifest)
    receipt = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    if receipt.get("contract") != "wan-action-clean-data-scale-split/1":
        raise DatasetLeakageError("unsupported data leakage receipt contract")
    digest = _sha256(manifest_path)
    expected: dict[str, int] = {}
    for prefix in ("small", "large"):
        expected_digest = receipt.get(f"{prefix}_manifest_sha256")
        expected_rows = receipt.get(f"{prefix}_rows")
        if isinstance(expected_digest, str) and isinstance(expected_rows, int):
            expected[expected_digest] = expected_rows
    if digest not in expected:
        raise DatasetLeakageError("training manifest digest is absent from leakage receipt")
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(rows) != expected[digest]:
        raise DatasetLeakageError("training manifest row count differs from leakage receipt")
    evaluation = {name: list(values) for name, values in evaluation_rows.items()}
    expected_eval = receipt.get("evaluation_identity_sha256")
    actual_eval = {
        name: _evaluation_identity_sha256(values)
        for name, values in sorted(evaluation.items())
    }
    if expected_eval != actual_eval:
        raise DatasetLeakageError("evaluation identities differ from leakage receipt")
    report = assert_zero_dataset_leakage(rows, evaluation)
    return {**report, "manifest_sha256": digest, "receipt": str(receipt_path)}


def validate_parent_dataset_lineage(
    parent_payload: Mapping[str, object],
    *,
    parent_manifest: Path | str,
    evaluation_rows: Mapping[str, Iterable[Mapping[str, object]]],
) -> dict[str, object]:
    """Prove that every inherited Adapter update used a known clean manifest."""
    stage1 = parent_payload.get("stage1")
    if not isinstance(stage1, Mapping):
        raise DatasetLeakageError("parent lacks auditable Stage-1 data lineage")
    manifest = Path(parent_manifest)
    digest = _sha256(manifest)
    if stage1.get("source_manifest_sha256") != digest:
        raise DatasetLeakageError("parent manifest hash differs from checkpoint lineage")
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    report = assert_zero_dataset_leakage(rows, evaluation_rows)
    return {
        **report,
        "parent_manifest": str(manifest),
        "parent_manifest_sha256": digest,
    }
