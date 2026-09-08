"""Frozen CPU-only fit of legacy 81-frame train160; run beside predicted14_ranker.py.

Reads the five authorized files in place. Never exports the original manifest,
uses holdout labels, reads old native16, changes routing, or invokes GPU scoring.
Run: /data/di/worldarena2_track1_20260815/envs/flowwam-v16/bin/python fit_train160.py
"""

from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
import predicted14_ranker as ranker


ARTIFACT_ROOT = Path("/data/di/worldarena2_track1_20260815")
ADAPTIVE_ROOT = ARTIFACT_ROOT / "runs/flowwam-official/adaptive-candidate-system"
SOURCE_FILES = {
    "manifest": (ADAPTIVE_ROOT / "manifests/selector-200.jsonl", "4b147c15ba6b9fad45d66cdc542e57bbe9a967c0cd8d05a5f267cad7c702a18e"),
    "seed1_raw": (ARTIFACT_ROOT / "official_track1_eval/checkpoints/step-0750/dev-clean-50/csv_results/aggregated_results.csv", "c0c0f4b579091f2f7a0d5df16d97eafc9344725339909abb65279d38d6c29ab8"),
    "seed4_raw": (ARTIFACT_ROOT / "official_track1_eval/checkpoints/step-0760/dev-clean-50/csv_results/aggregated_results.csv", "88da51d621c2250419a026a1f82fe61efd2130e507e0c8cb710599652cbfa7bf"),
    "seed1_corrected": (ADAPTIVE_ROOT / "latest-scorer/seed1-corrected.csv", "a1f1d64b9b7446d3f7eb6346c97177abceabab0cf5ce7b566d7fb0b0bfe4d8ae"),
    "seed4_corrected": (ADAPTIVE_ROOT / "latest-scorer/seed4-corrected.csv", "4e49a42e8c8f6c3b48930c60ff184fc3137645c038fdd3afe0a8e2f42c5c3d65"),
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def csv_identity(video_id):
    match = re.fullmatch(r"(.+)_episode_?(\d+)", video_id)
    if not match:
        raise ValueError(f"invalid task-qualified Video_ID: {video_id!r}")
    return match.group(1), int(match.group(2))


def manifest_index(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != 200 or Counter(row["split"] for row in rows) != {"train": 160, "holdout": 40}:
        raise ValueError("manifest must contain exactly train160 and holdout40")
    result = {}
    for row in rows:
        task, index = row["task"], row["episode_index"]
        if not isinstance(task, str) or not task or type(index) is not int or index < 0:
            raise ValueError("invalid manifest task/episode identity")
        identity = (task, index)
        if identity in result:
            raise ValueError("duplicate manifest task+episode")
        result[identity] = row["split"]
    return result


def score_index(path, expected_ids, model_name):
    result = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV requires nonduplicate columns")
        for row in reader:
            identity = csv_identity(row["Video_ID"])
            if identity in result or row["Model_Name"] != model_name:
                raise ValueError("duplicate CSV task+episode or unexpected model")
            result[identity] = row
    if set(result) != set(expected_ids):
        raise ValueError("CSV task+episode identities differ from exact manifest set")
    return result


def main():
    started = time.perf_counter()
    output = Path(__file__).resolve().parent
    for name in ("model.json", "train.rows.jsonl", "fit.complete.json"):
        if (output / name).exists():
            raise FileExistsError(f"refusing to overwrite existing fit artifact: {name}")
    inputs = {}
    for name, (path, expected_sha) in SOURCE_FILES.items():
        actual_sha = sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(f"frozen source SHA mismatch: {name}")
        inputs[name] = {"path": str(path), "sha256": actual_sha, "bytes": path.stat().st_size}
    manifest = manifest_index(SOURCE_FILES["manifest"][0])
    train_ids = sorted(identity for identity, split in manifest.items() if split == "train")
    holdout_ids = {identity for identity, split in manifest.items() if split == "holdout"}
    if len(train_ids) != 160 or set(train_ids) & holdout_ids:
        raise ValueError("train/holdout overlap or wrong train count")
    rows = []
    for seed in (1, 4):
        model_name = f"FlowWAMAdaptiveSeed{seed}Train200"
        raw = score_index(SOURCE_FILES[f"seed{seed}_raw"][0], manifest, model_name)
        corrected = score_index(SOURCE_FILES[f"seed{seed}_corrected"][0], manifest, model_name)
        for task, episode in train_ids:
            identity = (task, episode)
            rows.append({
                "task": task, "episode_id": f"{task}_episode{episode:06d}", "seed": seed,
                "features": {name: float(raw[identity][name]) for name in ranker.FEATURE_SCHEMA},
                "targets": {name: float(corrected[identity][name]) for name in ranker.TARGET_SCHEMA},
            })
    if len(rows) != 320:
        raise ValueError("expected exactly 320 training candidates")
    model = ranker.fit_ranker(rows)
    serialized = ranker.dumps_model(model) + "\n"
    if ranker.loads_model(serialized) != model:
        raise ValueError("serialization roundtrip changed fitted model")
    (output / "train.rows.jsonl").write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows))
    (output / "model.json").write_text(serialized)
    schemas = {"features": list(ranker.FEATURE_SCHEMA), "targets": list(ranker.TARGET_SCHEMA)}
    receipt = {
        "status": "cpu_fit_complete", "score_kind": "prediction_not_official",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started, "numpy_version": np.__version__,
        "train_episode_count": 160, "train_candidate_count": 320, "seeds": [1, 4],
        "train_episode_ids": model["train_episode_ids"],
        "train_task_counts": dict(sorted(Counter(task for task, _ in train_ids).items())),
        "manifest_episode_count": 200, "excluded_holdout_episode_count": len(holdout_ids),
        "train_holdout_overlap_count": 0, "holdout_labels_used": False,
        "join_key": "exact (task, episode_index); never bare episode number",
        "schemas": schemas,
        "schema_sha256": hashlib.sha256(json.dumps(schemas, sort_keys=True).encode()).hexdigest(),
        "alpha": 10.0, "sample_weight": "1 / candidate_count_per_episode",
        "inference_policy": {"delta_threshold": 0.003, "protected_max_drop": None, "status": "fixed_not_executed"},
        "training_label_domain": "legacy_81_frame_corrected14_not_native_validated",
        "label_domain_provenance": "project/user-supplied; no video probing performed in CPU fit",
        "fresh_native_full15_with_real_collection_jepa_required": True,
        "jepa_used_as_target": False, "old_native16_read_or_used": False,
        "routes_or_choices_modified": False, "full_manifest_exported": False,
        "inputs": inputs, "fit_data_digest": model["data_digest"],
        "code_sha256": {"fit_train160.py": sha256(Path(__file__)), "predicted14_ranker.py": sha256(Path(ranker.__file__))},
        "output_sha256": {name: sha256(output / name) for name in ("model.json", "train.rows.jsonl")},
    }
    (output / "fit.complete.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: receipt[key] for key in ("status", "train_episode_count", "train_candidate_count", "excluded_holdout_episode_count", "alpha", "output_sha256", "elapsed_seconds")}, sort_keys=True))


if __name__ == "__main__":
    main()
