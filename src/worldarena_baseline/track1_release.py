from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import cv2

_METADATA_FIELDS = {
    "model_name",
    "version",
    "organization",
    "release_year",
    "source_type",
    "control_type",
}
_SOURCE_TYPES = {"open_source", "closed_source"}
_CONTROL_TYPES = {"text_driven", "action_driven", "hybrid"}
_P0_SOURCE_METRICS = (
    "Instruction Following",
    "Interaction Quality",
    "Perspectivity",
    "Image Quality",
    "Aesthetic Quality",
    "Photometric Consistency",
    "Dynamic Degree",
    "Flow Score",
    "Motion Smoothness",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} missing: {path}")
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"{label} SHA256 mismatch: expected {expected}, got {actual}")


def _validate_metadata(metadata: dict[str, Any]) -> None:
    missing = sorted(_METADATA_FIELDS - metadata.keys())
    if missing:
        raise ValueError(f"submission metadata missing fields: {missing}")
    for field in ("model_name", "version", "organization"):
        if not isinstance(metadata[field], str) or not metadata[field].strip():
            raise ValueError(f"submission metadata field {field} must be non-empty")
    if not isinstance(metadata["release_year"], int):
        raise ValueError("submission metadata release_year must be an integer")
    if metadata["source_type"] not in _SOURCE_TYPES:
        raise ValueError(f"unsupported source_type: {metadata['source_type']}")
    if metadata["control_type"] not in _CONTROL_TYPES:
        raise ValueError(f"unsupported control_type: {metadata['control_type']}")


def freeze_p0_release(
    config_path: Path | str, output_receipt: Path | str
) -> dict[str, Any]:
    config_path = Path(config_path)
    output_receipt = Path(output_receipt)
    config = _read_json(config_path)
    if config.get("contract") != "worldarena-track1-p0-release-config/1":
        raise ValueError("unsupported P0 release contract")
    expected_count = config.get("expected_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        raise ValueError("expected_count must be a positive integer")
    metadata = config.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    _validate_metadata(metadata)

    p0 = config.get("p0")
    if not isinstance(p0, dict):
        raise ValueError("p0 must be an object")
    paths: dict[str, Path] = {}
    for key, label in (
        ("final_decision", "final decision"),
        ("model", "model"),
        ("policy", "policy"),
    ):
        path = Path(p0.get(key, ""))
        expected_hash = p0.get(f"{key}_sha256")
        if not isinstance(expected_hash, str):
            raise ValueError(f"{label} SHA256 missing")
        _require_hash(path, expected_hash, label)
        paths[key] = path

    dataset_receipt_path = Path(config.get("dataset_receipt", ""))
    dataset_receipt_hash = config.get("dataset_receipt_sha256")
    if not isinstance(dataset_receipt_hash, str):
        raise ValueError("dataset receipt SHA256 missing")
    _require_hash(dataset_receipt_path, dataset_receipt_hash, "dataset receipt")

    final = _read_json(paths["final_decision"])
    if final.get("completed") is not True:
        raise ValueError("final decision is not complete")
    if final.get("terminal_state") != "complete_retain_p0":
        raise ValueError("final decision does not retain P0")
    if final.get("final_winner") != "p0_postgen_selector":
        raise ValueError("final decision winner is not P0 postgen selector")
    if final.get("deployment_sha_gate") is not True:
        raise ValueError("final decision deployment SHA gate did not pass")
    deployment = final.get("deployment")
    if not isinstance(deployment, dict):
        raise ValueError("final decision deployment is missing")
    for key in ("model", "policy"):
        if Path(deployment.get(key, "")) != paths[key]:
            raise ValueError(f"final decision deployment {key} path mismatch")
        if deployment.get(f"{key}_sha256") != p0[f"{key}_sha256"]:
            raise ValueError(f"final decision deployment {key} SHA256 mismatch")

    model = _read_json(paths["model"])
    if model.get("feature_policy") != "generated_only_no_hidden_gt":
        raise ValueError("model feature policy permits hidden ground truth")
    policy = _read_json(paths["policy"])
    if policy.get("completed") is not True:
        raise ValueError("policy receipt is not complete")
    if policy.get("selected_policy") != "postgen_selector":
        raise ValueError("policy receipt did not select postgen selector")

    dataset = _read_json(dataset_receipt_path)
    if dataset.get("completed") is not True:
        raise ValueError("official dataset receipt is not complete")
    if dataset.get("episode_count") != expected_count:
        raise ValueError("official dataset episode count mismatch")
    manifest_path = Path(dataset.get("episode_manifest", ""))
    archive_path = Path(dataset.get("archive", ""))
    _require_hash(
        manifest_path, dataset.get("episode_manifest_sha256", ""), "episode manifest"
    )
    _require_hash(
        archive_path, dataset.get("archive_sha256", ""), "official dataset archive"
    )
    usage = dataset.get("usage_contract")
    required_usage = {
        "final_inference_and_submission_only": True,
        "model_selection_allowed": False,
        "selector_fit_allowed": False,
        "threshold_tuning_allowed": False,
        "training_allowed": False,
    }
    if not isinstance(usage, dict) or any(
        usage.get(k) is not v for k, v in required_usage.items()
    ):
        raise ValueError("official dataset usage contract is not enforced")

    receipt = {
        "completed": True,
        "terminal_state": "frozen_p0_release",
        "contract": "worldarena-track1-p0-frozen-release/1",
        "expected_count": expected_count,
        "config": str(config_path),
        "config_sha256": _sha256(config_path),
        "final_decision": str(paths["final_decision"]),
        "final_decision_sha256": p0["final_decision_sha256"],
        "model": str(paths["model"]),
        "model_sha256": p0["model_sha256"],
        "policy": str(paths["policy"]),
        "policy_sha256": p0["policy_sha256"],
        "dataset_receipt": str(dataset_receipt_path),
        "dataset_receipt_sha256": dataset_receipt_hash,
        "dataset_revision": dataset.get("revision"),
        "episode_manifest": str(manifest_path),
        "episode_manifest_sha256": dataset["episode_manifest_sha256"],
        "official_dataset_archive": str(archive_path),
        "official_dataset_archive_sha256": dataset["archive_sha256"],
        "metadata": metadata,
    }
    _atomic_write_json(output_receipt, receipt)
    return receipt


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _validate_contiguous(rows: list[dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows, got {len(rows)}")
    ids = [row.get("episode_id") for row in rows]
    if ids != list(range(1, expected_count + 1)):
        raise ValueError("episode IDs must be unique, ordered, and contiguous from 1")


def build_test1000_plan(
    manifest_path: Path | str,
    output_plan: Path | str,
    *,
    expected_count: int = 1000,
    shard_count: int = 4,
) -> dict[str, Any]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    manifest_path = Path(manifest_path)
    output_plan = Path(output_plan)
    rows = _read_jsonl(manifest_path)
    _validate_contiguous(rows, expected_count)
    planned = [
        {
            "episode_id": row["episode_id"],
            "task": row.get("task"),
            "files": row.get("files", {}),
            "seeds": [1, 4],
            "shard_index": min(
                (index * shard_count) // expected_count, shard_count - 1
            ),
        }
        for index, row in enumerate(rows)
    ]
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in planned
    )
    _atomic_write_text(output_plan, content)
    return {
        "completed": True,
        "contract": "worldarena-track1-test-plan/1",
        "episode_count": expected_count,
        "candidate_video_count": expected_count * 2,
        "seeds": [1, 4],
        "shard_count": shard_count,
        "plan": str(output_plan),
        "plan_sha256": _sha256(output_plan),
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256(manifest_path),
    }


def materialize_test1000_shards(
    *,
    plan_path: Path | str,
    source_root: Path | str,
    output_root: Path | str,
    expected_count: int = 1000,
    shard_count: int = 4,
) -> dict[str, Any]:
    """Materialize immutable, hash-checked symlink inputs for formal generation."""
    plan_path = Path(plan_path).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    output_root = Path(output_root)
    rows = _read_jsonl(plan_path)
    _validate_contiguous(rows, expected_count)
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    shard_rows: list[list[dict[str, Any]]] = [[] for _ in range(shard_count)]
    for row in rows:
        shard_index = row.get("shard_index")
        if not isinstance(shard_index, int) or not 0 <= shard_index < shard_count:
            raise ValueError("plan shard index is outside the frozen shard count")
        files = row.get("files")
        if not isinstance(files, dict) or set(files) != {
            "data",
            "first_frame",
            "instructions",
        }:
            raise ValueError(
                "plan row does not contain the exact formal input file set"
            )
        shard_rows[shard_index].append(row)

    receipts = []
    for shard_index, selected in enumerate(shard_rows):
        shard = output_root / f"shard{shard_index}"
        input_root = shard / "input"
        manifest_rows = []
        for row in selected:
            episode_id = int(row["episode_id"])
            manifest_rows.append(
                {"episode_id": episode_id, "episode_name": f"episode{episode_id}"}
            )
            for file_row in row["files"].values():
                relative = Path(file_row["relative_path"])
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("unsafe formal input relative path")
                source = source_root / relative
                if not source.is_file():
                    raise FileNotFoundError(source)
                if source.stat().st_size != int(file_row["bytes"]):
                    raise ValueError(f"formal input size mismatch: {source}")
                if _sha256(source) != file_row["sha256"]:
                    raise ValueError(f"formal input SHA256 mismatch: {source}")
                target = input_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.is_symlink():
                    if target.resolve() != source.resolve():
                        raise ValueError(f"immutable shard symlink changed: {target}")
                elif target.exists():
                    raise ValueError(f"shard target is not a symlink: {target}")
                else:
                    target.symlink_to(source)
        manifest = shard / "sample-manifest.jsonl"
        content = "".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
            for item in manifest_rows
        )
        if manifest.exists() and manifest.read_text(encoding="utf-8") != content:
            raise ValueError(f"immutable shard manifest changed: {manifest}")
        _atomic_write_text(manifest, content)
        receipts.append(
            {
                "shard_index": shard_index,
                "episode_count": len(selected),
                "first_episode_id": selected[0]["episode_id"] if selected else None,
                "last_episode_id": selected[-1]["episode_id"] if selected else None,
                "input_root": str(input_root),
                "sample_manifest": str(manifest),
                "sample_manifest_sha256": _sha256(manifest),
            }
        )
    if sum(item["episode_count"] for item in receipts) != expected_count:
        raise ValueError("materialized shard count mismatch")
    payload = {
        "completed": True,
        "contract": "worldarena-track1-formal-generation-shards/1",
        "episode_count": expected_count,
        "candidate_video_count": expected_count * 2,
        "shard_count": shard_count,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "source_root": str(source_root),
        "shards": receipts,
    }
    _atomic_write_json(output_root / "shards.complete.json", payload)
    return payload


def render_model_readme(metadata: dict[str, Any]) -> str:
    _validate_metadata(metadata)
    lines = ["---"]
    for key in (
        "model_name",
        "version",
        "organization",
        "release_year",
        "source_type",
        "control_type",
    ):
        lines.append(f"{key}: {json.dumps(metadata[key], ensure_ascii=False)}")
    lines.extend(
        [
            "---",
            "",
            "# WorldArena 2.0 Track 1 submission",
            "",
            "This package was produced by the frozen, receipt-gated release pipeline.",
            "",
        ]
    )
    return "\n".join(lines)


def _finite_unit_feature(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be numeric") from error
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{label} must be finite in [0, 1]")
    return number


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def select_frozen_p0_candidates(
    *,
    model_path: Path | str,
    policy_path: Path | str,
    candidates_path: Path | str,
    output_predictions: Path | str,
    output_receipt: Path | str,
    expected_count: int = 1000,
) -> dict[str, Any]:
    """Apply the frozen generated-only P0 selector to paired candidate features."""
    model_path = Path(model_path).resolve(strict=True)
    policy_path = Path(policy_path).resolve(strict=True)
    candidates_path = Path(candidates_path).resolve(strict=True)
    output_predictions = Path(output_predictions)
    model = _read_json(model_path)
    policy = _read_json(policy_path)
    if (
        policy.get("completed") is not True
        or policy.get("selected_policy") != "postgen_selector"
    ):
        raise ValueError("frozen policy does not select the post-generation selector")
    expected_names = (
        [f"seed1::{column}" for column in _P0_SOURCE_METRICS]
        + [f"seed4::{column}" for column in _P0_SOURCE_METRICS]
        + [f"delta_seed1_minus_seed4::{column}" for column in _P0_SOURCE_METRICS]
    )
    if (
        model.get("feature_policy") != "generated_only_no_hidden_gt"
        or model.get("source_metric_columns") != list(_P0_SOURCE_METRICS)
        or model.get("feature_names") != expected_names
        or model.get("feature_count") != len(expected_names)
    ):
        raise ValueError(
            "frozen P0 selector feature schema differs from the release contract"
        )
    mean = model.get("mean")
    scale = model.get("scale")
    coefficients = model.get("coefficients")
    if (
        not isinstance(mean, list)
        or not isinstance(scale, list)
        or not isinstance(coefficients, list)
        or len(mean) != len(expected_names)
        or len(scale) != len(expected_names)
        or len(coefficients) != len(expected_names) + 1
    ):
        raise ValueError("frozen P0 selector parameter shape mismatch")
    numeric_mean = [float(value) for value in mean]
    numeric_scale = [float(value) for value in scale]
    numeric_coefficients = [float(value) for value in coefficients]
    if any(
        not math.isfinite(value)
        for value in numeric_mean + numeric_scale + numeric_coefficients
    ):
        raise ValueError("frozen P0 selector parameters must be finite")
    if any(value <= 0.0 for value in numeric_scale):
        raise ValueError("frozen P0 selector scales must be positive")

    rows = _read_jsonl(candidates_path)
    _validate_contiguous(rows, expected_count)
    predictions: list[dict[str, Any]] = []
    counts = {"seed1": 0, "seed4": 0}
    for row in rows:
        episode_id = int(row["episode_id"])
        candidate_values: dict[int, list[float]] = {}
        candidate_metadata: dict[int, tuple[Path, str]] = {}
        for seed in (1, 4):
            candidate = row.get(f"seed{seed}")
            if not isinstance(candidate, dict):
                raise ValueError(
                    f"episode {episode_id} seed{seed} candidate is missing"
                )
            features = candidate.get("features")
            if not isinstance(features, dict) or set(features) != set(
                _P0_SOURCE_METRICS
            ):
                raise ValueError(
                    f"episode {episode_id} seed{seed} feature schema mismatch"
                )
            candidate_values[seed] = [
                _finite_unit_feature(
                    features[column], label=f"episode {episode_id} seed{seed} {column}"
                )
                for column in _P0_SOURCE_METRICS
            ]
            video = Path(candidate.get("video", ""))
            video_hash = candidate.get("video_sha256")
            if not isinstance(video_hash, str):
                raise ValueError(
                    f"episode {episode_id} seed{seed} video SHA256 is missing"
                )
            _require_hash(video, video_hash, f"episode {episode_id} seed{seed} video")
            candidate_metadata[seed] = (video, video_hash)
        left, right = candidate_values[1], candidate_values[4]
        vector = left + right + [a - b for a, b in zip(left, right, strict=True)]
        standardized = [
            (value - center) / spread
            for value, center, spread in zip(
                vector, numeric_mean, numeric_scale, strict=True
            )
        ]
        logit = numeric_coefficients[0] + sum(
            coefficient * value
            for coefficient, value in zip(
                numeric_coefficients[1:], standardized, strict=True
            )
        )
        probability = _sigmoid(logit)
        selected_seed = 1 if probability >= 0.5 else 4
        counts[f"seed{selected_seed}"] += 1
        selected_video, selected_hash = candidate_metadata[selected_seed]
        predictions.append(
            {
                "episode_id": episode_id,
                "selected_seed": selected_seed,
                "seed1_probability": probability,
                "selected_video": str(selected_video),
                "selected_video_sha256": selected_hash,
            }
        )
    content = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in predictions
    )
    _atomic_write_text(output_predictions, content)
    receipt = {
        "completed": True,
        "terminal_state": "frozen_p0_candidates_selected",
        "contract": "worldarena-track1-p0-selection/1",
        "row_count": expected_count,
        "selection_counts": counts,
        "feature_policy": "generated_only_no_hidden_gt",
        "deployment_sha_gate": True,
        "model": str(model_path),
        "model_sha256": _sha256(model_path),
        "policy": str(policy_path),
        "policy_sha256": _sha256(policy_path),
        "candidates": str(candidates_path),
        "candidates_sha256": _sha256(candidates_path),
        "predictions": str(output_predictions),
        "predictions_sha256": _sha256(output_predictions),
    }
    _atomic_write_json(Path(output_receipt), receipt)
    return receipt


def _inspect_video(path: Path, frames: int, width: int, height: int) -> None:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"video is not decodable: {path}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise ValueError(f"video dimensions mismatch: {path}")
            if float(frame.mean()) < 1.0 or float(frame.std()) < 0.5:
                raise ValueError(f"black or degenerate frame in video: {path}")
            count += 1
    finally:
        capture.release()
    if count != frames:
        raise ValueError(
            f"video frame count mismatch: expected {frames}, got {count}: {path}"
        )


def stage_selected_videos(
    predictions_path: Path | str,
    staging_dir: Path | str,
    *,
    metadata: dict[str, Any],
    expected_count: int,
    expected_frames: int,
    expected_width: int,
    expected_height: int,
) -> dict[str, Any]:
    _validate_metadata(metadata)
    predictions_path = Path(predictions_path)
    staging_dir = Path(staging_dir)
    rows = _read_jsonl(predictions_path)
    _validate_contiguous(rows, expected_count)
    if staging_dir.exists() and any(staging_dir.iterdir()):
        raise ValueError(f"staging directory is not empty: {staging_dir}")
    videos_dir = staging_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(staging_dir / "model_readme.md", render_model_readme(metadata))
    video_receipts: list[dict[str, Any]] = []
    for row in rows:
        episode_id = row["episode_id"]
        if row.get("selected_seed") not in (1, 4):
            raise ValueError(f"invalid selected seed for episode {episode_id}")
        source = Path(row.get("selected_video", ""))
        expected_hash = row.get("selected_video_sha256")
        if not isinstance(expected_hash, str):
            raise ValueError(f"selected video SHA256 missing for episode {episode_id}")
        _require_hash(source, expected_hash, f"selected video episode {episode_id}")
        _inspect_video(source, expected_frames, expected_width, expected_height)
        destination = videos_dir / f"episode_{episode_id:06d}.mp4"
        shutil.copy2(source, destination)
        staged_hash = _sha256(destination)
        if staged_hash != expected_hash:
            raise ValueError(f"staged video SHA256 mismatch for episode {episode_id}")
        video_receipts.append(
            {
                "episode_id": episode_id,
                "selected_seed": row["selected_seed"],
                "source": str(source),
                "source_sha256": expected_hash,
                "staged": str(destination),
                "staged_sha256": staged_hash,
            }
        )
    receipt = {
        "completed": True,
        "terminal_state": "submission_staged",
        "contract": "worldarena-track1-submission-staging/1",
        "video_count": expected_count,
        "metadata": metadata,
        "predictions": str(predictions_path),
        "predictions_sha256": _sha256(predictions_path),
        "staging_dir": str(staging_dir),
        "deployment_sha_gate": True,
        "videos": video_receipts,
    }
    _atomic_write_json(staging_dir / "submission-staging.complete.json", receipt)
    return receipt


def _tar_info(name: str, *, size: int, directory: bool) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = size
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.mode = 0o755 if directory else 0o644
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mtime = 0
    return info


def build_deterministic_submission_archive(
    staging_dir: Path | str, archive_path: Path | str
) -> dict[str, Any]:
    staging_dir = Path(staging_dir)
    archive_path = Path(archive_path)
    readme = staging_dir / "model_readme.md"
    videos = sorted((staging_dir / "videos").glob("episode_*.mp4"))
    if not readme.is_file() or not videos:
        raise ValueError("staging must contain model_readme.md and videos")
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{archive_path.name}.", dir=archive_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        with temporary_path.open("wb") as raw:
            with (
                gzip.GzipFile(
                    filename="", mode="wb", fileobj=raw, mtime=0
                ) as compressed,
                tarfile.open(fileobj=compressed, mode="w") as archive,
            ):
                data = readme.read_bytes()
                archive.addfile(
                    _tar_info("model_readme.md", size=len(data), directory=False),
                    io.BytesIO(data),
                )
                archive.addfile(_tar_info("videos", size=0, directory=True))
                for video in videos:
                    data = video.read_bytes()
                    archive.addfile(
                        _tar_info(
                            f"videos/{video.name}", size=len(data), directory=False
                        ),
                        io.BytesIO(data),
                    )
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(temporary_path, archive_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return {
        "completed": True,
        "contract": "worldarena-track1-submission-archive/1",
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "video_count": len(videos),
    }


def _safe_member(member: tarfile.TarInfo) -> bool:
    path = PurePosixPath(member.name)
    return (
        not path.is_absolute()
        and ".." not in path.parts
        and not member.issym()
        and not member.islnk()
        and (member.isfile() or member.isdir())
    )


def _parse_front_matter(value: str) -> dict[str, Any]:
    lines = value.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError("model_readme.md is missing YAML front matter")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("model_readme.md front matter is not closed") from error
    metadata: dict[str, Any] = {}
    for line in lines[1:end]:
        key, separator, raw = line.partition(":")
        if not separator:
            raise ValueError("invalid model_readme.md metadata line")
        metadata[key.strip()] = json.loads(raw.strip())
    _validate_metadata(metadata)
    return metadata


def validate_submission_archive_strict(
    archive_path: Path | str, *, expected_count: int
) -> dict[str, Any]:
    archive_path = Path(archive_path)
    expected_names = ["model_readme.md", "videos"] + [
        f"videos/episode_{episode_id:06d}.mp4"
        for episode_id in range(1, expected_count + 1)
    ]
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        if any(not _safe_member(member) for member in members):
            raise ValueError("archive contains unsafe members")
        names = [member.name.rstrip("/") for member in members]
        if names != expected_names:
            raise ValueError("archive members do not match the strict Track 1 layout")
        readme = archive.extractfile(members[0])
        if readme is None:
            raise ValueError("model_readme.md could not be read")
        metadata = _parse_front_matter(readme.read().decode("utf-8"))
    return {
        "completed": True,
        "contract": "worldarena-track1-submission-validation/1",
        "archive": str(archive_path),
        "archive_sha256": _sha256(archive_path),
        "video_count": expected_count,
        "metadata": metadata,
    }


def _require_completed(payload: dict[str, Any], label: str) -> None:
    if payload.get("completed") is not True:
        raise ValueError(f"{label} is not complete")


def _validate_generation_smoke(path: Path, *, expected_seed: int) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("contract") != "flowwam-worldarena-official-stage1-only/1":
        raise ValueError(f"seed{expected_seed} generation receipt contract mismatch")
    if payload.get("stage2_refiner") != "identity-disabled":
        raise ValueError(
            f"seed{expected_seed} generation unexpectedly enables a refiner"
        )
    stage1 = payload.get("stage1")
    if not isinstance(stage1, dict) or stage1.get("seed") != expected_seed:
        raise ValueError(f"seed{expected_seed} generation seed mismatch")
    videos = payload.get("videos")
    if (
        not isinstance(videos, list)
        or len(videos) != 1
        or not isinstance(videos[0], dict)
    ):
        raise ValueError(f"seed{expected_seed} smoke must contain exactly one video")
    video = videos[0]
    video_path = path.parent / "FlowWAMOfficialStage1_test" / str(video.get("name", ""))
    expected_hash = video.get("sha256")
    if not isinstance(expected_hash, str):
        raise ValueError(f"seed{expected_seed} video SHA256 is missing")
    _require_hash(video_path, expected_hash, f"seed{expected_seed} generated video")
    _inspect_video(
        video_path,
        int(stage1.get("num_output_frames", 0)),
        int(stage1.get("width", 0)),
        int(stage1.get("height", 0)),
    )
    return {
        "receipt": str(path),
        "receipt_sha256": _sha256(path),
        "seed": expected_seed,
        "video": str(video_path),
        "video_sha256": expected_hash,
        "checkpoint_sha256": payload.get("checkpoint_sha256"),
        "official_commit": payload.get("official_commit"),
    }


def finalize_release_verification(
    *,
    freeze_receipt: Path | str,
    plan_receipt: Path | str,
    frozen_selector_receipt: Path | str,
    selector_replay_receipt: Path | str,
    seed1_generation_receipt: Path | str,
    seed4_generation_receipt: Path | str,
    package_receipt: Path | str,
    hf_upload_receipt: Path | str,
    anonymous_receipt: Path | str,
    output_receipt: Path | str,
) -> dict[str, Any]:
    """Close the non-submittable Track 1 release smoke with cross-receipt gates."""
    paths = {
        "freeze": Path(freeze_receipt).resolve(strict=True),
        "plan": Path(plan_receipt).resolve(strict=True),
        "frozen_selector": Path(frozen_selector_receipt).resolve(strict=True),
        "selector_replay": Path(selector_replay_receipt).resolve(strict=True),
        "seed1_generation": Path(seed1_generation_receipt).resolve(strict=True),
        "seed4_generation": Path(seed4_generation_receipt).resolve(strict=True),
        "package": Path(package_receipt).resolve(strict=True),
        "hf_upload": Path(hf_upload_receipt).resolve(strict=True),
        "anonymous": Path(anonymous_receipt).resolve(strict=True),
    }
    frozen = _read_json(paths["freeze"])
    plan = _read_json(paths["plan"])
    selector = _read_json(paths["frozen_selector"])
    replay = _read_json(paths["selector_replay"])
    package = _read_json(paths["package"])
    upload = _read_json(paths["hf_upload"])
    anonymous = _read_json(paths["anonymous"])
    for label, payload in (
        ("P0 freeze", frozen),
        ("test-1000 plan", plan),
        ("frozen selector", selector),
        ("selector replay", replay),
        ("submission package", package),
        ("HF upload", upload),
        ("anonymous download", anonymous),
    ):
        _require_completed(payload, label)

    expected_count = frozen.get("expected_count")
    if frozen.get("terminal_state") != "frozen_p0_release" or expected_count != 1000:
        raise ValueError("P0 freeze receipt is not the 1000-episode frozen release")
    if (
        plan.get("episode_count") != expected_count
        or plan.get("candidate_video_count") != expected_count * 2
        or plan.get("seeds") != [1, 4]
        or plan.get("manifest_sha256") != frozen.get("episode_manifest_sha256")
    ):
        raise ValueError("test-1000 plan does not match the frozen release")

    selector_fields = (
        "terminal_state",
        "deployment_sha_gate",
        "row_count",
        "selection_counts",
        "predictions_sha256",
        "video_validation_sha256",
        "input_sha256s",
        "gate",
    )
    if any(selector.get(field) != replay.get(field) for field in selector_fields):
        raise ValueError("selector replay differs from the frozen selector evidence")
    if (
        selector.get("terminal_state") != "pass"
        or selector.get("gate", {}).get("passed") is not True
    ):
        raise ValueError("frozen selector did not pass")
    if replay.get("input_sha256s", {}).get("model") != frozen.get(
        "model_sha256"
    ) or replay.get("input_sha256s", {}).get("policy") != frozen.get("policy_sha256"):
        raise ValueError(
            "selector replay model or policy differs from the frozen release"
        )

    generation = {
        "seed1": _validate_generation_smoke(paths["seed1_generation"], expected_seed=1),
        "seed4": _validate_generation_smoke(paths["seed4_generation"], expected_seed=4),
    }
    if (
        generation["seed1"]["checkpoint_sha256"]
        != generation["seed4"]["checkpoint_sha256"]
    ):
        raise ValueError("dual-seed smoke used different checkpoints")
    if generation["seed1"]["official_commit"] != generation["seed4"]["official_commit"]:
        raise ValueError("dual-seed smoke used different source commits")

    archive_hash = package.get("archive_sha256")
    validation = package.get("validation")
    if (
        not isinstance(archive_hash, str)
        or package.get("video_count") != 1
        or not isinstance(validation, dict)
        or validation.get("completed") is not True
        or validation.get("video_count") != 1
        or validation.get("archive_sha256") != archive_hash
    ):
        raise ValueError("submission smoke package validation failed")
    if (
        upload.get("terminal_state") != "hf_public_smoke_uploaded"
        or upload.get("archive_sha256") != archive_hash
        or upload.get("repo_public") is not True
        or upload.get("repo_gated") is not False
        or upload.get("official_submission_not_triggered") is not True
    ):
        raise ValueError("HF public smoke upload evidence failed")
    anonymous_validation = anonymous.get("archive_validation")
    if (
        anonymous.get("terminal_state") != "hf_public_smoke_anonymously_verified"
        or anonymous.get("download_sha256") != archive_hash
        or anonymous.get("repo_id") != upload.get("repo_id")
        or anonymous.get("revision") != upload.get("commit_sha")
        or anonymous.get("authorization_header_used") is not False
        or anonymous.get("official_submission_not_triggered") is not True
        or not isinstance(anonymous_validation, dict)
        or anonymous_validation.get("archive_sha256") != archive_hash
        or anonymous_validation.get("video_count") != 1
    ):
        raise ValueError("anonymous HF round-trip evidence failed")

    receipt = {
        "completed": True,
        "terminal_state": "release_pipeline_smoke_verified",
        "contract": "worldarena-track1-p0-release-verification/1",
        "ready_for_full_test1000": True,
        "full_test1000_executed": False,
        "official_submission_not_triggered": True,
        "verified_scope": {
            "frozen_p0": True,
            "deterministic_test1000_plan": True,
            "frozen_selector_exact_replay": True,
            "official_input_dual_seed_generation_smoke": True,
            "deterministic_package_smoke": True,
            "hf_public_upload_and_anonymous_download_smoke": True,
        },
        "selector": {
            "predictions_sha256": replay["predictions_sha256"],
            "selection_counts": replay["selection_counts"],
            "gate": replay["gate"],
        },
        "generation": generation,
        "hf": {
            "repo_id": upload["repo_id"],
            "commit_sha": upload["commit_sha"],
            "archive_sha256": archive_hash,
        },
        "evidence": {
            label: {"path": str(path), "sha256": _sha256(path)}
            for label, path in paths.items()
        },
    }
    _atomic_write_json(Path(output_receipt), receipt)
    return receipt


def publish_hf_smoke(
    archive_path: Path | str,
    *,
    repo_id: str,
    api: Any,
    expected_sha256: str,
) -> dict[str, Any]:
    """Publish a non-submittable public smoke artifact and verify repo state."""
    archive_path = Path(archive_path)
    if "smoke" not in repo_id.lower() or "smoke" not in archive_path.name.lower():
        raise ValueError("HF smoke publishing requires smoke in repo and archive names")
    _require_hash(archive_path, expected_sha256, "HF smoke archive")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    commit = api.upload_file(
        path_or_fileobj=str(archive_path),
        path_in_repo=archive_path.name,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload WorldArena Track 1 P0 submission smoke artifact",
    )
    commit_sha = getattr(commit, "oid", None)
    if not isinstance(commit_sha, str) or not commit_sha:
        raise ValueError("Hugging Face upload did not return a commit SHA")
    files = api.list_repo_files(
        repo_id=repo_id, repo_type="dataset", revision=commit_sha
    )
    root_archives = sorted(
        path
        for path in files
        if "/" not in path and (path.endswith(".tar.gz") or path.endswith(".tgz"))
    )
    if root_archives != [archive_path.name]:
        raise ValueError(f"HF repo must have exactly one root archive: {root_archives}")
    info = api.repo_info(repo_id=repo_id, repo_type="dataset", revision=commit_sha)
    if getattr(info, "private", None) is not False:
        raise ValueError("HF smoke repo is not public")
    if getattr(info, "gated", False) not in (False, None, "false"):
        raise ValueError("HF smoke repo is gated")
    return {
        "completed": True,
        "terminal_state": "hf_public_smoke_uploaded",
        "contract": "worldarena-track1-hf-smoke/1",
        "repo_id": repo_id,
        "repo_type": "dataset",
        "repo_public": True,
        "repo_gated": False,
        "commit_sha": commit_sha,
        "archive": str(archive_path),
        "archive_sha256": expected_sha256,
        "remote_filename": archive_path.name,
        "root_archives": root_archives,
        "official_submission_not_triggered": True,
    }
