from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from worldarena_baseline.track1_release import (
    build_deterministic_submission_archive,
    build_test1000_plan,
    finalize_release_verification,
    freeze_p0_release,
    materialize_test1000_shards,
    publish_hf_smoke,
    render_model_readme,
    select_frozen_p0_candidates,
    stage_selected_videos,
    validate_submission_archive_strict,
)
from worldarena_baseline.track1_release_cli import main as release_main


class _Commit:
    oid = "d" * 40


class _RepoInfo:
    private = False
    gated = False
    sha = "d" * 40


class _FakeHfApi:
    def __init__(self) -> None:
        self.uploaded: tuple[str, str] | None = None

    def create_repo(self, **kwargs: object) -> str:
        assert kwargs["repo_type"] == "dataset"
        assert kwargs["private"] is False
        return "https://huggingface.co/datasets/org/repo-smoke"

    def upload_file(self, **kwargs: object) -> _Commit:
        self.uploaded = (str(kwargs["repo_id"]), str(kwargs["path_in_repo"]))
        return _Commit()

    def list_repo_files(self, **kwargs: object) -> list[str]:
        return [".gitattributes", "submission-smoke.tar.gz"]

    def repo_info(self, **kwargs: object) -> _RepoInfo:
        return _RepoInfo()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def _write_video(
    path: Path, *, frames: int = 3, width: int = 16, height: int = 12
) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24.0, (width, height)
    )
    assert writer.isOpened()
    for index in range(frames):
        frame = np.full((height, width, 3), 40 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _p0_fixture(tmp_path: Path) -> Path:
    model = tmp_path / "postgen-selector.model.json"
    policy = tmp_path / "policy-selection.complete.json"
    manifest = tmp_path / "episode-manifest.jsonl"
    archive = tmp_path / "dataset_track1.tar.gz"
    dataset_receipt = tmp_path / "official-track1-dataset.complete.json"
    final = tmp_path / "final-decision.complete.json"

    _write_json(model, {"feature_policy": "generated_only_no_hidden_gt"})
    _write_json(policy, {"completed": True, "selected_policy": "postgen_selector"})
    manifest.write_text(
        json.dumps(
            {
                "episode_id": 1,
                "task": "task-a",
                "files": {
                    "data": {"relative_path": "data/episode1.hdf5", "sha256": "a" * 64},
                    "first_frame": {
                        "relative_path": "first/episode1.png",
                        "sha256": "b" * 64,
                    },
                    "instructions": {
                        "relative_path": "instructions/episode1.json",
                        "sha256": "c" * 64,
                    },
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    archive.write_bytes(b"official-dataset")
    _write_json(
        dataset_receipt,
        {
            "completed": True,
            "episode_count": 1,
            "episode_manifest": str(manifest),
            "episode_manifest_sha256": _sha256(manifest),
            "archive": str(archive),
            "archive_sha256": _sha256(archive),
            "revision": "f" * 40,
            "usage_contract": {
                "final_inference_and_submission_only": True,
                "model_selection_allowed": False,
                "selector_fit_allowed": False,
                "threshold_tuning_allowed": False,
                "training_allowed": False,
            },
        },
    )
    _write_json(
        final,
        {
            "completed": True,
            "terminal_state": "complete_retain_p0",
            "final_winner": "p0_postgen_selector",
            "deployment_sha_gate": True,
            "deployment": {
                "model": str(model),
                "model_sha256": _sha256(model),
                "policy": str(policy),
                "policy_sha256": _sha256(policy),
            },
        },
    )
    config = tmp_path / "release-config.json"
    _write_json(
        config,
        {
            "contract": "worldarena-track1-p0-release-config/1",
            "expected_count": 1,
            "p0": {
                "final_decision": str(final),
                "final_decision_sha256": _sha256(final),
                "model": str(model),
                "model_sha256": _sha256(model),
                "policy": str(policy),
                "policy_sha256": _sha256(policy),
            },
            "dataset_receipt": str(dataset_receipt),
            "dataset_receipt_sha256": _sha256(dataset_receipt),
            "metadata": {
                "model_name": "FlowWAM-P0-PostGen-Selector-Smoke",
                "version": "smoke-r1",
                "organization": "Huazhi AI",
                "release_year": 2026,
                "source_type": "closed_source",
                "control_type": "hybrid",
            },
        },
    )
    return config


def test_freeze_rejects_any_p0_hash_drift(tmp_path: Path) -> None:
    config = _p0_fixture(tmp_path)
    receipt = freeze_p0_release(config, tmp_path / "freeze.complete.json")
    assert receipt["terminal_state"] == "frozen_p0_release"
    assert receipt["expected_count"] == 1

    model = tmp_path / "postgen-selector.model.json"
    model.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model SHA256 mismatch"):
        freeze_p0_release(config, tmp_path / "second.complete.json")


def test_test1000_plan_requires_contiguous_episode_ids(tmp_path: Path) -> None:
    manifest = tmp_path / "episodes.jsonl"
    rows = [{"episode_id": index, "task": "task-a", "files": {}} for index in (1, 2, 4)]
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="contiguous"):
        build_test1000_plan(
            manifest,
            tmp_path / "plan.jsonl",
            expected_count=3,
            shard_count=2,
        )


def test_test1000_plan_is_deterministic_and_paired(tmp_path: Path) -> None:
    manifest = tmp_path / "episodes.jsonl"
    rows = [
        {"episode_id": index, "task": "task-a", "files": {}} for index in range(1, 5)
    ]
    manifest.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    first = build_test1000_plan(
        manifest, tmp_path / "first.jsonl", expected_count=4, shard_count=2
    )
    second = build_test1000_plan(
        manifest, tmp_path / "second.jsonl", expected_count=4, shard_count=2
    )
    assert first["plan_sha256"] == second["plan_sha256"]
    planned = [
        json.loads(line) for line in (tmp_path / "first.jsonl").read_text().splitlines()
    ]
    assert [row["seeds"] for row in planned] == [[1, 4]] * 4
    assert [row["shard_index"] for row in planned] == [0, 0, 1, 1]


def test_materialize_test1000_shards_hash_checks_and_preserves_episode_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    rows = []
    for episode_id in range(1, 5):
        files = {}
        for kind, relative in (
            ("data", f"data/fixed_scene_task/episode{episode_id}.hdf5"),
            ("first_frame", f"first_frame/fixed_scene_task/episode{episode_id}.png"),
            ("instructions", f"instructions/fixed_scene_task/episode{episode_id}.json"),
        ):
            path = source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{kind}-{episode_id}".encode())
            files[kind] = {
                "relative_path": relative,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        rows.append(
            {
                "episode_id": episode_id,
                "task": "fixed_scene_task",
                "files": files,
                "seeds": [1, 4],
                "shard_index": 0 if episode_id <= 2 else 1,
            }
        )
    plan = tmp_path / "plan.jsonl"
    plan.write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = tmp_path / "shards"
    receipt = materialize_test1000_shards(
        plan_path=plan,
        source_root=source,
        output_root=output,
        expected_count=4,
        shard_count=2,
    )
    assert [row["episode_count"] for row in receipt["shards"]] == [2, 2]
    assert (
        output / "shard0/input/first_frame/fixed_scene_task/episode1.png"
    ).is_symlink()
    manifests = [
        [
            json.loads(line)
            for line in (output / f"shard{i}/sample-manifest.jsonl")
            .read_text()
            .splitlines()
        ]
        for i in range(2)
    ]
    assert manifests == [
        [
            {"episode_id": 1, "episode_name": "episode1"},
            {"episode_id": 2, "episode_name": "episode2"},
        ],
        [
            {"episode_id": 3, "episode_name": "episode3"},
            {"episode_id": 4, "episode_name": "episode4"},
        ],
    ]


def test_stage_and_archive_are_hash_gated_and_reproducible(tmp_path: Path) -> None:
    source = tmp_path / "candidate.mp4"
    _write_video(source)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "episode_id": 1,
                "selected_seed": 4,
                "selected_video": str(source),
                "selected_video_sha256": _sha256(source),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    metadata = {
        "model_name": "FlowWAM-P0-PostGen-Selector-Smoke",
        "version": "smoke-r1",
        "organization": "Huazhi AI",
        "release_year": 2026,
        "source_type": "closed_source",
        "control_type": "hybrid",
    }
    staging = tmp_path / "staging"
    staged = stage_selected_videos(
        predictions,
        staging,
        metadata=metadata,
        expected_count=1,
        expected_frames=3,
        expected_width=16,
        expected_height=12,
    )
    assert staged["deployment_sha_gate"] is True
    assert (staging / "videos" / "episode_000001.mp4").is_file()

    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    build_deterministic_submission_archive(staging, first)
    build_deterministic_submission_archive(staging, second)
    assert _sha256(first) == _sha256(second)
    report = validate_submission_archive_strict(first, expected_count=1)
    assert report["video_count"] == 1
    with tarfile.open(first, "r:gz") as handle:
        assert [member.name for member in handle.getmembers()] == [
            "model_readme.md",
            "videos",
            "videos/episode_000001.mp4",
        ]


def test_stage_rejects_selected_video_sha_mismatch(tmp_path: Path) -> None:
    source = tmp_path / "candidate.mp4"
    _write_video(source)
    predictions = tmp_path / "predictions.jsonl"
    predictions.write_text(
        json.dumps(
            {
                "episode_id": 1,
                "selected_seed": 1,
                "selected_video": str(source),
                "selected_video_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        stage_selected_videos(
            predictions,
            tmp_path / "staging",
            metadata={
                "model_name": "x",
                "version": "v1",
                "organization": "org",
                "release_year": 2026,
                "source_type": "closed_source",
                "control_type": "hybrid",
            },
            expected_count=1,
            expected_frames=3,
            expected_width=16,
            expected_height=12,
        )


def test_metadata_uses_official_control_type_values() -> None:
    metadata = {
        "model_name": "x",
        "version": "v1",
        "organization": "org",
        "release_year": 2026,
        "source_type": "closed_source",
        "control_type": "action_driven",
    }
    assert 'control_type: "action_driven"' in render_model_readme(metadata)
    metadata["control_type"] = "action"
    with pytest.raises(ValueError, match="unsupported control_type"):
        render_model_readme(metadata)


def test_hf_smoke_publish_requires_public_smoke_repo_and_one_archive(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "submission-smoke.tar.gz"
    archive.write_bytes(b"archive")
    api = _FakeHfApi()
    receipt = publish_hf_smoke(
        archive,
        repo_id="org/repo-smoke",
        api=api,
        expected_sha256=_sha256(archive),
    )
    assert receipt["commit_sha"] == "d" * 40
    assert receipt["repo_public"] is True
    assert receipt["root_archives"] == ["submission-smoke.tar.gz"]
    assert api.uploaded == ("org/repo-smoke", "submission-smoke.tar.gz")

    with pytest.raises(ValueError, match="smoke"):
        publish_hf_smoke(
            archive,
            repo_id="org/real-submission",
            api=api,
            expected_sha256=_sha256(archive),
        )


def test_release_cli_builds_paired_plan(tmp_path: Path) -> None:
    manifest = tmp_path / "episodes.jsonl"
    manifest.write_text(
        "".join(
            json.dumps({"episode_id": index, "task": "task-a", "files": {}}) + "\n"
            for index in range(1, 5)
        ),
        encoding="utf-8",
    )
    output = tmp_path / "plan.jsonl"
    receipt = tmp_path / "plan.complete.json"
    assert (
        release_main(
            [
                "plan",
                "--manifest",
                str(manifest),
                "--output",
                str(output),
                "--receipt",
                str(receipt),
                "--expected-count",
                "4",
                "--shard-count",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(receipt.read_text())["candidate_video_count"] == 8


def test_finalize_release_verification_closes_the_smoke_chain(tmp_path: Path) -> None:
    video1 = tmp_path / "seed1" / "FlowWAMOfficialStage1_test" / "episode1.mp4"
    video4 = tmp_path / "seed4" / "FlowWAMOfficialStage1_test" / "episode1.mp4"
    video1.parent.mkdir(parents=True)
    video4.parent.mkdir(parents=True)
    _write_video(video1, frames=3, width=16, height=12)
    _write_video(video4, frames=3, width=16, height=12)

    paths = {
        name: tmp_path / f"{name}.json"
        for name in (
            "freeze",
            "plan",
            "selector",
            "replay",
            "seed1",
            "seed4",
            "package",
            "upload",
            "anonymous",
        )
    }
    paths["seed1"] = video1.parents[1] / "stage1-only.receipt.json"
    paths["seed4"] = video4.parents[1] / "stage1-only.receipt.json"
    _write_json(
        paths["freeze"],
        {
            "completed": True,
            "terminal_state": "frozen_p0_release",
            "expected_count": 1000,
            "episode_manifest_sha256": "a" * 64,
            "model_sha256": "b" * 64,
            "policy_sha256": "c" * 64,
        },
    )
    _write_json(
        paths["plan"],
        {
            "completed": True,
            "episode_count": 1000,
            "candidate_video_count": 2000,
            "seeds": [1, 4],
            "manifest_sha256": "a" * 64,
        },
    )
    selector = {
        "completed": True,
        "terminal_state": "pass",
        "deployment_sha_gate": True,
        "row_count": 50,
        "selection_counts": {"seed1": 27, "seed4": 23},
        "predictions_sha256": "d" * 64,
        "video_validation_sha256": "e" * 64,
        "input_sha256s": {"model": "b" * 64, "policy": "c" * 64},
        "gate": {"passed": True, "selected_raw_mean": 0.66},
    }
    _write_json(paths["selector"], selector)
    _write_json(paths["replay"], selector)
    for seed, video, key in ((1, video1, "seed1"), (4, video4, "seed4")):
        _write_json(
            paths[key],
            {
                "contract": "flowwam-worldarena-official-stage1-only/1",
                "stage2_refiner": "identity-disabled",
                "checkpoint_sha256": "f" * 64,
                "official_commit": "1" * 40,
                "stage1": {
                    "seed": seed,
                    "num_output_frames": 3,
                    "width": 16,
                    "height": 12,
                },
                "videos": [{"name": video.name, "sha256": _sha256(video)}],
            },
        )
    archive_sha = "9" * 64
    _write_json(
        paths["package"],
        {
            "completed": True,
            "archive_sha256": archive_sha,
            "video_count": 1,
            "validation": {
                "completed": True,
                "archive_sha256": archive_sha,
                "video_count": 1,
            },
        },
    )
    _write_json(
        paths["upload"],
        {
            "completed": True,
            "terminal_state": "hf_public_smoke_uploaded",
            "archive_sha256": archive_sha,
            "commit_sha": "2" * 40,
            "repo_id": "org/track1-smoke",
            "repo_public": True,
            "repo_gated": False,
            "official_submission_not_triggered": True,
        },
    )
    _write_json(
        paths["anonymous"],
        {
            "completed": True,
            "terminal_state": "hf_public_smoke_anonymously_verified",
            "download_sha256": archive_sha,
            "revision": "2" * 40,
            "repo_id": "org/track1-smoke",
            "authorization_header_used": False,
            "official_submission_not_triggered": True,
            "archive_validation": {
                "completed": True,
                "archive_sha256": archive_sha,
                "video_count": 1,
            },
        },
    )

    output = tmp_path / "track1-release-verification.complete.json"
    receipt = finalize_release_verification(
        freeze_receipt=paths["freeze"],
        plan_receipt=paths["plan"],
        frozen_selector_receipt=paths["selector"],
        selector_replay_receipt=paths["replay"],
        seed1_generation_receipt=paths["seed1"],
        seed4_generation_receipt=paths["seed4"],
        package_receipt=paths["package"],
        hf_upload_receipt=paths["upload"],
        anonymous_receipt=paths["anonymous"],
        output_receipt=output,
    )
    assert receipt["terminal_state"] == "release_pipeline_smoke_verified"
    assert receipt["ready_for_full_test1000"] is True
    assert receipt["full_test1000_executed"] is False
    assert receipt["official_submission_not_triggered"] is True
    assert output.is_file()


def test_select_frozen_p0_candidates_writes_hash_gated_predictions(
    tmp_path: Path,
) -> None:
    columns = [
        "Instruction Following",
        "Interaction Quality",
        "Perspectivity",
        "Image Quality",
        "Aesthetic Quality",
        "Photometric Consistency",
        "Dynamic Degree",
        "Flow Score",
        "Motion Smoothness",
    ]
    feature_names = (
        [f"seed1::{column}" for column in columns]
        + [f"seed4::{column}" for column in columns]
        + [f"delta_seed1_minus_seed4::{column}" for column in columns]
    )
    model = tmp_path / "model.json"
    policy = tmp_path / "policy.json"
    _write_json(
        model,
        {
            "feature_policy": "generated_only_no_hidden_gt",
            "source_metric_columns": columns,
            "feature_names": feature_names,
            "feature_count": 27,
            "mean": [0.0] * 27,
            "scale": [1.0] * 27,
            "coefficients": [-0.5, 1.0] + [0.0] * 26,
        },
    )
    _write_json(policy, {"completed": True, "selected_policy": "postgen_selector"})
    candidates = tmp_path / "candidates.jsonl"
    rows = []
    for episode_id, seed1_instruction in ((1, 1.0), (2, 0.0)):
        seed_rows = {}
        for seed in (1, 4):
            video = tmp_path / f"episode{episode_id}-seed{seed}.mp4"
            video.write_bytes(f"episode{episode_id}-seed{seed}".encode())
            features = {column: 0.5 for column in columns}
            if seed == 1:
                features[columns[0]] = seed1_instruction
            seed_rows[f"seed{seed}"] = {
                "video": str(video),
                "video_sha256": _sha256(video),
                "features": features,
            }
        rows.append({"episode_id": episode_id, **seed_rows})
    candidates.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    predictions = tmp_path / "predictions.jsonl"
    receipt_path = tmp_path / "selection.complete.json"
    receipt = select_frozen_p0_candidates(
        model_path=model,
        policy_path=policy,
        candidates_path=candidates,
        output_predictions=predictions,
        output_receipt=receipt_path,
        expected_count=2,
    )
    selected = [json.loads(line) for line in predictions.read_text().splitlines()]
    assert [row["selected_seed"] for row in selected] == [1, 4]
    assert receipt["selection_counts"] == {"seed1": 1, "seed4": 1}
    assert receipt["deployment_sha_gate"] is True
