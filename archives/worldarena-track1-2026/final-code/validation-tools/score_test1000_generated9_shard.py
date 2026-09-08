#!/usr/bin/env python3
"""MVP runner: score one immutable test1000 candidate shard using generated-only metrics.

The underlying official scorer is reused unchanged.  This wrapper merely binds an
explicit manifest to the scorer's four-row dev adapter; no GT video or metric is
opened or passed to it.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def merge_shard(plan, artifact_root: Path) -> Path:
    """Merge the manifest-sized shard; leave the shared four-row adapter unchanged."""
    from worldarena_baseline import orb_n8_scoring as scorer

    paths = scorer._paths(scorer.OfficialEvalLayout(artifact_root), plan, plan.package_root)
    for phase in ("base", "vlm"):
        receipt = paths.receipts_root / f"generated-only-{phase}.complete.json"
        if not receipt.is_file() or json.loads(receipt.read_text()).get("completed") is not True:
            raise RuntimeError(f"generated-only {phase} is incomplete")
    package = json.loads((paths.receipts_root / "package.complete.json").read_text())
    videos = package.get("videos", [])
    expected = set(plan.episode_ids)
    if len(videos) != len(expected) or {v["episode_id"] for v in videos} != expected:
        raise ValueError("package episode identities differ from manifest")
    for video in videos:
        name = f"fixed_scene_task_episode_{video['episode_id']:06d}.mp4"
        path = paths.primary_videos / name
        if video["score_video"] != name or sha256(path) != video["score_video_sha256"]:
            raise ValueError(f"staged video SHA or name mismatch: {name}")
    output = paths.csv_root / "generated-only.csv"
    scorer.extract_generated_only_score_csv(
        base_json=paths.output_root / "generated_results.json",
        vlm_json=paths.vlm_output_root / paths.model_name / f"{paths.model_name}_summary_val_all_intern.json",
        output_csv=output,
        expected_count=len(plan.episode_ids),
    )
    with output.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row["Video_ID"] for row in rows} != {
        f"fixed_scene_task_episode_{i:06d}" for i in expected
    }:
        raise ValueError("score episode identities differ from manifest")
    return output


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--score-root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--artifact-root", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--ffmpeg", type=Path, required=True)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--merge-only", action="store_true", help="CPU recovery; never stage or score")
    a = p.parse_args()

    rows = [json.loads(line) for line in a.manifest.read_text().splitlines() if line]
    episode_ids = tuple(int(row["episode_id"]) for row in rows)
    if len(episode_ids) != 125 or len(set(episode_ids)) != 125:
        raise ValueError("test1000 shard must contain exactly 125 unique episodes")
    source_receipt = a.source_root / "stage1-only.receipt.json"
    generated = a.source_root / "FlowWAMOfficialStage1_test"
    if not source_receipt.is_file() or not generated.is_dir():
        raise FileNotFoundError("missing immutable Stage1 input")

    # The existing adapter is the pinned generated-only scorer.  It stages aliases
    # to candidate videos, input first frames, and released instruction JSON only.
    import worldarena_baseline.orb_n8_scoring as scorer

    scorer.EPISODE_IDS = episode_ids
    base = scorer.build_orb_n8_package_plan(
        seed=1, source_root=a.source_root, score_root=a.score_root
    )
    plan = dataclasses.replace(
        base,
        seed=a.seed,
        model_name=f"Test1000Seed{a.seed}Shard{a.shard}",
        package_root=a.score_root / f"seed{a.seed}" / f"shard{a.shard}" / "package",
        generated_root=generated,
        receipt_path=source_receipt,
    )
    if not a.merge_only:
        scorer.stage_package(plan=plan, artifact_root=a.artifact_root, dataset=a.dataset, ffmpeg=a.ffmpeg)
        for phase in ("base", "vlm"):
            scorer.run_phase(plan=plan, artifact_root=a.artifact_root, phase=phase, physical_gpu=a.gpu)
    csv_path = merge_shard(plan, a.artifact_root)
    receipt = plan.package_root / "receipts" / "test1000-generated9.complete.json"
    payload = {
        "completed": True,
        "contract": "worldarena-test1000-generated9-shard-mvp/1",
        "seed": a.seed,
        "shard": a.shard,
        "rows": len(episode_ids),
        "episodes_sha256": sha256(a.manifest),
        "source_stage1_receipt": str(source_receipt),
        "source_stage1_receipt_sha256": sha256(source_receipt),
        "generated9_csv": str(csv_path),
        "generated9_csv_sha256": sha256(csv_path),
        "wrapper_sha256": sha256(Path(__file__)),
        "merge_only": a.merge_only,
        "package_video_count_verified": len(episode_ids),
        "legacy_phase_rows_ignored": True,
        "hidden_ground_truth_used": False,
        "inputs": ["candidate_video", "released_first_frame", "released_instruction"],
    }
    tmp = receipt.with_suffix(".json.partial")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, receipt)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
