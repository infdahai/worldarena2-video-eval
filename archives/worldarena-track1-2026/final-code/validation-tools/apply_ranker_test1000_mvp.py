#!/usr/bin/env python3
"""Apply the frozen predicted14 ranker to existing native test1000 videos."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import predicted14_ranker as ranker

SEEDS = (1, 4, 2, 3)
MODEL_SHA = "1409145a4efadfa7d274b2742a85318da6a56322472bc687e3f13844a7eb0fd5"
P0_CANDIDATES_SHA = "64aaeb9aede4f41db16c8f9661eadd59cd8a8907996a385c00594eb317459499"
P0_PREDICTIONS_SHA = "ae70bc87505ee10aad0138651283efe2cb07dcab35b2c65c83e29acc90671101"
GATES = {"ewm_gain_ge_0_3_points", "nonjepa14_mean_gain_gt_0",
         "nonjepa14_one_sided90_lower_gt_0", "jepa_delta_ge_minus_0_005",
         "trajectory_delta_ge_minus_0_005"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def require_go(result):
    gates = result.get("gates")
    if result.get("completed") is not True or result.get("decision") != "GO":
        raise ValueError("fresh16 comparison is not completed GO")
    if not isinstance(gates, dict) or set(gates) != GATES or not all(gates.values()):
        raise ValueError("preregistered gate missing or failed")


def require_package(package, seed):
    if (package.get("completed") is not True
            or package.get("contract") != "worldarena-track1-orb-n8-generated-only-package/1"
            or package.get("seed") != seed or package.get("rows") != 4
            or package.get("hidden_ground_truth_used") is not False
            or len(package.get("videos", [])) != 125):
        raise ValueError("invalid generated-only package receipt")


def format_predictions(decisions, sources):
    rows, details = [], []
    for decision in decisions:
        identity = decision.get("episode_id", "")
        prefix = "test1000:episode"
        if not isinstance(identity, str) or not identity.startswith(prefix):
            raise ValueError("invalid global episode identity")
        suffix = identity[len(prefix):]
        if len(suffix) != 6 or not suffix.isdecimal():
            raise ValueError("invalid global episode identity")
        episode = int(suffix)
        if episode not in sources or decision.get("selected_seed") not in sources[episode]:
            raise ValueError("selected source missing")
        selected = sources[episode][decision["selected_seed"]]
        rows.append({"episode_id": episode, "prior_seed": decision["incumbent_seed"],
                     "selected_seed": decision["selected_seed"],
                     "selected_video": selected["video"],
                     "selected_video_sha256": selected["sha256"],
                     "estimated_delta14": decision["estimated_delta14"]})
        details.append(decision)
    return rows, details


def run(*, model_path, comparison_path, p0_candidates_path, p0_predictions_path,
        extra_scores_root, output_root):
    fixed = {str(model_path): MODEL_SHA, str(p0_candidates_path): P0_CANDIDATES_SHA,
             str(p0_predictions_path): P0_PREDICTIONS_SHA}
    for path, expected in fixed.items():
        if sha(path) != expected:
            raise ValueError(f"frozen input SHA mismatch: {path}")
    comparison_sha = sha(comparison_path)
    comparison = json.loads(comparison_path.read_text())
    require_go(comparison)
    model = ranker.loads_model(model_path.read_text())

    p0_candidates = jsonl(p0_candidates_path)
    p0_predictions = jsonl(p0_predictions_path)
    by_episode = {row["episode_id"]: row for row in p0_candidates}
    incumbents = {row["episode_id"]: row for row in p0_predictions}
    if (len(by_episode) != 1000 or len(incumbents) != 1000
            or set(by_episode) != set(range(1, 1001)) or set(incumbents) != set(by_episode)):
        raise ValueError("P0 coverage must be exact episode1..1000")

    sources, features = {}, {}
    for episode, row in by_episode.items():
        sources[episode], features[episode] = {}, {}
        prediction = incumbents[episode]
        if (prediction["selected_seed"] not in (1, 4)
                or prediction["selected_video_sha256"]
                != row[f"seed{prediction['selected_seed']}"]["video_sha256"]):
            raise ValueError("P0 incumbent differs from frozen candidates")
        for seed in (1, 4):
            item = row[f"seed{seed}"]
            sources[episode][seed] = {"video": item["video"], "sha256": item["video_sha256"]}
            features[episode][seed] = item["features"]

    tracked = {str(comparison_path): comparison_sha, **fixed}
    metrics = set(ranker.FEATURE_SCHEMA)
    for seed in (2, 3):
        seen = set()
        for shard in range(8):
            root = extra_scores_root / f"seed{seed}" / f"shard{shard}" / "package"
            receipt_path = root / "receipts/test1000-generated9.complete.json"
            package_path = root / "receipts/package.complete.json"
            receipt, package = json.loads(receipt_path.read_text()), json.loads(package_path.read_text())
            if (receipt.get("contract") != "worldarena-test1000-generated9-shard-mvp/1"
                    or receipt.get("completed") is not True or receipt.get("hidden_ground_truth_used") is not False
                    or receipt.get("seed") != seed or receipt.get("shard") != shard or receipt.get("rows") != 125):
                raise ValueError("invalid generated9/package receipt")
            require_package(package, seed)
            score_path = Path(receipt["generated9_csv"])
            source_receipt = Path(receipt["source_stage1_receipt"])
            if (score_path != root / "csv_results/generated-only.csv"
                    or sha(score_path) != receipt["generated9_csv_sha256"]
                    or sha(source_receipt) != receipt["source_stage1_receipt_sha256"]):
                raise ValueError("generated9 input SHA/path mismatch")
            tracked.update({str(receipt_path): sha(receipt_path), str(package_path): sha(package_path),
                            str(score_path): sha(score_path), str(source_receipt): sha(source_receipt)})
            mapping = {item["episode_id"]: item for item in package.get("videos", [])}
            if len(mapping) != 125:
                raise ValueError("package episode coverage mismatch")
            with score_path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                if set(reader.fieldnames or []) != {"Video_ID", *metrics}:
                    raise ValueError("generated9 CSV schema mismatch")
                score_rows = list(reader)
            if len(score_rows) != 125:
                raise ValueError("generated9 row count mismatch")
            for score in score_rows:
                name = score["Video_ID"]
                if not name.startswith("fixed_scene_task_episode_") or not name[-6:].isdecimal():
                    raise ValueError("generated9 identity mismatch")
                episode = int(name[-6:])
                item = mapping.get(episode)
                if episode in seen or item is None:
                    raise ValueError("duplicate or missing package episode")
                values = {metric: float(score[metric]) for metric in ranker.FEATURE_SCHEMA}
                if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values.values()):
                    raise ValueError("generated9 value outside [0,1]")
                video = Path(item["source_video"])
                if video.name != f"episode{episode}.mp4" or sha(video) != item["source_sha256"]:
                    raise ValueError("native source video SHA/identity mismatch")
                sources[episode][seed] = {"video": str(video), "sha256": item["source_sha256"]}
                features[episode][seed] = values
                seen.add(episode)
        if len(seen) != 1000:
            raise ValueError("seed coverage must be 1000")

    candidates = []
    frozen_incumbents = {}
    for episode in range(1, 1001):
        identity = f"test1000:episode{episode:06d}"
        frozen_incumbents[identity] = incumbents[episode]["selected_seed"]
        for seed in SEEDS:
            candidates.append({"task": "test1000", "episode_id": identity,
                               "seed": seed, "features": features[episode][seed]})
    decisions = ranker.rank_candidates(model, candidates, frozen_incumbents,
                                       delta_threshold=.003, protected_max_drop=None)
    rows, details = format_predictions(decisions, sources)
    if set(row["episode_id"] for row in rows) != set(range(1, 1001)):
        raise ValueError("output coverage mismatch")
    for path, digest in tracked.items():
        if sha(path) != digest:
            raise ValueError(f"input changed during routing: {path}")
    if output_root.exists():
        raise FileExistsError("refuse to overwrite formal routing output")
    output_root.mkdir(parents=True)
    predictions_path = output_root / "test1000-predicted14-predictions.jsonl"
    details_path = output_root / "test1000-predicted14-details.json"
    predictions_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    details_path.write_text(json.dumps(details, sort_keys=True, allow_nan=False) + "\n")
    result = {"completed": True, "contract": "worldarena-test1000-predicted14-routing-mvp/1",
              "score_kind": "prediction_not_official", "rows": 1000, "candidate_rows": 4000,
              "seeds": list(SEEDS), "delta_threshold": .003, "protected_max_drop": None,
              "hidden_ground_truth_used": False, "real_collection_jepa_used_for_routing": False,
              "comparison_decision": "GO", "comparison_sha256": comparison_sha,
              "model_sha256": MODEL_SHA, "input_sha256": tracked,
              "selection_counts": dict(sorted(Counter(row["selected_seed"] for row in rows).items())),
              "changed_from_p0": sum(row["selected_seed"] != row["prior_seed"] for row in rows),
              "predictions": str(predictions_path), "predictions_sha256": sha(predictions_path),
              "details": str(details_path), "details_sha256": sha(details_path),
              "script_sha256": sha(__file__)}
    receipt_path = output_root / "route.complete.json"
    receipt_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-path", "comparison-path", "p0-candidates-path",
                 "p0-predictions-path", "extra-scores-root", "output-root"):
        p.add_argument("--" + name, required=True, type=Path)
    run(**vars(p.parse_args()))
