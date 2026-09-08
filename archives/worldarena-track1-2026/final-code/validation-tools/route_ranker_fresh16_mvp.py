#!/usr/bin/env python3
"""Frozen P0 incumbent -> four-seed predicted14 ranking; outputs native121.

Requires all 16 completed score receipts. No training, tuning or GT input.
Predicted14 is not official full15: real collection JEPA remains an external gate.
PYTHONPATH must include baseline/src, these tools, and the isolated ranker-fit dir.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import predicted14_ranker as predictor
from score_holdout16_generated9_mvp import atomic_json, read_manifest, sha256

METRICS = ('Instruction Following', 'Interaction Quality', 'Perspectivity',
           'Image Quality', 'Aesthetic Quality', 'Photometric Consistency',
           'Dynamic Degree', 'Flow Score', 'Motion Smoothness')
MODEL_SHA256 = '4e97e87fa645793f885675fc98d6ae460c216395069c0c35a720264886cbf5fa'
POLICY_SHA256 = 'aa551c7b849b03158864c9cb5d2a85f15ddfde4ee0d5c6527f8caf73c86e0af7'
RANKER_MODEL_SHA256 = '1409145a4efadfa7d274b2742a85318da6a56322472bc687e3f13844a7eb0fd5'
SEEDS = (1, 4, 2, 3)


def rank_four_seeds(rows: list[dict], raw: dict, selected: list[dict], fitted: dict) -> list[dict]:
    episodes = {row['episode_id'] for row in rows}
    if set(raw) != set(SEEDS) or any(set(raw[seed]) != episodes for seed in SEEDS):
        raise ValueError('ranker requires exactly four seeds 1/4/2/3 for every episode')
    candidates, incumbents, probabilities = [], {}, {}
    for internal_id, (row, selection) in enumerate(zip(rows, selected, strict=True), 1):
        if selection['episode_id'] != internal_id or selection['selected_seed'] not in (1, 4):
            raise ValueError('frozen P0 selection identity/seed mismatch')
        episode = row['episode_id']
        incumbents[episode] = selection['selected_seed']
        probabilities[episode] = selection['seed1_probability']
        for seed in SEEDS:
            candidates.append(dict(task=row['task'], episode_id=episode, seed=seed,
                                   features=raw[seed][episode]))
    decisions = predictor.rank_candidates(fitted, candidates, incumbents,
                                          delta_threshold=0.003, protected_max_drop=None)
    for decision in decisions:
        decision['p0_seed1_probability'] = probabilities[decision['episode_id']]
    return decisions


def route(*, manifest: Path, score_receipts: list[Path], model: Path,
          policy: Path, ranker_model: Path, output_root: Path) -> dict:
    # First gate: never route with a refitted or substituted ranker model.
    if sha256(ranker_model) != RANKER_MODEL_SHA256:
        raise ValueError('frozen ranker model SHA mismatch')
    fitted = predictor.loads_model(ranker_model.read_text())
    from worldarena_baseline.track1_release import select_frozen_p0_candidates
    rows = read_manifest(manifest)
    if len(rows) != 16 or len(score_receipts) != 16:
        raise ValueError('routing requires the full 16 holdout samples and 16 score receipts')
    if sha256(model) != MODEL_SHA256 or sha256(policy) != POLICY_SHA256:
        raise ValueError('frozen model/policy SHA differs from deployed P0')
    raw = {seed: {} for seed in SEEDS}
    sources = {seed: {} for seed in SEEDS}
    seen_shards, source_csvs = set(), []
    manifest_sha = sha256(manifest)
    for receipt_path in score_receipts:
        receipt = json.loads(receipt_path.read_text())
        seed, shard = receipt.get('seed'), receipt.get('shard')
        if (receipt.get('contract') != 'worldarena-holdout16-generated9-shard-mvp/1'
                or receipt.get('completed') is not True
                or receipt.get('hidden_ground_truth_used') is not False
                or receipt.get('source_frames') != 121 or receipt.get('scoring_frames') != 81
                or receipt.get('rows') != 4 or receipt.get('manifest_sha256') != manifest_sha
                or seed not in SEEDS or shard not in range(4)
                or (seed, shard) in seen_shards):
            raise ValueError('invalid, duplicate or stale generated9 receipt')
        seen_shards.add((seed, shard))
        score_csv = Path(receipt['generated9_csv'])
        if sha256(score_csv) != receipt['generated9_csv_sha256']:
            raise ValueError('generated9 CSV SHA mismatch')
        source_csvs.append(dict(seed=seed, shard=shard, path=str(score_csv), sha256=sha256(score_csv)))
        with score_csv.open(newline='') as stream:
            reader = csv.DictReader(stream)
            if set(reader.fieldnames or []) != {'Video_ID', *METRICS}:
                raise ValueError('generated9 CSV must contain exactly the nine frozen metrics')
            score_rows = list(reader)
        mappings = receipt.get('mapping', [])
        expected = {f'fixed_scene_task_episode_{i+1:06d}': rows[i]
                    for i in range(shard*4, (shard+1)*4)}
        if (len(score_rows) != 4 or {r['Video_ID'] for r in score_rows} != set(expected)
                or len(mappings) != 4 or {r['score_video_id'] for r in mappings} != set(expected)):
            raise ValueError('generated9 identity coverage mismatch')
        for mapping in mappings:
            row = expected[mapping['score_video_id']]
            video = Path(mapping['source_video'])
            if (mapping['episode_id'] != row['episode_id']
                    or mapping['stage1_episode_name'] != row['stage1_episode_name']
                    or video.name != row['stage1_episode_name'] + '.mp4'
                    or mapping.get('source_frames') != 121 or mapping.get('scoring_frames') != 81
                    or mapping['source_sha256'] == mapping.get('score_video_sha256')
                    or sha256(video) != mapping['source_sha256']):
                raise ValueError('native121 source identity/SHA contract failed')
            sources[seed][row['episode_id']] = mapping
        for score in score_rows:
            features = {metric: float(score[metric]) for metric in METRICS}
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in features.values()):
                raise ValueError('generated9 metrics must be finite in [0,1]')
            raw[seed][expected[score['Video_ID']]['episode_id']] = features
    if seen_shards != {(seed, shard) for seed in SEEDS for shard in range(4)}:
        raise ValueError('incomplete four-seed holdout receipts')
    # All read-only checks above precede any output side effect.
    if output_root.exists():
        raise ValueError('routing output must be new; preserve previous evidence')
    output_root.mkdir(parents=True)
    candidates = []
    for internal_id, row in enumerate(rows, 1):
        candidate = {'episode_id': internal_id}
        for seed in (1, 4):
            source = sources[seed][row['episode_id']]
            candidate[f'seed{seed}'] = {'features': raw[seed][row['episode_id']],
                                       'video': source['source_video'],
                                       'video_sha256': source['source_sha256']}
        candidates.append(candidate)
    candidates_path = output_root / 'p0-candidates.internal.jsonl'
    candidates_path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in candidates))
    internal_predictions = output_root / 'p0-selected.internal.jsonl'
    select_frozen_p0_candidates(model_path=model, policy_path=policy,
        candidates_path=candidates_path, output_predictions=internal_predictions,
        output_receipt=output_root / 'p0-selector.complete.json', expected_count=16)
    selected = [json.loads(line) for line in internal_predictions.read_text().splitlines()]
    decisions = rank_four_seeds(rows, raw, selected, fitted)
    decisions_by_episode = {item['episode_id']: item for item in decisions}
    p0, final = [], []
    for row in rows:
        sample = row['episode_id']
        decision = decisions_by_episode[sample]
        for output, seed in ((p0, decision['incumbent_seed']), (final, decision['selected_seed'])):
            source = sources[seed][sample]
            output.append(dict(episode_id=sample, video_path=source['source_video'],
                               video_sha256=source['source_sha256'], seed=seed))
    p0_path = output_root / 'p0-selected16.jsonl'
    final_path = output_root / 'ranker-selected16.jsonl'
    for path, output in ((p0_path, p0), (final_path, final)):
        path.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in output))
    decisions_path = output_root / 'route-decisions.json'
    atomic_json(decisions_path, decisions)
    result = dict(completed=True, contract='worldarena-ranker-fresh16-native-route-mvp/1',
                  rows=16, model_sha256=sha256(model), policy_sha256=sha256(policy),
                  ranker_model_sha256=sha256(ranker_model), ranker_model=str(ranker_model),
                  ranker_module_sha256=sha256(Path(predictor.__file__)),
                  manifest_sha256=manifest_sha, hidden_gt_used_for_selection=False,
                  score_kind='prediction_not_official', delta_threshold=0.003,
                  protected_max_drop=None, real_collection_jepa_full15_gate_required=True,
                  source_frames=121, scoring_frames=81,
                  p0_predictions=str(p0_path), p0_predictions_sha256=sha256(p0_path),
                  ranker_predictions=str(final_path), ranker_predictions_sha256=sha256(final_path),
                  decisions=str(decisions_path), decisions_sha256=sha256(decisions_path),
                  source_csvs=source_csvs,
                  score_receipts=[{'path': str(p), 'sha256': sha256(p)} for p in score_receipts],
                  wrapper_sha256=sha256(Path(__file__)))
    atomic_json(output_root / 'route.complete.json', result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'model', 'policy', 'ranker-model', 'output-root'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--score-receipts', type=Path, nargs='+', required=True)
    print(json.dumps(route(**vars(parser.parse_args())), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
