"""CPU-only strict seed3 append; never generate, score, package, or upload."""
import argparse
import csv
import fcntl
import json
import math
from collections import Counter
from pathlib import Path

from worldarena_freeze_seed142_mvp import A, R, METRICS, mean9, sha

PRIOR_RECEIPT_SHA = '1368b724cfa831da20268ff1ccd059e50eeab6a6135b01a13cfe2c792b335f74'
PRIOR_PREDICTIONS_SHA = '0e3bed24e98b2c696b37e22bbeb43bb69a72ea0be7e06f0d4017dbe046c08a5f'
HELPER_SHA = '2e35d5c0e3bd9cbb119b8634845d048d7d764d3fe17f626b877e91eadae49c16'
PARENT_SHA = 'e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def append_seed3(prior, score3, video3, sha3):
    seed = prior['selected_seed']
    require(seed in (1, 4, 2), 'incumbent must be a frozen seed1/4/2 choice')
    incumbent = prior[f'seed{seed}_mean9']
    require(all(math.isfinite(v) and 0 <= v <= 1 for v in (incumbent, score3)),
            'invalid generated9 mean')
    row = dict(prior, seed142_selected_seed=seed, seed3_mean9=score3)
    if score3 > incumbent:
        row.update(selected_seed=3, selected_video=video3, selected_video_sha256=sha3)
    return row


def require_complete_inputs(root):
    paths = [root / f'scoring/generated9/seed3/shard{i}/package/receipts/test1000-generated9.complete.json'
             for i in range(8)]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError('seed3 terminal receipts pending: ' + ', '.join(missing))
    for shard, path in enumerate(paths):
        done = json.loads(path.read_text())
        require(done.get('completed') is True and done.get('seed') == 3
                and done.get('shard') == shard and done.get('rows') == 125
                and done.get('hidden_ground_truth_used') is False,
                f'invalid seed3 terminal receipt: {path}')
    return paths


def collect_seed3(root, artifact):
    receipts = require_complete_inputs(root)
    candidates, evidence = {}, []
    for shard, receipt_path in enumerate(receipts):
        done = json.loads(receipt_path.read_text())
        csv_path = receipt_path.parent.parent / 'csv_results/generated-only.csv'
        manifest = artifact / f'submission/releases/p0-dual-seed-parallel-r1/manifests/seed1.gpu{shard}.jsonl'
        rows = [json.loads(line) for line in manifest.read_text().splitlines() if line]
        ids = {int(row['episode_id']) for row in rows}
        require(len(rows) == len(ids) == 125, 'invalid manifest coverage')
        require(done['episodes_sha256'] == sha(manifest), 'manifest SHA mismatch')
        require(done['generated9_csv'] == str(csv_path)
                and done['generated9_csv_sha256'] == sha(csv_path), 'CSV binding mismatch')
        source = root / f'candidates/seed3/shard{shard}'
        source_receipt = source / 'stage1-only.receipt.json'
        require(done['source_stage1_receipt'] == str(source_receipt)
                and done['source_stage1_receipt_sha256'] == sha(source_receipt),
                'source receipt binding mismatch')
        stage = json.loads(source_receipt.read_text())
        require(stage['checkpoint_sha256'] == PARENT_SHA
                and stage['official_commit'] == 'f06fa46042e97738c6619c868f1097be6749d48d'
                and stage['stage1']['seed'] == 3
                and stage['sample_manifest_sha256'] == sha(manifest), 'source lineage mismatch')
        videos = {v['name']: v for v in stage['videos']}
        require(len(stage['videos']) == len(videos) == 125
                and set(videos) == {f'episode{i}.mp4' for i in ids}, 'source membership mismatch')
        with csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            require(len(reader.fieldnames) == 10
                    and set(reader.fieldnames) == {'Video_ID', *METRICS}, 'wrong nine metrics')
            scores = list(reader)
        require(len(scores) == 125, 'wrong CSV row count')
        seen = set()
        for score in scores:
            require(score['Video_ID'].startswith('fixed_scene_task_episode_'), 'invalid video ID')
            episode = int(score['Video_ID'].removeprefix('fixed_scene_task_episode_'))
            require(score['Video_ID'] == f'fixed_scene_task_episode_{episode:06d}'
                    and episode in ids and episode not in seen and episode not in candidates,
                    'duplicate or foreign episode')
            seen.add(episode)
            video = videos[f'episode{episode}.mp4']
            path = source / 'FlowWAMOfficialStage1_test' / video['name']
            require(video['declared_frames'] == video['decoded_frames'] == 121
                    and video['width'] == 640 and video['height'] == 480
                    and video['black_frames'] == 0, 'invalid source video structure')
            require(sha(path) == video['sha256'], 'source video SHA mismatch')
            candidates[episode] = (score, mean9(score), str(path), video['sha256'])
        require(seen == ids, 'incomplete shard')
        evidence.append(dict(receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                             csv=str(csv_path), csv_sha256=sha(csv_path),
                             source_receipt_sha256=sha(source_receipt)))
    require(set(candidates) == set(range(1, 1001)), 'incomplete seed3 test1000')
    return candidates, evidence


def main(check_inputs=False):
    # Missing terminal input fails before any output directory or lock is created.
    require_complete_inputs(R)
    helper = Path(__file__).with_name('worldarena_freeze_seed142_mvp.py')
    require(sha(helper) == HELPER_SHA, 'frozen helper changed')
    prior_root = R / 'routing/seed142-frozen-v1'
    prior_receipt = prior_root / 'seed142-frozen.complete.json'
    prior_predictions = prior_root / 'test1000-seed142-predictions.jsonl'
    require(sha(prior_receipt) == PRIOR_RECEIPT_SHA
            and sha(prior_predictions) == PRIOR_PREDICTIONS_SHA, 'frozen seed142 changed')
    prior = [json.loads(line) for line in prior_predictions.read_text().splitlines() if line]
    require(len(prior) == 1000 and {row['episode_id'] for row in prior} == set(range(1, 1001)),
            'invalid frozen seed142 coverage')
    candidates, evidence = collect_seed3(R, A)
    route = []
    for row in sorted(prior, key=lambda item: item['episode_id']):
        require(sha(Path(row['selected_video'])) == row['selected_video_sha256'],
                'frozen selected video changed')
        _, score, path, video_sha = candidates[row['episode_id']]
        route.append(append_seed3(row, score, path, video_sha))
    counts = dict(Counter(str(row['selected_seed']) for row in route))
    if check_inputs:
        print(json.dumps(dict(inputs_valid=True, rows=1000, selection_counts=counts,
                              output_written=False)))
        return
    output = R / 'routing/seed1423-frozen-v1'
    with output.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        scores_path = output / 'seed3-generated9.csv'
        with scores_path.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Video_ID', *METRICS], lineterminator='\n')
            writer.writeheader()
            writer.writerows(candidates[i][0] for i in sorted(candidates))
        predictions_path = output / 'test1000-seed1423-predictions.jsonl'
        with predictions_path.open('x') as stream:
            for row in route:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        receipt = dict(completed=True, contract='worldarena-test1000-frozen-seed1423-mvp/1',
                       rows=1000, selection_counts=counts,
                       selection_rule='append seed3 only if mean9 > frozen selected seed142 mean9; ties keep',
                       hidden_gt_used_for_selection=False, full15_used_for_selection=False,
                       formal_submission_created=False, prior_receipt_sha256=PRIOR_RECEIPT_SHA,
                       prior_predictions_sha256=PRIOR_PREDICTIONS_SHA,
                       code_sha256=sha(Path(__file__)), helper_sha256=HELPER_SHA,
                       seed3_inputs=evidence, generated9_csv=str(scores_path),
                       generated9_csv_sha256=sha(scores_path), predictions=str(predictions_path),
                       predictions_sha256=sha(predictions_path))
        receipt_path = output / 'seed1423-frozen.complete.json'
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(receipt=str(receipt_path), receipt_sha256=sha(receipt_path),
                          selection_counts=counts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-inputs', action='store_true', help='validate only; write no output')
    args = parser.parse_args()
    try:
        main(check_inputs=args.check_inputs)
    except (FileNotFoundError, ValueError) as exc:
        parser.exit(2, f'{exc}\n')
