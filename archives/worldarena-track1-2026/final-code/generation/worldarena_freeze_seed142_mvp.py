"""CPU-only, write-once seed2 append to the immutable P0 test1000 route."""
import csv
import fcntl
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

A = Path('/data/di/worldarena2_track1_20260815')
R = A / 'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp'
P0 = A / 'submission/releases/p0-dual-seed-parallel-r1/final-p0-r3'
METRICS = ('Instruction Following', 'Interaction Quality', 'Perspectivity',
           'Image Quality', 'Aesthetic Quality', 'Photometric Consistency',
           'Dynamic Degree', 'Flow Score', 'Motion Smoothness')


def choose(incumbent, score1, score4, score2):
    if incumbent not in (1, 4) or any(not math.isfinite(x) or not 0 <= x <= 1
                                     for x in (score1, score4, score2)):
        raise ValueError('invalid incumbent or generated9 mean')
    return 2 if score2 > max(score1, score4) else incumbent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def mean9(values):
    numbers = [float(values[name]) for name in METRICS]
    if any(not math.isfinite(v) or not 0 <= v <= 1 for v in numbers):
        raise ValueError('invalid generated9 feature')
    return sum(numbers) / 9


def main():
    pred = P0 / 'test1000-predictions.jsonl'
    cand = P0 / 'test1000-scored-candidates.jsonl'
    assert sha(pred) == 'ae70bc87505ee10aad0138651283efe2cb07dcab35b2c65c83e29acc90671101'
    assert sha(cand) == '64aaeb9aede4f41db16c8f9661eadd59cd8a8907996a385c00594eb317459499'
    predictions = [json.loads(x) for x in pred.read_text().splitlines() if x]
    candidates = [json.loads(x) for x in cand.read_text().splitlines() if x]
    old = {int(x['episode_id']): x for x in predictions}
    features = {int(x['episode_id']): x for x in candidates}
    expected = set(range(1, 1001))
    assert len(predictions) == len(candidates) == 1000 and set(old) == set(features) == expected
    seed2, evidence = {}, []
    for shard in range(8):
        root = R / f'scoring/generated9/seed2/shard{shard}/package'
        receipt = root / 'receipts/test1000-generated9.complete.json'
        done = json.loads(receipt.read_text())
        csv_path = root / 'csv_results/generated-only.csv'
        assert done['completed'] is True and done['seed'] == 2 and done['shard'] == shard
        assert done['hidden_ground_truth_used'] is False and done['generated9_csv'] == str(csv_path)
        assert done['generated9_csv_sha256'] == sha(csv_path)
        manifest = A / f'submission/releases/p0-dual-seed-parallel-r1/manifests/seed1.gpu{shard}.jsonl'
        ids = {int(json.loads(x)['episode_id']) for x in manifest.read_text().splitlines() if x}
        assert len(ids) == 125 and done['episodes_sha256'] == sha(manifest)
        source = R / f'candidates/seed2/shard{shard}'
        source_path = source / 'stage1-only.receipt.json'
        assert done['source_stage1_receipt_sha256'] == sha(source_path)
        stage1 = json.loads(source_path.read_text())
        assert stage1['stage1']['seed'] == 2
        assert stage1['sample_manifest_sha256'] == sha(manifest)
        assert stage1['checkpoint_sha256'] == 'e211e32b6b79b293f7dec1a70794a69c3c1bf922483c06aef3c5f6d5c3be96c4'
        videos = {v['name']: v for v in stage1['videos']}
        assert len(stage1['videos']) == len(videos) == 125
        with csv_path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            assert set(reader.fieldnames) == {'Video_ID', *METRICS}
            rows = list(reader)
        assert len(rows) == 125
        seen = set()
        for row in rows:
            video_id = row['Video_ID']
            assert video_id.startswith('fixed_scene_task_episode_')
            episode = int(video_id.removeprefix('fixed_scene_task_episode_'))
            assert episode in ids and episode not in seen and episode not in seed2
            seen.add(episode)
            video = videos[f'episode{episode}.mp4']
            path = source / 'FlowWAMOfficialStage1_test' / video['name']
            assert video['declared_frames'] == video['decoded_frames'] == 121
            assert video['width'] == 640 and video['height'] == 480 and video['black_frames'] == 0
            assert sha(path) == video['sha256']
            seed2[episode] = (row, mean9(row), str(path), video['sha256'])
        assert seen == ids
        evidence.append({'receipt': str(receipt), 'receipt_sha256': sha(receipt),
                         'csv': str(csv_path), 'csv_sha256': sha(csv_path)})
    assert set(seed2) == expected
    route = []
    for episode in sorted(expected):
        incumbent = old[episode]
        prior_seed = int(incumbent['selected_seed'])
        f = features[episode]
        prior = f[f'seed{prior_seed}']
        assert incumbent['selected_video'] == prior['video']
        assert incumbent['selected_video_sha256'] == prior['video_sha256']
        assert sha(Path(prior['video'])) == prior['video_sha256']
        score1, score4 = mean9(f['seed1']['features']), mean9(f['seed4']['features'])
        _, score2, path2, sha2 = seed2[episode]
        winner = choose(prior_seed, score1, score4, score2)
        route.append({'episode_id': episode, 'prior_seed': prior_seed, 'selected_seed': winner,
                      'seed1_mean9': score1, 'seed4_mean9': score4, 'seed2_mean9': score2,
                      'selected_video': path2 if winner == 2 else prior['video'],
                      'selected_video_sha256': sha2 if winner == 2 else prior['video_sha256']})
    output = R / 'routing/seed142-frozen-v1'
    output.parent.mkdir(exist_ok=True)
    with (output.parent / 'seed142-frozen-v1.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        output.mkdir(exist_ok=False)
        scores = output / 'seed2-generated9.csv'
        with scores.open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['Video_ID', *METRICS], lineterminator='\n')
            writer.writeheader()
            writer.writerows(seed2[i][0] for i in sorted(expected))
        selection = output / 'test1000-seed142-predictions.jsonl'
        with selection.open('x') as stream:
            for row in route:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
        receipt = {'completed': True, 'contract': 'worldarena-test1000-frozen-seed142-mvp/1',
                   'rows': 1000, 'selection_rule': 'keep P0 incumbent unless seed2_mean9 > max(seed1_mean9, seed4_mean9)',
                   'hidden_gt_used_for_selection': False, 'full15_used_for_selection': False,
                   'seed3_used': False, 'formal_submission_created': False,
                   'selection_counts': dict(Counter(str(row['selected_seed']) for row in route)),
                   'p0_predictions_sha256': sha(pred), 'p0_candidates_sha256': sha(cand),
                   'seed2_inputs': evidence, 'code_sha256': sha(Path(__file__)),
                   'generated9_csv': str(scores), 'generated9_csv_sha256': sha(scores),
                   'predictions': str(selection), 'predictions_sha256': sha(selection)}
        receipt_path = output / 'seed142-frozen.complete.json'
        with receipt_path.open('x') as stream:
            stream.write(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    print(json.dumps({'receipt': str(receipt_path), 'receipt_sha256': sha(receipt_path), **receipt}, sort_keys=True))


if __name__ == '__main__':
    main()
