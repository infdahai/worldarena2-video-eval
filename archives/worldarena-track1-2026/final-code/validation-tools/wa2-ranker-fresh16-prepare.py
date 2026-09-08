"""Freeze a new 16-episode generation-only campaign using existing staging."""
import hashlib
import json
import shutil
from pathlib import Path

import h5py
from worldarena_baseline.adaptive_corpus import materialize_adaptive_shard

A = Path('/data/di/worldarena2_track1_20260815')
E = A / 'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901'
H = E / 'ranker-fresh16-native-20260903'
OLD = E / 'holdout16-nativecap-20260903'
R = A / 'runs/flowwam-official/adaptive-candidate-system'
SOURCE = A / 'datasets/FlowWAM_RoboTwin_extracted'
SALT = 'wa2-ranker-fresh24-20260904:'
TASKS = ['open_microwave', 'press_stapler', 'put_object_cabinet', 'stack_bowls_three']

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()

def rows(p):
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]

def key(r):
    return r['task'], r.get('variant', 'aloha-agilex_clean_50'), int(r['episode_index'])

def write(p, obj):
    with p.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')

assert shutil.disk_usage(A).free >= 120 * (1 << 30), 'must reserve 20GiB above 100GiB floor'
assert not (H / 'manifest.jsonl').exists(), 'new manifest already exists: do not overwrite'
old = rows(OLD / 'manifest.jsonl')
selected = []
for task in TASKS:
    excluded = {int(r['episode_index']) for r in old if r['task'] == task}
    indices = sorted(set(range(10, 50)) - excluded,
                     key=lambda i: hashlib.sha256(f'{SALT}{task}:{i}'.encode()).hexdigest())[:4]
    template = next(r for r in old if r['task'] == task)
    for i in indices:
        r = dict(template)
        rel = Path(task) / 'aloha-agilex_clean_50'
        gt = SOURCE / rel / 'data' / f'episode{i}.hdf5'
        robot = SOURCE / rel / 'robot_only/data' / f'episode{i}.hdf5'
        instruction = SOURCE / rel / 'instructions' / f'episode{i}.json'
        prompt = next(x.strip() for x in json.loads(instruction.read_text())['seen']
                      if isinstance(x, str) and x.strip())
        with h5py.File(gt, 'r') as handle:
            frames = len(handle['observation/head_camera/rgb'])
        assert frames >= 2
        sample = f'{task}__aloha-agilex_clean_50__episode_{i:06d}'
        r.update(episode_id=sample, sample=sample, episode_index=i,
                 gt_frame_count=frames, gt_hdf5=str(gt), hdf5=str(gt.relative_to(SOURCE)),
                 hdf5_sha256=sha(gt), robot_hdf5=str(robot), robot_hdf5_sha256=sha(robot),
                 instruction=str(instruction.relative_to(SOURCE)), instruction_sha256=sha(instruction),
                 prompt=prompt, split='ranker-fresh16-independent-validation', physical_gpu=0,
                 shard_index=len(selected), stage1_episode_name=f'episode{len(selected):06d}')
        selected.append(r)
keys = {key(r) for r in selected}
assert len(keys) == 16
exclusions = [R/'manifests/selector-200.jsonl', R/'expertpool-v2/manifests/train160/train160.jsonl',
              R/'challenger/manifests/fresh40/fresh40.jsonl', A/'runs/v13-scoreboost/data/dev-clean50.jsonl',
              A/'eval/dev-fast-20-v3/dev-fast-20.jsonl', OLD/'manifest.jsonl']
checks = []
for p in exclusions:
    prior = rows(p)
    overlap = keys & {key(r) for r in prior}
    assert not overlap, (str(p), overlap)
    checks.append(dict(path=str(p), sha256=sha(p), rows=len(prior), episode_overlap=0,
                       overlapping_tasks=sorted(set(TASKS) & {r['task'] for r in prior})))
for d in ['jobs', 'claims', 'logs', 'generation']:
    (H/d).mkdir(exist_ok=True)
with (H/'manifest.jsonl').open('x') as f:
    for r in selected:
        f.write(json.dumps(r, sort_keys=True) + '\n')
jobs = []
for task in TASKS:
    for seed in [1, 4, 2, 3]:
        p = H/'jobs'/f'{task}.seed{seed}.jsonl'
        with p.open('x') as f:
            for r in selected:
                if r['task'] == task:
                    f.write(json.dumps(dict(r, seed=seed), sort_keys=True) + '\n')
        jobs.append(dict(task=task, seed=seed, sample_manifest=str(p), sha256=sha(p),
                         expected_count=4, output_dir=str(H/'generation'/f'seed{seed}'/task)))
receipt = materialize_adaptive_shard(manifest_path=H/'manifest.jsonl', source_root=SOURCE,
    output_root=H/'input-staging', physical_gpu=0, expected_count=16)
contract = dict(contract='ranker-fresh16-generation-input/1', completed=True, selection_salt=SALT,
    selection_rule='exclude old native16; SHA256 salt+task+colon+index from 10..49; first four per task',
    manifest_sha256=sha(H/'manifest.jsonl'), episode_disjoint=True, task_disjoint=False,
    official_parent_pretraining_exposure='unknown', no_scores_used=True, old_native16_excluded=True,
    generated9_scoring_authorized_here=False, exclusions=checks, jobs=jobs,
    source_root=str(SOURCE), source_revision=selected[0]['source_revision'],
    max_new_bytes=20*(1<<30), min_free_bytes=100*(1<<30),
    input_receipt_sha256=sha(H/'input-staging/shard-input.complete.json'),
    selected=[dict(task=r['task'], episode_index=r['episode_index'], gt_frames=r['gt_frame_count']) for r in selected])
write(H/'input-contract.json', contract)
print(json.dumps(contract, sort_keys=True))
