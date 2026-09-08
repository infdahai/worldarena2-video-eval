#!/usr/bin/env python3
"""Four-row holdout adapter around the frozen test1000 generated9 pipeline.

Set PYTHONPATH to the existing baseline/src and tools directories. Pass the full
16-row manifest, --shard 0..3, and that task's generation --source-root. The only
scoring inputs are candidate video, released first frame and instruction JSON.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    os.replace(temporary, path)


def read_manifest(path: Path) -> list[dict]:
    # Deliberate allow-list: no GT path, GT frame count, action or label field.
    fields = ('episode_id', 'stage1_episode_name', 'task', 'prompt')
    rows = [{key: row[key] for key in fields} for row in
            (json.loads(line) for line in path.read_text().splitlines() if line.strip())]
    if len(rows) not in (4, 16) or len({r['episode_id'] for r in rows}) != len(rows):
        raise ValueError('manifest must have 4 or 16 unique holdout samples')
    for i, row in enumerate(rows):
        if not isinstance(row['episode_id'], str) or not row['episode_id']:
            raise ValueError('holdout episode_id must be a nonempty string')
        if row['stage1_episode_name'] != f'episode{i:06d}':
            raise ValueError('manifest must retain ordered zero-based Stage1 names')
        if not isinstance(row['prompt'], str) or not row['prompt'].strip():
            raise ValueError('manifest prompt is missing')
    return rows


def make_plan(*, seed: int, shard: int, source_root: Path,
              score_root: Path, manifest: Path):
    from worldarena_baseline.orb_n8_scoring import OrbN8PackagePlan
    rows = read_manifest(manifest)
    if seed not in (1, 4, 2, 3) or shard not in range(len(rows) // 4):
        raise ValueError('invalid seed or four-row shard')
    selected = rows[shard*4:(shard+1)*4]
    if len({row['task'] for row in selected}) != 1:
        raise ValueError('each generation shard must contain one task')
    return OrbN8PackagePlan(
        seed=seed, episode_ids=tuple(range(shard*4+1, shard*4+5)),
        model_name=f'Holdout16Seed{seed}Shard{shard}',
        source_root=source_root.resolve(), score_root=score_root.resolve(),
        package_root=score_root.resolve() / f'seed{seed}' / f'shard{shard}' / 'package',
        generated_root=source_root.resolve() / 'FlowWAMOfficialStage1_test',
        receipt_path=source_root.resolve() / 'stage1-only.receipt.json')


def stage_package(*, plan, manifest: Path, artifact_root: Path,
                  dataset: Path, ffmpeg: Path):
    from worldarena_baseline import orb_n8_scoring as scorer
    rows = read_manifest(manifest)
    selected = [rows[i-1] for i in plan.episode_ids]
    source_receipt = json.loads(plan.receipt_path.read_text())
    videos = source_receipt.get('videos', [])
    names = {row['stage1_episode_name'] + '.mp4' for row in selected}
    if (source_receipt.get('contract') != 'flowwam-worldarena-official-stage1-only/1'
            or source_receipt.get('stage1', {}).get('seed') != plan.seed
            or len(videos) != 4 or {v['name'] for v in videos} != names):
        raise ValueError('Stage1 receipt seed/video set differs from holdout shard')
    source_by_name = {v['name']: v for v in videos}
    alias_root = plan.package_root.parent / 'stage-input'
    alias_videos = alias_root / 'videos'
    alias_dataset = alias_root / 'input'
    alias_receipt = alias_root / 'stage1-only.receipt.json'
    metadata_path = alias_root / 'inputs.json'
    identity = {'manifest_sha256': sha256(manifest),
                'source_receipt_sha256': sha256(plan.receipt_path),
                'dataset': str(dataset.resolve()), 'source_root': str(plan.source_root)}
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != identity:
        raise ValueError('staged holdout source identity changed')
    alias_videos.mkdir(parents=True, exist_ok=True)
    aliases, mapping = [], []
    for internal_id, row in zip(plan.episode_ids, selected, strict=True):
        name = row['stage1_episode_name']
        source = plan.generated_root / (name + '.mp4')
        receipt_video = source_by_name[name + '.mp4']
        if sha256(source) != receipt_video['sha256']:
            raise ValueError('original native121 source SHA changed')
        first = dataset.resolve() / 'first_frame/fixed_scene_task' / (name + '.png')
        instruction = dataset.resolve() / 'instructions/fixed_scene_task' / (name + '.json')
        if json.loads(instruction.read_text()).get('instruction') != row['prompt']:
            raise ValueError('released instruction differs from holdout manifest prompt')
        links = [(alias_videos / f'episode{internal_id}.mp4', source),
                 (alias_dataset / 'first_frame/fixed_scene_task' / f'episode{internal_id}.png', first),
                 (alias_dataset / 'instructions/fixed_scene_task' / f'episode{internal_id}.json', instruction)]
        for link, target in links:
            if not target.is_file():
                raise FileNotFoundError(target)
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.is_symlink():
                if link.resolve() != target.resolve():
                    raise ValueError(f'input alias changed: {link}')
            elif link.exists():
                raise ValueError(f'input alias is not a symlink: {link}')
            else:
                link.symlink_to(target)
        aliases.append(dict(receipt_video, name=f'episode{internal_id}.mp4'))
        mapping.append(dict(episode_id=row['episode_id'], stage1_episode_name=name,
                            score_video_id=f'fixed_scene_task_episode_{internal_id:06d}',
                            source_video=str(source), source_sha256=receipt_video['sha256'],
                            source_frames=121, scoring_frames=81))
    atomic_json(alias_receipt, dict(source_receipt, videos=aliases))
    atomic_json(metadata_path, identity)
    # EXACT existing finaltest1000 preprocessing, including _resample_generated.
    alias_plan = dataclasses.replace(plan, generated_root=alias_videos, receipt_path=alias_receipt)
    paths = scorer.stage_package(plan=alias_plan, artifact_root=artifact_root,
                                 dataset=alias_dataset, ffmpeg=ffmpeg)
    package_path = paths.receipts_root / 'package.complete.json'
    package = json.loads(package_path.read_text())
    if [v['episode_id'] for v in package['videos']] != list(plan.episode_ids):
        raise ValueError('existing scoring package IDs differ from holdout shard')
    for video, original in zip(package['videos'], mapping, strict=True):
        if sha256(paths.primary_videos / video['score_video']) != video['score_video_sha256']:
            raise ValueError('staged 81-frame scoring video SHA changed')
        original['score_video_sha256'] = video['score_video_sha256']
        video['source_video'] = original['source_video']
        video['source_sha256'] = original['source_sha256']
    package.update(source_frames=121, scoring_frames=81, mapping=mapping,
                   manifest_sha256=identity['manifest_sha256'])
    atomic_json(package_path, package)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source-root', 'score-root', 'manifest', 'artifact-root', 'dataset', 'ffmpeg'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('seed', 'shard'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--merge-only', action='store_true')
    args = parser.parse_args()
    if not args.merge_only and args.gpu is None:
        parser.error('--gpu is required unless --merge-only')
    from worldarena_baseline import orb_n8_scoring as scorer
    from score_test1000_generated9_shard import merge_shard
    plan = make_plan(seed=args.seed, shard=args.shard, source_root=args.source_root,
                     score_root=args.score_root, manifest=args.manifest)
    if not args.merge_only:
        stage_package(plan=plan, manifest=args.manifest, artifact_root=args.artifact_root,
                      dataset=args.dataset, ffmpeg=args.ffmpeg)
        for phase in ('base', 'vlm'):
            scorer.run_phase(plan=plan, artifact_root=args.artifact_root,
                             phase=phase, physical_gpu=args.gpu)
    package = json.loads((plan.package_root / 'receipts/package.complete.json').read_text())
    if package.get('manifest_sha256') != sha256(args.manifest):
        raise ValueError('package manifest SHA changed')
    for mapping in package['mapping']:
        if sha256(Path(mapping['source_video'])) != mapping['source_sha256']:
            raise ValueError('native121 source SHA changed before merge')
    score_csv = merge_shard(plan, args.artifact_root)
    receipt = dict(completed=True, contract='worldarena-holdout16-generated9-shard-mvp/1',
                   seed=args.seed, shard=args.shard, rows=4,
                   manifest_sha256=sha256(args.manifest),
                   source_stage1_receipt=str(plan.receipt_path),
                   source_stage1_receipt_sha256=sha256(plan.receipt_path),
                   source_frames=121, scoring_frames=81, mapping=package['mapping'],
                   generated9_csv=str(score_csv), generated9_csv_sha256=sha256(score_csv),
                   hidden_ground_truth_used=False,
                   inputs=['candidate_video', 'released_first_frame', 'released_instruction'],
                   wrapper_sha256=sha256(Path(__file__)))
    atomic_json(plan.package_root / 'receipts/holdout16-generated9.complete.json', receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
