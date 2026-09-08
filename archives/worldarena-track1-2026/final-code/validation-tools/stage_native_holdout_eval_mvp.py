"""CPU-only staging of frozen native121 predictions with full, real holdout GT.

This does not select candidates, run metrics, or establish holdout independence.
The caller must freeze/verify membership and selection before invoking it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import h5py
import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def inspect_video(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'cannot open video: {path}')
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    declared_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = 0
    shape = None
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if shape is not None and frame.shape != shape:
                raise ValueError(f'variable video dimensions: {path}')
            shape = frame.shape
            frames += 1
    finally:
        capture.release()
    if frames < 1 or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f'empty video or invalid FPS: {path}')
    return dict(frames=frames, declared_frames=declared_frames,
                fps=fps, width=shape[1], height=shape[0])


def _rows(path):
    text = Path(path).read_text()
    rows = json.loads(text) if text.lstrip().startswith('[') else [
        json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'expected a JSON list or JSONL objects: {path}')
    return rows


def _key(row):
    value = row.get('episode_id')
    if value is None or isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('each manifest/prediction row requires a string or integer episode_id')
    if not str(value):
        raise ValueError('empty episode_id')
    return str(value)


def _write_json(path, value):
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _codec_args():
    return ['-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '12',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart']


def _encode_gt(ffmpeg, hdf5_path, output, image):
    with tempfile.TemporaryDirectory(prefix='gt-frames-', dir=output.parent) as directory:
        frame_root = Path(directory)
        with h5py.File(hdf5_path, 'r') as handle:
            encoded = handle['observation/head_camera/rgb']
            for index in range(len(encoded)):
                payload = bytes(encoded[index])
                frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError(f'undecodable GT frame {index}: {hdf5_path}')
                (frame_root / f'frame_{index:06d}.jpg').write_bytes(payload)
                if index == 0:
                    first = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
                    if not cv2.imwrite(str(image), first):
                        raise RuntimeError(f'cannot write first GT frame: {image}')
        subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
                        '-framerate', '24', '-start_number', '0', '-i',
                        str(frame_root / 'frame_%06d.jpg'), '-vf', 'scale=640:480',
                        *_codec_args(), str(output)], check=True)


def stage_package(*, manifest, predictions, output_root, model_name, ffmpeg,
                  expected_count=16):
    manifest = Path(manifest).resolve(strict=True)
    predictions = Path(predictions).resolve(strict=True)
    output_root = Path(output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f'refuse existing output: {output_root}')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', model_name):
        raise ValueError('invalid model_name')
    ffmpeg = shutil.which(str(ffmpeg))
    if ffmpeg is None:
        raise FileNotFoundError('ffmpeg executable is missing')
    rows, selected = _rows(manifest), _rows(predictions)
    keys, selected_keys = [_key(row) for row in rows], [_key(row) for row in selected]
    if expected_count < 1 or len(rows) != expected_count or len(set(keys)) != len(keys):
        raise ValueError('manifest count or unique episode_id coverage differs')
    if len(selected_keys) != len(set(selected_keys)) or set(keys) != set(selected_keys):
        raise ValueError('prediction keys must exactly match unique manifest episode_id values')
    by_key = dict(zip(selected_keys, selected))
    prepared = []
    names = set()
    # Validate all membership/source SHA bindings before creating any package output.
    for row in rows:
        key, chosen = _key(row), by_key[_key(row)]
        task = str(row['task'])
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', task):
            raise ValueError(f'invalid task name: {task}')
        episode = int(row['episode_index'])
        if episode < 0:
            raise ValueError('negative episode_index')
        name = f'{task}_episode_{episode:06d}.mp4'
        if name in names:
            raise ValueError(f'duplicate canonical task/episode: {name}')
        names.add(name)
        prompt = row['prompt']
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('prompt must be a nonempty string')
        raw_video = chosen.get('video_path') or chosen.get('selected_video')
        if not raw_video:
            raise ValueError(f'prediction has no video_path/selected_video: {key}')
        source = Path(raw_video).resolve(strict=True)
        source_sha = sha256(source)
        expected_sha = chosen.get('selected_video_sha256') or chosen.get('video_sha256')
        if expected_sha is not None and source_sha != expected_sha:
            raise ValueError(f'selected source SHA mismatch: {key}')
        info = inspect_video(source)
        if (info['frames'], info['declared_frames'], info['width'], info['height']) != (121, 121, 640, 480):
            raise ValueError(f'expected native121 640x480 source: {source}')
        gt = Path(row['gt_hdf5']).resolve(strict=True)
        gt_sha = sha256(gt)
        if row.get('hdf5_sha256') is not None and row['hdf5_sha256'] != gt_sha:
            raise ValueError(f'manifest GT HDF5 SHA mismatch: {gt}')
        with h5py.File(gt, 'r') as handle:
            gt_frames = len(handle['observation/head_camera/rgb'])
        if gt_frames < 2:
            raise ValueError(f'full15 needs at least two real GT frames: {gt}')
        prepared.append(dict(key=key, row=row, name=name, source=source,
                             source_sha=source_sha, source_info=info,
                             gt=gt, gt_sha=gt_sha, gt_frames=gt_frames))
    input_hashes = dict(manifest_sha256=sha256(manifest),
                        predictions_sha256=sha256(predictions))
    primary = output_root / 'videos' / f'{model_name}_test'
    gt_flat = output_root / 'videos' / 'gt_test'
    output_root.mkdir(parents=True, exist_ok=False)
    primary.mkdir(parents=True)
    gt_flat.mkdir()
    (output_root / 'receipts').mkdir()
    with (output_root / 'predictions.snapshot.jsonl').open('x') as stream:
        for item in prepared:
            # Snapshot is selection-only; never add GT paths/counts/metrics or a new choice.
            stream.write(json.dumps(dict(episode_id=item['key'],
                video_path=str(item['source']), video_sha256=item['source_sha']),
                sort_keys=True) + '\n')
    summaries, records = [], []
    for item in prepared:
        row, source = item['row'], item['source']
        generated = primary / item['name']
        count = min(121, item['gt_frames'])
        trimmed = count < 121
        if trimmed:
            # No -r, fps filter, timestamp reset or uniform sampling: only keep the prefix.
            subprocess.run([str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-n',
                            '-i', str(source), '-frames:v', str(count),
                            '-fps_mode', 'passthrough', '-map_metadata', '-1',
                            *_codec_args(), str(generated)],
                           check=True)
        else:
            shutil.copyfile(source, generated)
        generated_info = inspect_video(generated)
        generated_sha = sha256(generated)
        if (generated_info['frames'] != count or generated_info['declared_frames'] != count or
                (generated_info['width'], generated_info['height']) != (640, 480) or
                abs(generated_info['fps'] - item['source_info']['fps']) > 0.01):
            raise RuntimeError(f'generated frame-count/dimensions/FPS drift: {generated}')
        if not trimmed and generated_sha != item['source_sha']:
            raise RuntimeError(f'uncapped source bytes changed: {source}')
        if sha256(source) != item['source_sha']:
            raise RuntimeError(f'source changed during staging: {source}')
        gt_source = (output_root / 'gt_source' / row['task'] / 'a' / 'b' / 'c' /
                     f"episode_{int(row['episode_index']):06d}.mp4")
        gt_source.parent.mkdir(parents=True, exist_ok=True)
        image = gt_source.with_suffix('.png')
        _encode_gt(ffmpeg, item['gt'], gt_source, image)
        gt_info = inspect_video(gt_source)
        if gt_info != dict(frames=item['gt_frames'], declared_frames=item['gt_frames'],
                           fps=24.0, width=640, height=480):
            raise RuntimeError(f'GT frame-count/FPS/dimensions drift: {gt_source}')
        shutil.copyfile(gt_source, gt_flat / item['name'])
        summaries.append(dict(gt_path=str(gt_source), image=str(image), prompt=[row['prompt']]))
        records.append(dict(episode_id=item['key'], task=row['task'],
            episode_index=int(row['episode_index']), name=item['name'], trimmed=trimmed,
            frame_cap=item['gt_frames'], source=dict(path=str(source),
                sha256=item['source_sha'], **item['source_info']),
            generated=dict(path=str(generated), sha256=generated_sha, **generated_info),
            gt_hdf5=str(item['gt']), gt_hdf5_sha256=item['gt_sha'],
            gt=dict(path=str(gt_source), sha256=sha256(gt_source),
                                          **gt_info), image_sha256=sha256(image)))
    if input_hashes != dict(manifest_sha256=sha256(manifest),
                            predictions_sha256=sha256(predictions)):
        raise RuntimeError('manifest or frozen predictions changed during staging')
    _write_json(output_root / 'summary.json', summaries)
    receipt = dict(completed=True, contract='worldarena-native121-holdout-package-mvp/1',
        model_name=model_name, rows=len(records), real_gt_staged=True,
        candidate_selection_performed=False, gt_used_for_frame_cap=True,
        generated_temporal_resampling=False, gt_temporal_resampling=False,
        black_frames_checked=False,
        source_frames=121, gt_encode_fps=24, trim_policy='prefix min(121, full_gt_frame_count)',
        manifest=str(manifest), predictions=str(predictions), **input_hashes,
        code_sha256=sha256(Path(__file__)), summary_sha256=sha256(output_root / 'summary.json'),
        predictions_snapshot_sha256=sha256(output_root / 'predictions.snapshot.jsonl'),
        videos=records)
    _write_json(output_root / 'receipts' / 'package.complete.json', receipt)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--model-name', required=True)
    parser.add_argument('--ffmpeg', default='ffmpeg')
    parser.add_argument('--expected-count', type=int, default=16)
    args = parser.parse_args()
    result = stage_package(**vars(args))
    print(json.dumps(dict(completed=result['completed'], rows=result['rows'],
        receipt=str(args.output_root.resolve() / 'receipts/package.complete.json'))))
