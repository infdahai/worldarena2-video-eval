#!/usr/bin/env python3
"""One-shot frozen four-seed final ZIP; no upload, deletion, or GPU use."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import runpy
import shutil
import subprocess
import sys
import time
import zipfile

A = Path('/data/di/worldarena2_track1_20260815')
F = A / 'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp/routing/predicted14-alpha10-v1'
OLD = A / 'submission/final-email/p0-dualseed-selector-v1'
OUT = A / 'submission/final-email/predicted14-routing-v1'
VERSION = 'P0-Predicted14-Routing-v1'
WORKERS = 4
GIB = 1024**3


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def make_plan(rows, caps, root, count=1000):
    if len(rows) != count or {r['episode_id'] for r in rows} != set(range(1,count+1)):
        raise ValueError('missing or duplicate episode IDs')
    result = []
    for r in sorted(rows, key=lambda r:r['episode_id']):
        source = Path(r['selected_video']).resolve()
        if not source.is_relative_to(root.resolve()) or source.suffix != '.mp4':
            raise ValueError('source is outside allowed root or not mp4')
        cap = caps[r['episode_id']]
        if cap <= 0 or r['selected_seed'] not in (1,2,3,4) or len(r['selected_video_sha256']) != 64:
            raise ValueError('invalid cap, selected seed, or checksum')
        result.append({**r, 'gt_frame_cap': cap, 'expected_frames': min(121,cap),
                       'needs_trim': cap < 121, 'output_name': f"episode{r['episode_id']}.mp4"})
    return result


def validate_route_receipt(receipt):
    required = {
        'completed': True,
        'comparison_decision': 'GO',
        'hidden_ground_truth_used': False,
        'rows': 1000,
        'candidate_rows': 4000,
        'selection_counts': {'1': 380, '2': 128, '3': 181, '4': 311},
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        raise ValueError('predicted14 routing receipt invalid')


def trim_command(ffmpeg, source, target, frames):
    return [str(ffmpeg), '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
            '-threads', '2', '-i', str(source), '-map', '0:v:0', '-map_metadata', '-1',
            '-an', '-frames:v', str(frames), '-c:v', 'libx264', '-preset', 'medium',
            '-crf', '12', '-pix_fmt', 'yuv420p', '-threads', '2', '-movflags', '+faststart',
            '-fps_mode', 'passthrough', str(target)]


def verify_archive(archive, expected):
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('duplicate ZIP members')
        if set(names) != set(expected):
            raise ValueError('ZIP member set mismatch')
        for name in names:
            p = PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name:
                raise ValueError('unsafe ZIP member')
            h = hashlib.sha256()
            with z.open(name) as f:
                for block in iter(lambda: f.read(1024*1024), b''):
                    h.update(block)
            if h.hexdigest() != expected[name]:
                raise ValueError(f'ZIP member checksum mismatch: {name}')
        if z.testzip() is not None:
            raise ValueError('ZIP CRC failed')
    return len(names)


def disk_guard():
    free = shutil.disk_usage(A).free
    if free < 100*GIB:
        raise RuntimeError(f'disk safety floor breached: {free}')
    used = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) if OUT.exists() else 0
    if used > 5*GIB:
        raise RuntimeError(f'isolated output budget breached: {used}')
    return free, used


def probe(path):
    import cv2
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    if not cap.isOpened():
        raise ValueError(f'cannot open {path}')
    result = {'declared_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
              'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
              'fps': cap.get(cv2.CAP_PROP_FPS), 'decoded_frames': 0, 'black_frames': 0}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        result['decoded_frames'] += 1
        if float(frame.mean()) < 1.0 or float(frame.std()) < 0.5:
            result['black_frames'] += 1
    cap.release()
    return result


def check_probe(result, frames):
    actual = tuple(result[k] for k in ('declared_frames','decoded_frames','width','height','black_frames'))
    if actual != (frames,frames,640,480,0) or result['fps'] <= 0:
        raise ValueError(f'video contract failure: {result}; expected_frames={frames}')


def process(row):
    disk_guard()
    source = Path(row['selected_video'])
    source_sha = sha(source)
    if source_sha != row['selected_video_sha256']:
        raise ValueError(f"frozen source SHA mismatch episode {row['episode_id']}")
    source_probe = probe(source)
    check_probe(source_probe, 121)
    target = OUT / 'staging/HZ-World' / row['output_name']
    command = None
    if row['needs_trim']:
        command = trim_command(A / 'bin/ffmpeg', source, target, row['expected_frames'])
        subprocess.run(command, check=True)
    else:
        with target.open('xb') as fout, source.open('rb') as fin:
            shutil.copyfileobj(fin, fout)
    output_sha = sha(target)
    if not row['needs_trim'] and output_sha != source_sha:
        raise ValueError('byte-copy hash mismatch')
    output_probe = probe(target)
    check_probe(output_probe, row['expected_frames'])
    if abs(source_probe['fps'] - output_probe['fps']) > 0.000001:
        raise ValueError(f'FPS changed: {source_probe} -> {output_probe}')
    if sha(source) != source_sha:
        raise ValueError('source changed while packaging')
    return {**row, 'source_probe': source_probe, 'output_probe': output_probe,
            'sha256': output_sha, 'size_bytes': target.stat().st_size,
            'byte_identical_to_source': output_sha == source_sha,
            'temporal_resampling': False, 'trim_command': command}


def main():
    import cv2
    cv2.setNumThreads(1)
    start = time.monotonic()
    if OUT.exists():
        raise FileExistsError(f'output already exists; refusing overwrite: {OUT}')
    inputs = [
        (F/'test1000-predicted14-predictions.jsonl', 'ea7046831a4837c880e6fdff77f7b38ca646090414cd8c8897ac2ffc883021c3'),
        (F/'route.complete.json', '20d02d7c959a02b8d7ea328a96731fce4e6517fd8cea1a081403e23673469706'),
        (A/'runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903/analysis/full15-final/comparison.complete.json', '585655d738f0fae403282da53c8f7c09d400fa4189d282e74b3f59917a5d72cf'),
        (OLD/'gt-frame-caps.json', '742e17a8de0a1046b0118b7d3966ed5d1004294b1cdbb8357e7a8175fa7c718e'),
        (OLD/'tools/prepare_track1_email_submission.py', 'e7ad465b3c56f29c5955451e639da4af6df92532f51fb84f04740ebd4260073f')]
    for path, expected in inputs:
        if sha(path) != expected:
            raise ValueError(f'input checksum mismatch: {path}')
    frozen = json.loads(inputs[1][0].read_text())
    validate_route_receipt(frozen)
    comparison = json.loads(inputs[2][0].read_text())
    if comparison.get('decision') != 'GO':
        raise ValueError('native full15 holdout gate is not GO')
    rows = [json.loads(line) for line in inputs[0][0].read_text().splitlines()]
    caps = {int(k):int(v) for k,v in json.loads(inputs[3][0].read_text()).items()}
    plan = make_plan(rows, caps, A)
    counts = dict(Counter(r['selected_seed'] for r in plan))
    if counts != {1:380,2:128,3:181,4:311} or sum(r['needs_trim'] for r in plan) != 76:
        raise ValueError('seed or trim counts mismatch')
    audit_path = Path(__file__).with_name('source-audit.json')
    audit = json.loads(audit_path.read_text())
    if {(r['episode_id'],r['expected_frames']) for r in plan if r['needs_trim']} != {(r['episode_id'],r['target_frames']) for r in audit['required_trims']}:
        raise ValueError('trim list differs from independent source audit')
    source_bytes = sum(Path(r['selected_video']).stat().st_size for r in plan)
    trim_bytes = sum(Path(r['selected_video']).stat().st_size for r in plan if r['needs_trim'])
    estimated = 2*(source_bytes + trim_bytes*10) + 64*1024**2
    free, _ = disk_guard()
    if estimated > 5*GIB or free-estimated < 100*GIB:
        raise RuntimeError('insufficient conservative disk budget')
    identity_path = OLD / 'pretest-100-draft/staging/README.md'
    fields = dict(line.split(': ',1) for line in identity_path.read_text().splitlines() if ': ' in line)
    identity = {k:fields[k] for k in ('Model Name','Organization','Responsible Person','Contact Email')}
    if identity['Model Name'] != 'HZ-World':
        raise ValueError('unexpected model identity')
    helper = runpy.run_path(str(inputs[4][0]))
    readme = helper['render_readme'](model_name='HZ-World', version=VERSION,
        organization=identity['Organization'], responsible_person=identity['Responsible Person'],
        contact_email=identity['Contact Email'], video_count=1000)
    OUT.mkdir()
    (OUT/'staging/HZ-World').mkdir(parents=True)
    (OUT/'tools').mkdir()
    shutil.copy2(__file__, OUT/'tools/package.py')
    shutil.copy2(audit_path, OUT/'tools/source-audit.json')
    shutil.copy2(Path(__file__).with_name('test_package.py'), OUT/'tools/test_package.py')
    (OUT/'package.pid').write_text(str(os.getpid())+'\n')
    (OUT/'staging/README.md').write_text(readme)
    print(json.dumps({'status':'started','pid':os.getpid(),'free_bytes':free,'estimated_additional_bytes':estimated,'workers':WORKERS,'source_bytes':source_bytes}), flush=True)
    videos = []
    with (OUT/'progress.jsonl').open('x') as progress:
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            futures = [pool.submit(process,row) for row in plan]
            for future in as_completed(futures):
                video = future.result()
                videos.append(video)
                progress.write(json.dumps(video,sort_keys=True)+'\n')
                progress.flush()
                if len(videos)%25 == 0:
                    print(json.dumps({'completed_videos':len(videos),'elapsed_s':round(time.monotonic()-start)}), flush=True)
    videos.sort(key=lambda r:r['episode_id'])
    archive = OUT/f'HZ-World_{VERSION}_Track1_Submission.zip'
    expected = {'README.md':sha(OUT/'staging/README.md')}
    expected.update({f"HZ-World/{r['output_name']}":r['sha256'] for r in videos})
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_STORED) as z:
        for name in expected:
            z.write(OUT/'staging'/name,name)
    helper['validate_zip_members'](archive, model_name='HZ-World', start=1, end=1000)
    members = verify_archive(archive,expected)
    for path, expected_sha in inputs:
        if sha(path) != expected_sha:
            raise ValueError(f'frozen input changed while packaging: {path}')
    free_after, used = disk_guard()
    payload = {'completed':True, 'completed_at':datetime.now(timezone.utc).isoformat(),
        'contract':'worldarena2-track1-final-email-zip/1', 'model_name':'HZ-World', 'version':VERSION,
        'pid':os.getpid(), 'inputs':[{'path':str(p),'sha256':s} for p,s in inputs],
        'code_path':str(OUT/'tools/package.py'), 'code_sha256':sha(OUT/'tools/package.py'),
        'independent_source_audit_sha256':sha(audit_path), 'identity':identity,
        'identity_source':str(identity_path), 'identity_source_sha256':sha(identity_path),
        'identity_note':'README Responsible Person retained as Disaster; existing pretest email signature was Leihai Nie.',
        'archive_path':str(archive), 'archive_sha256':sha(archive), 'archive_size_bytes':archive.stat().st_size,
        'archive_member_count':members, 'zip_crc_pass':True, 'zip_member_sha_matches_staging':True,
        'video_count':1000, 'trimmed_count':76, 'byte_copied_count':924, 'seed_counts':counts,
        'all_sources_fresh_sha_and_decode_pass':True, 'all_outputs_fresh_decode_pass':True,
        'all_native_fps_preserved':True, 'temporal_resampling':False, 'black_frames':0,
        'full15_used_for_selection':False, 'hidden_gt_used_for_selection':False,
        'gt_used_only_for_output_length_cap':True, 'workers':WORKERS,'ffmpeg_threads_per_process':2,
        'ffmpeg_version':subprocess.check_output([str(A/'bin/ffmpeg'),'-version'],text=True).splitlines()[0],
        'estimated_additional_bytes':estimated,'output_total_bytes':used, 'free_after_bytes':free_after,
        'uploaded':False,'email_sent':False,'unfinished_flags':{'uploaded':False,'email_sent':False},
        'holdout_nativecap_full15_gate':'GO',
        'holdout_nativecap_full15_comparison_sha256':'585655d738f0fae403282da53c8f7c09d400fa4189d282e74b3f59917a5d72cf',
        'elapsed_s':round(time.monotonic()-start,2), 'videos':videos}
    with (OUT/'submission.complete.json').open('x') as f:
        json.dump(payload,f,indent=2,sort_keys=True)
        f.write('\n')
    print(json.dumps({k:v for k,v in payload.items() if k!='videos'},sort_keys=True),flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        if OUT.exists() and not (OUT/'submission.complete.json').exists():
            with (OUT/f'failure-{os.getpid()}.json').open('x') as f:
                json.dump({'completed':False,'error':repr(error),'uploaded':False,'email_sent':False},f)
        raise
