"""Publish the validated predicted14 ZIP and verify an anonymous immutable download."""
from pathlib import Path
import hashlib
import io
import json
import os
import shlex
import zipfile
import fcntl
import requests
from huggingface_hub import HfApi

A = Path('/data/di/worldarena2_track1_20260815')
ROOT = A / 'submission/final-email/predicted14-routing-v1'
REPO = 'clusternlh/worldarena-track1-hz-wam'
DEST = 'final-1000/P0-Predicted14-Routing-v1/submission.zip'
lock = (ROOT / 'publish.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
receipt = json.loads((ROOT / 'submission.complete.json').read_text())
assert receipt['completed'] and receipt['version'] == 'P0-Predicted14-Routing-v1'
assert receipt['video_count'] == 1000 and receipt['trimmed_count'] == 76
assert receipt['zip_crc_pass'] and receipt['all_native_fps_preserved']
assert receipt['holdout_nativecap_full15_gate'] == 'GO'
archive = Path(receipt['archive_path'])
assert archive.parent == ROOT
assert archive.stat().st_size == receipt['archive_size_bytes']
expected_sha = receipt['archive_sha256']
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected_sha
upload_receipt = ROOT / 'hf-upload.complete.json'
anonymous_receipt = ROOT / 'hf-anonymous.complete.json'
if anonymous_receipt.exists():
    verified = json.loads(anonymous_receipt.read_text())
    assert verified['sha256'] == expected_sha and verified['filename'] == DEST
    print(json.dumps(verified), flush=True)
    raise SystemExit(0)
values = []
for line in (A / 'submission/project-hf.env').read_text().splitlines():
    line = line.removeprefix('export ').strip()
    if line.startswith('HF_TOKEN_PATH='):
        values.append(shlex.split(line.split('=', 1)[1])[0])
assert len(values) == 1
token_path = Path(values[0])
assert token_path.is_absolute()
api = HfApi(token=token_path.read_text().strip())
if upload_receipt.exists():
    uploaded = json.loads(upload_receipt.read_text())
    assert uploaded['repo_id'] == REPO and uploaded['filename'] == DEST
    assert uploaded['sha256'] == expected_sha
    revision = uploaded['revision']
else:
    info = api.repo_info(REPO, repo_type='dataset')
    assert not info.private
    if DEST in api.list_repo_files(REPO, repo_type='dataset', revision=info.sha):
        revision = info.sha  # Never overwrite: anonymous SHA below must match.
    else:
        print('Uploading validated final package; no email is sent.', flush=True)
        revision = api.upload_file(path_or_fileobj=str(archive), path_in_repo=DEST,
            repo_id=REPO, repo_type='dataset', parent_commit=info.sha,
            commit_message='Add HZ-World predicted14 routed final ZIP; preserve previous submissions').oid
        uploaded = dict(completed=True, repo_id=REPO, filename=DEST, revision=revision,
                        sha256=expected_sha, bytes=archive.stat().st_size, email_sent=False)
        with upload_receipt.open('x') as stream:
            json.dump(uploaded, stream, indent=2)
url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{DEST}?download=true'
session = requests.Session()
session.trust_env = False
proxy = os.environ.get('HTTPS_PROXY')
print('Verifying anonymous download at exact revision ' + revision, flush=True)
response = session.get(url, timeout=(30, 300), proxies={'https': proxy} if proxy else {})
response.raise_for_status()
payload = response.content
assert len(payload) == archive.stat().st_size
assert hashlib.sha256(payload).hexdigest() == expected_sha
with zipfile.ZipFile(io.BytesIO(payload)) as package:
    expected = {'README.md', *(f'HZ-World/episode{i}.mp4' for i in range(1, 1001))}
    assert len(package.namelist()) == 1001 and set(package.namelist()) == expected
    assert package.testzip() is None
verified = dict(completed=True, repo_id=REPO, filename=DEST, revision=revision,
                download_url=url, sha256=expected_sha, bytes=len(payload),
                video_count=1000, anonymous_download=True, zip_crc_pass=True,
                package_receipt_sha256=hashlib.sha256((ROOT/'submission.complete.json').read_bytes()).hexdigest(),
                email_sent=False, official_receipt_confirmed=False)
with anonymous_receipt.open('x') as stream:
    json.dump(verified, stream, indent=2)
print(json.dumps(verified), flush=True)
