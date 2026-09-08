"""Publish the already-validated pretest ZIP; never replace historical archives."""
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import zipfile

import requests
from huggingface_hub import HfApi

A = Path('/data/di/worldarena2_track1_20260815')
ROOT = A / 'submission/final-email/p0-dualseed-selector-v1/pretest-100-draft'
ZIP = ROOT / 'HZ-World_P0-DualSeed-Selector-v1_Track1_Submission.zip'
SHA = 'e11561c38f3229d4b89abadef05f81bd257f5959385a10dd658ee7184f6914de'
REPO = 'clusternlh/worldarena-track1-hz-wam'
DEST = 'pretest-100/P0-DualSeed-Selector-v1/submission.zip'
assert hashlib.sha256(ZIP.read_bytes()).hexdigest() == SHA
values = {}
for line in (A / 'submission/project-hf.env').read_text().splitlines():
    line = line.removeprefix('export ').strip()
    if line.startswith('HF_TOKEN_PATH='):
        values['token_path'] = shlex.split(line.split('=', 1)[1])[0]
token_path = Path(values['token_path'])
assert token_path.is_absolute()
api = HfApi(token=token_path.read_text().strip())
upload_receipt = ROOT / 'hf-pretest-upload.complete.json'
if upload_receipt.exists():
    uploaded = json.loads(upload_receipt.read_text())
    assert uploaded['repo_id'] == REPO and uploaded['filename'] == DEST
    assert uploaded['sha256'] == SHA
    revision = uploaded['revision']
elif DEST in api.list_repo_files(REPO, repo_type='dataset'):
    revision = api.repo_info(REPO, repo_type='dataset').sha
else:
    revision = api.upload_file(
        path_or_fileobj=str(ZIP), path_in_repo=DEST,
        repo_id=REPO, repo_type='dataset',
        commit_message='Add first-100 P0 pretest ZIP; preserve all previous submissions',
    ).oid
    uploaded = dict(repo_id=REPO, filename=DEST, revision=revision, sha256=SHA,
                    completed=True, kind='pretest-100-upload-not-formal-submission')
    upload_receipt.write_text(json.dumps(uploaded, indent=2) + '\n')
url = f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{DEST}?download=true'
session = requests.Session()
session.trust_env = False
proxy = os.environ.get('HTTPS_PROXY')
response = session.get(url, timeout=(30, 120), proxies={'https': proxy} if proxy else {})
response.raise_for_status()
data = response.content
assert hashlib.sha256(data).hexdigest() == SHA
with zipfile.ZipFile(io.BytesIO(data)) as package:
    expected = {'README.md', *(f'HZ-World/episode{i}.mp4' for i in range(1, 101))}
    assert len(package.namelist()) == 101 and set(package.namelist()) == expected
    assert package.testzip() is None
verified = dict(completed=True, repo_id=REPO, filename=DEST, revision=revision,
                download_url=url, sha256=SHA, bytes=len(data), video_count=100,
                anonymous_download=True, formal_submission=False)
(ROOT / 'hf-pretest-anonymous.complete.json').write_text(json.dumps(verified, indent=2) + '\n')
print(json.dumps(verified))
