#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-0}
if [[ "$GPU" != "0" ]]; then
  echo "resource-constrained formal submission is pinned to physical GPU0" >&2
  exit 64
fi

ROOT=/data/di/worldarena2_track1_20260815
BASE=/home/huazhi/nlh/baseline
SOURCE=/home/huazhi/nlh/FlowWAM_WorldArena
RELEASE="$ROOT/submission/releases/seed4-resource-constrained-r1"
DATASET="$ROOT/datasets/WorldArena2.0-official-track1/extracted/dataset_track1"
MANIFEST="$ROOT/datasets/WorldArena2.0-official-track1/episode-manifest.jsonl"
EMBODIMENT="$ROOT/sources/RoboTwin/assets"
CHECKPOINT="$ROOT/models/FlowWAM/flowwam_worldarena_stage1.safetensors"
MODEL_LAYOUT="$ROOT/models/FlowWAM-layout"
RUN="$RELEASE/generation/seed4"
VIDEO_DIR="$RUN/FlowWAMOfficialStage1_test"
SMOKE="$ROOT/submission/releases/p0-r1/formal-input-smoke/seed4"
PROJECT_ENV="$ROOT/submission/project-hf.env"
HF_HOME_PROJECT="$ROOT/submission/hf-home"
REPO_ID=clusternlh/worldarena-track1-flowwam-stage1-seed4-20260826
ARCHIVE_NAME=worldarena-track1-flowwam-stage1-seed4-20260826.tar.gz

mkdir -p "$RELEASE/logs" "$RUN" "$RELEASE/locks"
exec 9>"$RELEASE/launcher.lock"
flock -n 9 || exit 75
printf '%s\n' "$$" >"$RELEASE/launcher.pid"
trap 'rm -f "$RELEASE/launcher.pid"' EXIT

exec 8>>"$ROOT/official_track1_eval/tmp/gpu0.adaptive-training.lock"
flock -n 8 || exit 75
GPU_UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
mapfile -t GPU_PIDS < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits \
  | awk -F', ' -v uuid="$GPU_UUID" '$1 == uuid {print $2}')
if ((${#GPU_PIDS[@]})); then
  printf 'GPU%s is occupied; refusing to start:' "$GPU" >&2
  printf ' %s' "${GPU_PIDS[@]}" >&2
  printf '\n' >&2
  exit 75
fi

mkdir -p "$VIDEO_DIR"
if [[ ! -e "$VIDEO_DIR/episode1.mp4" ]]; then
  ln "$SMOKE/FlowWAMOfficialStage1_test/episode1.mp4" "$VIDEO_DIR/episode1.mp4"
fi
"$ROOT/venv_reuse/bin/python" -B - "$SMOKE/stage1-only.receipt.json" "$VIDEO_DIR/episode1.mp4" "$RELEASE/reused-smoke.complete.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
source_receipt,video,target=map(Path,sys.argv[1:])
receipt=json.loads(source_receipt.read_text())
row=receipt["videos"][0]
digest=hashlib.sha256(video.read_bytes()).hexdigest()
if receipt.get("stage2_refiner") != "identity-disabled" or receipt["stage1"]["seed"] != 4:
    raise RuntimeError("formal smoke lineage mismatch")
if row["name"] != "episode1.mp4" or row["sha256"] != digest:
    raise RuntimeError("formal smoke video SHA mismatch")
payload={"completed":True,"contract":"worldarena-track1-formal-smoke-reuse/1","episode_id":1,"seed":4,"video":str(video),"video_sha256":digest,"source_receipt":str(source_receipt),"source_receipt_sha256":hashlib.sha256(source_receipt.read_bytes()).hexdigest()}
tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
PY

if [[ ! -s "$RUN/stage1-only.receipt.json" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
  export TOKENIZERS_PARALLELISM=false
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  export PYTHONPATH="$ROOT/submission/python-overlays/sapien-3.0.0b1:$BASE/src:$SOURCE/src"
  cd "$BASE"
  "$ROOT/venv_reuse/bin/python" -B scripts/run_flowwam_official_stage1.py \
    --artifact-root "$ROOT" \
    --test-dataset-dir "$DATASET" \
    --embodiment-dir "$EMBODIMENT" \
    --output-dir "$RUN" \
    --checkpoint "$CHECKPOINT" \
    --local-model-path "$MODEL_LAYOUT" \
    --expected-count 1000 \
    --sample-manifest "$MANIFEST" \
    --resume-valid-existing \
    --physical-gpu "$GPU" \
    --seed 4 \
    --flow-max-magnitude 20
fi

flock -u 8

# The resource-constrained seed4 run is a reusable generation asset, but it is
# not automatically the highest-scoring submission when seed1 capacity later
# becomes available.  A release-scoped sentinel provides a fail-closed decision
# gate between expensive generation and the externally visible HF upload.
UPLOAD_HOLD="$RELEASE/HOLD_BEFORE_UPLOAD"
if [[ -e "$UPLOAD_HOLD" ]]; then
  printf 'generation complete; upload held by %s\n' "$UPLOAD_HOLD" >&2
  printf 'remove the sentinel only after choosing seed4 fallback or frozen P0\n' >&2
  exit 76
fi

"$ROOT/venv_reuse/bin/python" -B - "$RUN/stage1-only.receipt.json" "$VIDEO_DIR" "$RELEASE/seed4-predictions.jsonl" "$RELEASE/seed4-selection.complete.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
receipt_path,video_dir,predictions,target=map(Path,sys.argv[1:])
receipt=json.loads(receipt_path.read_text())
if receipt.get("stage2_refiner") != "identity-disabled" or receipt["stage1"]["seed"] != 4:
    raise RuntimeError("formal seed4 receipt contract mismatch")
rows={row["name"]:row for row in receipt["videos"]}
output=[]
for episode_id in range(1,1001):
    name=f"episode{episode_id}.mp4"; video=video_dir/name
    if name not in rows or not video.is_file(): raise RuntimeError(f"missing {name}")
    digest=hashlib.sha256(video.read_bytes()).hexdigest()
    if rows[name]["sha256"] != digest: raise RuntimeError(f"SHA mismatch {name}")
    output.append({"episode_id":episode_id,"selected_seed":4,"selected_video":str(video),"selected_video_sha256":digest})
content="".join(json.dumps(row,sort_keys=True,separators=(",",":"))+"\n" for row in output)
tmp=predictions.with_suffix(".partial"); tmp.write_text(content); os.replace(tmp,predictions)
payload={"completed":True,"terminal_state":"resource_constrained_seed4_selected","contract":"worldarena-track1-seed4-selection/1","row_count":1000,"selected_seed":4,"predictions":str(predictions),"predictions_sha256":hashlib.sha256(predictions.read_bytes()).hexdigest(),"generation_receipt":str(receipt_path),"generation_receipt_sha256":hashlib.sha256(receipt_path.read_bytes()).hexdigest()}
tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
PY

"$ROOT/venv_reuse/bin/python" -B - "$RELEASE/metadata.json" <<'PY'
import json,os,sys
from pathlib import Path
target=Path(sys.argv[1])
payload={"model_name":"HZ-World","version":"Stage1-Seed4-v1","organization":"Huazhi AI","release_year":2026,"source_type":"open_source","control_type":"hybrid"}
tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
PY

export PYTHONPATH="$ROOT/submission/code"
if [[ ! -s "$RELEASE/staging/submission-staging.complete.json" ]]; then
  "$ROOT/venv_reuse/bin/python" -B "$ROOT/submission/code/run_track1_release.py" stage \
    --predictions "$RELEASE/seed4-predictions.jsonl" \
    --staging "$RELEASE/staging" \
    --metadata "$RELEASE/metadata.json" \
    --expected-count 1000
fi
if [[ ! -s "$RELEASE/package.complete.json" ]]; then
  "$ROOT/venv_reuse/bin/python" -B "$ROOT/submission/code/run_track1_release.py" package \
    --staging "$RELEASE/staging" \
    --archive "$RELEASE/$ARCHIVE_NAME" \
    --receipt "$RELEASE/package.complete.json" \
    --expected-count 1000
fi

set -a
source "$PROJECT_ENV"
set +a
IFS= read -r HF_TOKEN <"$HF_TOKEN_PATH"
export HF_TOKEN
export HF_HOME="$HF_HOME_PROJECT"
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
"$ROOT/submission/venv/bin/python" -B - "$RELEASE/$ARCHIVE_NAME" "$REPO_ID" "$RELEASE/hf-upload.complete.json" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
from huggingface_hub import HfApi
archive=Path(sys.argv[1]).resolve(strict=True); repo_id=sys.argv[2]; target=Path(sys.argv[3])
if "smoke" in repo_id.lower() or "smoke" in archive.name.lower(): raise RuntimeError("formal repo/archive cannot be smoke")
digest_hash=hashlib.sha256()
with archive.open("rb") as handle:
    for chunk in iter(lambda:handle.read(8<<20),b""): digest_hash.update(chunk)
digest=digest_hash.hexdigest(); api=HfApi(token=os.environ["HF_TOKEN"])
api.create_repo(repo_id=repo_id,repo_type="dataset",private=False,exist_ok=True)
existing=api.list_repo_files(repo_id=repo_id,repo_type="dataset")
archives=[name for name in existing if "/" not in name and name.endswith((".tar.gz",".tgz"))]
if archives and archives != [archive.name]: raise RuntimeError(f"formal repo already has another root archive: {archives}")
commit=api.upload_file(path_or_fileobj=str(archive),path_in_repo=archive.name,repo_id=repo_id,repo_type="dataset",commit_message="Upload frozen WorldArena 2.0 Track 1 submission")
info=api.repo_info(repo_id=repo_id,repo_type="dataset")
if info.private or getattr(info,"gated",False): raise RuntimeError("formal dataset repo is not public")
files=api.list_repo_files(repo_id=repo_id,repo_type="dataset",revision=info.sha)
archives=[name for name in files if "/" not in name and name.endswith((".tar.gz",".tgz"))]
if archives != [archive.name]: raise RuntimeError(f"formal repo root archive set mismatch: {archives}")
payload={"completed":True,"terminal_state":"formal_hf_submission_uploaded","contract":"worldarena-track1-hf-formal-upload/1","repo_id":repo_id,"revision":info.sha,"archive":str(archive),"archive_filename":archive.name,"archive_sha256":digest,"public":True,"gated":False,"root_archives":archives,"commit_url":str(commit)}
tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
print(json.dumps({"repo_id":repo_id,"revision":info.sha,"archive_sha256":digest},sort_keys=True))
PY

"$ROOT/submission/venv/bin/python" -B - "$RELEASE/hf-upload.complete.json" "$RELEASE/anonymous-download.tar.gz" "$RELEASE/anonymous-verify.complete.json" <<'PY'
import hashlib,json,os,shutil,sys,tarfile,tempfile,urllib.request
from pathlib import Path
upload_path,download,target=map(Path,sys.argv[1:]); upload=json.loads(upload_path.read_text())
repo=upload["repo_id"]; revision=upload["revision"]; filename=upload["archive_filename"]
url=f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}?download=true"
opener=urllib.request.build_opener(urllib.request.ProxyHandler({"http":"http://127.0.0.1:7890","https":"http://127.0.0.1:7890"}))
request=urllib.request.Request(url,headers={"User-Agent":"worldarena-track1-public-verifier/1"})
download.parent.mkdir(parents=True,exist_ok=True)
with opener.open(request,timeout=3600) as response, download.open("wb") as output: shutil.copyfileobj(response,output)
digest_hash=hashlib.sha256()
with download.open("rb") as handle:
    for chunk in iter(lambda:handle.read(8<<20),b""): digest_hash.update(chunk)
digest=digest_hash.hexdigest()
if digest != upload["archive_sha256"]: raise RuntimeError("anonymous archive SHA mismatch")
with tarfile.open(download,"r:gz") as archive:
    names=archive.getnames()
expected=["model_readme.md","videos"]+[f"videos/episode_{i:06d}.mp4" for i in range(1,1001)]
if names != expected: raise RuntimeError("anonymous archive member set/order mismatch")
payload={"completed":True,"terminal_state":"formal_hf_submission_anonymously_verified","contract":"worldarena-track1-hf-formal-anonymous/1","repo_id":repo,"revision":revision,"filename":filename,"authorization_header_used":False,"download":str(download),"download_sha256":digest,"video_count":1000}
tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
print(json.dumps({"repo_id":repo,"revision":revision,"anonymous_sha256":digest},sort_keys=True))
PY

"$ROOT/venv_reuse/bin/python" -B - "$RELEASE" "$GPU_UUID" <<'PY'
import hashlib,json,os,sys
from pathlib import Path
root=Path(sys.argv[1]); gpu_uuid=sys.argv[2]
upload=json.loads((root/"hf-upload.complete.json").read_text()); anonymous=json.loads((root/"anonymous-verify.complete.json").read_text())
if upload["archive_sha256"] != anonymous["download_sha256"]: raise RuntimeError("deployment SHA gate failed")
payload={"completed":True,"terminal_state":"formal_track1_submission_complete","contract":"worldarena-track1-resource-constrained-final/1","winner":"official_stage1_seed4","resource_policy":{"max_gpu_count":1,"physical_gpu":0,"gpu_uuid":gpu_uuid,"postgen_p0_skipped_reason":"user_required_at_most_one_gpu_and_submission_deadline"},"repo_id":upload["repo_id"],"revision":upload["revision"],"archive":upload["archive"],"archive_sha256":upload["archive_sha256"],"anonymous_sha_gate":True,"receipts":{"generation":str(root/"generation/seed4/stage1-only.receipt.json"),"selection":str(root/"seed4-selection.complete.json"),"package":str(root/"package.complete.json"),"upload":str(root/"hf-upload.complete.json"),"anonymous":str(root/"anonymous-verify.complete.json")}}
target=root/"final-submission.complete.json"; tmp=target.with_suffix(".partial"); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n"); os.replace(tmp,target)
print(json.dumps(payload,sort_keys=True))
PY
