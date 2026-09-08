#!/usr/bin/env bash
set -euo pipefail

ART=/data/di/worldarena2_track1_20260815
BASE=/home/huazhi/nlh/baseline
ROOT="$ART/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/test1000-routing-mvp"
PY="$ART/envs/flowwam-v16/bin/python"
RUNNER="$BASE/scripts/run_flowwam_official_stage1.py"
DATA="$ART/datasets/WorldArena2.0-official-track1/extracted/dataset_track1"
EMBODIMENT="$ART/sources/RoboTwin/assets"
CHECKPOINT="$ART/models/FlowWAM/flowwam_worldarena_stage1.safetensors"
# The original layout lost its backing Wan2.2 base-model directory during the
# later cleanup. Reuse the already-complete ORB base-model layout; the Stage1
# checkpoint remains the frozen Official FlowWAM parent below.
LAYOUT="$ART/runs/flowwam-orb-20260904/model-layout"
MANIFEST_ROOT="$ART/submission/releases/p0-dual-seed-parallel-r1/manifests"

run_one() {
  local gpu="$1" seed="$2" shard="$3" step="$4"
  local manifest="$MANIFEST_ROOT/seed1.gpu${shard}.jsonl"
  local out="$ROOT/candidates/seed${seed}/shard${shard}"
  local receipt="$out/stage1-only.receipt.json"
  if [[ -f "$receipt" ]]; then
    echo "skip terminal seed=$seed shard=$shard step=$step receipt=$receipt"
    return 0
  fi
  echo "start seed=$seed shard=$shard step=$step gpu=$gpu manifest=$manifest"
  env CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false \
    PYTHONPATH="$ART/envs/flowwam-renderer-py310/lib/python3.10/site-packages:$BASE/src:/home/huazhi/nlh/FlowWAM_WorldArena/src" \
    "$PY" -B "$RUNNER" \
      --artifact-root "$ART" \
      --test-dataset-dir "$DATA" \
      --embodiment-dir "$EMBODIMENT" \
      --output-dir "$out" \
      --checkpoint "$CHECKPOINT" \
      --local-model-path "$LAYOUT" \
      --expected-count 125 \
      --sample-manifest "$manifest" \
      --resume-valid-existing \
      --physical-gpu "$gpu" \
      --seed "$seed" \
      --flow-max-magnitude 20
  echo "done seed=$seed shard=$shard step=$step gpu=$gpu receipt=$receipt"
}

worker() {
  local gpu="$1"
  case "$gpu" in
    0)
      run_one 0 2 0 1207
      run_one 0 2 7 1214
      run_one 0 3 0 1215
      ;;
    1)
      run_one 1 2 1 1208
      run_one 1 3 1 1216
      run_one 1 3 7 1222
      ;;
    2)
      run_one 2 2 2 1209
      run_one 2 3 2 1217
      ;;
    3)
      run_one 3 2 3 1210
      run_one 3 3 3 1218
      ;;
    4)
      run_one 4 2 4 1211
      run_one 4 3 4 1219
      ;;
    5)
      run_one 5 2 5 1212
      run_one 5 3 5 1220
      ;;
    6)
      run_one 6 2 6 1213
      run_one 6 3 6 1221
      ;;
    *)
      echo "invalid GPU: $gpu" >&2
      return 2
      ;;
  esac
}

if [[ "${1:-}" == worker ]]; then
  worker "$2"
  exit 0
fi

mkdir -p "$ROOT/tools" "$ROOT/logs" "$ROOT/pids" "$ROOT/locks" "$ROOT/candidates/seed2" "$ROOT/candidates/seed3"
for gpu in 0 1 2 3 4 5 6; do
  if nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | grep -Eq '^[0-9]+$'; then
    echo "refuse occupied gpu=$gpu" >&2
    continue
  fi
  log="$ROOT/logs/gpu${gpu}.log"
  pidfile="$ROOT/pids/gpu${gpu}.pid"
  lock="$ROOT/locks/gpu${gpu}.lock"
  nohup flock -n "$lock" "$0" worker "$gpu" >"$log" 2>&1 &
  pid=$!
  printf '%s\n' "$pid" >"$pidfile"
  echo "launched gpu=$gpu pid=$pid log=$log"
done
