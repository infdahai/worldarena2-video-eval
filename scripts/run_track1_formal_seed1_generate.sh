#!/usr/bin/env bash
set -euo pipefail

GPU=${1:-2}
if [[ "$GPU" != "2" ]]; then
  echo "formal P0 seed1 generation is pinned to physical GPU2" >&2
  exit 64
fi

ROOT=/data/di/worldarena2_track1_20260815
BASE=/home/huazhi/nlh/baseline
SOURCE=/home/huazhi/nlh/FlowWAM_WorldArena
RELEASE="$ROOT/submission/releases/p0-dual-seed-r1"
DATASET="$ROOT/datasets/WorldArena2.0-official-track1/extracted/dataset_track1"
MANIFEST="$ROOT/datasets/WorldArena2.0-official-track1/episode-manifest.jsonl"
EMBODIMENT="$ROOT/sources/RoboTwin/assets"
CHECKPOINT="$ROOT/models/FlowWAM/flowwam_worldarena_stage1.safetensors"
MODEL_LAYOUT="$ROOT/models/FlowWAM-layout"
RUN="$RELEASE/generation/seed1"

mkdir -p "$RELEASE/logs" "$RUN" "$RELEASE/locks"
exec 9>"$RELEASE/launcher.lock"
flock -n 9 || exit 75
printf '%s\n' "$$" >"$RELEASE/launcher.pid"
trap 'rm -f "$RELEASE/launcher.pid"' EXIT

exec 8>>"$ROOT/official_track1_eval/tmp/gpu2.track1-p0-seed1.lock"
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
    --seed 1 \
    --flow-max-magnitude 20
fi

flock -u 8
printf 'formal seed1 generation complete: %s\n' "$RUN/stage1-only.receipt.json"
