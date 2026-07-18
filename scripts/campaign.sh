#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/di/worldarena2_track1_baseline}"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"
HF="${HF:-$ROOT/venv/bin/hf}"
DATASET="${DATASET:-$ROOT/data/dataset_track1}"
CONTROLS="${CONTROLS:-$ROOT/controls}"
OSCAR="${OSCAR:-$ROOT/oscar-public}"
CHECKPOINT="${CHECKPOINT:-$ROOT/checkpoints}"
COSMOS_REASON_PATH="${COSMOS_REASON_PATH:-$ROOT/cosmos-reason1-7b}"
STAGING="${STAGING:-$ROOT/submission}"
LOGS="${LOGS:-$ROOT/logs}"
SMOKE_DIR="${SMOKE_DIR:-$ROOT/smoke/videos}"
GATE_DIR="${GATE_DIR:-$ROOT/gate/videos}"
ARCHIVE="${ARCHIVE:-$ROOT/submission.tar.gz}"
PUBLIC_VERIFY_DIR="${PUBLIC_VERIFY_DIR:-$ROOT/public_verify_cache}"
PUBLISH_RESULT="${PUBLISH_RESULT:-$ROOT/publish-result.json}"
GPU_INDICES="${GPU_INDICES:-0,1,2,3,4,5,6,7}"
MIN_FREE_MIB="${MIN_FREE_MIB:-22000}"
MAX_UTIL_PERCENT="${MAX_UTIL_PERCENT:-10}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-60}"
SMOKE_COUNT="${SMOKE_COUNT:-3}"
GATE_COUNT="${GATE_COUNT:-20}"
EXPECTED_COUNT="${EXPECTED_COUNT:-1000}"
NO_PUBLISH="${NO_PUBLISH:-0}"
export COSMOS_REASON_PATH
export HF_HOME="${HF_HOME:-/data/di/hf_cache}"

source "$(dirname "${BASH_SOURCE[0]}")/gpu_indices.sh"
if ! parse_gpu_indices "$GPU_INDICES"; then
  exit 2
fi
WORKER_COUNT="${#GPU_LIST[@]}"

mkdir -p "$LOGS" "$STAGING/videos" "$SMOKE_DIR" "$GATE_DIR"

fail() {
  code=$?
  echo "$(date -Is) campaign failed with exit code $code" | tee -a "$LOGS/campaign.log"
  touch "$ROOT/CAMPAIGN_FAILED"
  exit "$code"
}
trap fail ERR
rm -f "$ROOT/CAMPAIGN_FAILED"

gpu_ready() {
  local gpu="$1"
  nvidia-smi --id="$gpu" --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits | awk \
    -v min_free="$MIN_FREE_MIB" -v max_util="$MAX_UTIL_PERCENT" \
    'NF == 2 && $1 >= min_free && $2 <= max_util { ready = 1 } END { exit !ready }'
}

echo "$(date -Is) waiting for selected GPUs: $GPU_INDICES" | tee -a "$LOGS/campaign.log"
touch "$ROOT/WAITING_FOR_GPUS"
while true; do
  READY=1
  for GPU in "${GPU_LIST[@]}"; do
    if ! gpu_ready "$GPU"; then
      READY=0
      break
    fi
  done
  if [ "$READY" -eq 1 ]; then
    break
  fi
  sleep "$GPU_POLL_SECONDS"
done
rm -f "$ROOT/WAITING_FOR_GPUS"

INSPECTION=$("$PYTHON" -m worldarena_baseline.cli inspect \
  --dataset-root "$DATASET" --smoke-count "$SMOKE_COUNT" --gate-count "$GATE_COUNT")
SMOKE_IDS=$(printf '%s' "$INSPECTION" | "$PYTHON" -c \
  'import json,sys; print(",".join(map(str,json.load(sys.stdin)["smoke_episode_ids"])))')
GATE_IDS=$(printf '%s' "$INSPECTION" | "$PYTHON" -c \
  'import json,sys; print(",".join(map(str,json.load(sys.stdin)["gate_episode_ids"])))')
echo "$(date -Is) smoke=$SMOKE_IDS gate=$GATE_IDS" | tee -a "$LOGS/campaign.log"

run_workers() {
  local episode_ids="$1"
  local output_dir="$2"
  local label="$3"
  local WORKER_INDEX GPU

  # Smoke and gate workers are deliberately serial, in selected-GPU order.
  for WORKER_INDEX in "${!GPU_LIST[@]}"; do
    GPU="${GPU_LIST[$WORKER_INDEX]}"
    "$PYTHON" -m worldarena_baseline.worker \
      --dataset-root "$DATASET" --controls-dir "$CONTROLS" \
      --output-dir "$output_dir" --checkpoint "$CHECKPOINT" \
      --oscar-repo "$OSCAR" --gpu-index "$GPU" \
      --worker-index "$WORKER_INDEX" --worker-count "$WORKER_COUNT" \
      --episode-ids "$episode_ids" --num-steps 5 \
      >"$LOGS/${label}-gpu${GPU}.log" 2>&1
  done
}

copy_validated_videos() {
  local source_dir="$1"
  local episode_ids="$2"
  local episode_id filename
  IFS=',' read -r -a _episode_ids <<< "$episode_ids"
  for episode_id in "${_episode_ids[@]}"; do
    printf -v filename 'episode_%06d.mp4' "$episode_id"
    cp "$source_dir/$filename" "$STAGING/videos/$filename"
  done
}

run_workers "$SMOKE_IDS" "$SMOKE_DIR" "smoke"
"$PYTHON" -m worldarena_baseline.cli validate-videos --videos-dir "$SMOKE_DIR" --episode-ids "$SMOKE_IDS"
touch "$ROOT/SMOKE_COMPLETE"
copy_validated_videos "$SMOKE_DIR" "$SMOKE_IDS"

run_workers "$GATE_IDS" "$GATE_DIR" "gate"
"$PYTHON" -m worldarena_baseline.cli validate-videos --videos-dir "$GATE_DIR" --episode-ids "$GATE_IDS"
touch "$ROOT/GATE_COMPLETE"
copy_validated_videos "$GATE_DIR" "$GATE_IDS"

echo "$(date -Is) full generation" | tee -a "$LOGS/campaign.log"
PIDS=()
for WORKER_INDEX in "${!GPU_LIST[@]}"; do
  GPU="${GPU_LIST[$WORKER_INDEX]}"
  "$PYTHON" -m worldarena_baseline.worker \
    --dataset-root "$DATASET" --controls-dir "$CONTROLS" \
    --output-dir "$STAGING/videos" --checkpoint "$CHECKPOINT" \
    --oscar-repo "$OSCAR" --gpu-index "$GPU" \
    --worker-index "$WORKER_INDEX" --worker-count "$WORKER_COUNT" \
    --num-steps 5 >"$LOGS/full-gpu${GPU}.log" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done

"$PYTHON" -m worldarena_baseline.cli validate-videos \
  --videos-dir "$STAGING/videos" --expected-count "$EXPECTED_COUNT"
"$PYTHON" -m worldarena_baseline.cli write-readme --output "$STAGING/model_readme.md"
"$PYTHON" -m worldarena_baseline.cli package \
  --staging-dir "$STAGING" --archive "$ARCHIVE" --expected-count "$EXPECTED_COUNT"

if [ "$NO_PUBLISH" = "1" ]; then
  touch "$ROOT/PACKAGE_COMPLETE"
  echo "$(date -Is) package complete; NO_PUBLISH=1" | tee -a "$LOGS/campaign.log"
  exit 0
fi

echo "$(date -Is) waiting for Hugging Face authentication" | tee -a "$LOGS/campaign.log"
touch "$ROOT/WAITING_FOR_HF_AUTH"
while ! "$HF" auth whoami >/dev/null 2>&1; do
  sleep 60
done
rm -f "$ROOT/WAITING_FOR_HF_AUTH"

"$PYTHON" -m worldarena_baseline.publish \
  --archive "$ARCHIVE" \
  --verify-dir "$PUBLIC_VERIFY_DIR" | tee "$PUBLISH_RESULT"
touch "$ROOT/COMPLETE"
echo "$(date -Is) complete" | tee -a "$LOGS/campaign.log"
