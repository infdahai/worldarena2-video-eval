#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/di/worldarena2_track1_baseline}"
PYTHON="$ROOT/venv/bin/python"
HF="$ROOT/venv/bin/hf"
DATASET="$ROOT/data/dataset_track1"
CONTROLS="$ROOT/controls"
OSCAR="$ROOT/oscar-public"
CHECKPOINT="$ROOT/checkpoints"
COSMOS_REASON_PATH="${COSMOS_REASON_PATH:-$ROOT/cosmos-reason1-7b}"
export COSMOS_REASON_PATH
STAGING="$ROOT/submission"
LOGS="$ROOT/logs"
mkdir -p "$LOGS" "$STAGING/videos" "$ROOT/smoke/videos"

fail() {
  code=$?
  echo "$(date -Is) campaign failed with exit code $code" | tee -a "$LOGS/campaign.log"
  touch "$ROOT/CAMPAIGN_FAILED"
  exit "$code"
}
trap fail ERR
rm -f "$ROOT/CAMPAIGN_FAILED"

echo "$(date -Is) waiting for all GPUs" | tee -a "$LOGS/campaign.log"
touch "$ROOT/WAITING_FOR_GPUS"
while true; do
  READY=$(nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | awk '$1 >= 22000 && $2 <= 10 {n++} END {print n+0}')
  if [ "$READY" -eq 8 ]; then
    break
  fi
  sleep 60
done
rm -f "$ROOT/WAITING_FOR_GPUS"

SMOKE_IDS=$($PYTHON -m worldarena_baseline.cli inspect --dataset-root "$DATASET" | $PYTHON -c 'import json,sys; print(",".join(map(str,json.load(sys.stdin)["smoke_episode_ids"])))')
echo "$(date -Is) smoke=$SMOKE_IDS" | tee -a "$LOGS/campaign.log"
PIDS=()
for GPU in 0 1 2; do
  "$PYTHON" -m worldarena_baseline.worker \
    --dataset-root "$DATASET" --controls-dir "$CONTROLS" \
    --output-dir "$ROOT/smoke/videos" --checkpoint "$CHECKPOINT" \
    --oscar-repo "$OSCAR" --gpu-index "$GPU" --worker-index "$GPU" \
    --worker-count 3 --episode-ids "$SMOKE_IDS" --num-steps 5 \
    >"$LOGS/smoke-gpu${GPU}.log" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done

$PYTHON -m worldarena_baseline.cli validate-videos \
  --videos-dir "$ROOT/smoke/videos" --episode-ids "$SMOKE_IDS"
touch "$ROOT/SMOKE_COMPLETE"
cp "$ROOT"/smoke/videos/episode_*.mp4 "$STAGING/videos/"

echo "$(date -Is) full generation" | tee -a "$LOGS/campaign.log"
PIDS=()
for GPU in 0 1 2 3 4 5 6 7; do
  "$PYTHON" -m worldarena_baseline.worker \
    --dataset-root "$DATASET" --controls-dir "$CONTROLS" \
    --output-dir "$STAGING/videos" --checkpoint "$CHECKPOINT" \
    --oscar-repo "$OSCAR" --gpu-index "$GPU" --worker-index "$GPU" \
    --worker-count 8 --num-steps 5 >"$LOGS/full-gpu${GPU}.log" 2>&1 &
  PIDS+=("$!")
done
for PID in "${PIDS[@]}"; do wait "$PID"; done

$PYTHON -m worldarena_baseline.cli validate-videos --videos-dir "$STAGING/videos" --expected-count 1000
$PYTHON -m worldarena_baseline.cli write-readme --output "$STAGING/model_readme.md"
$PYTHON -m worldarena_baseline.cli package --staging-dir "$STAGING" --archive "$ROOT/submission.tar.gz"

echo "$(date -Is) waiting for Hugging Face authentication" | tee -a "$LOGS/campaign.log"
touch "$ROOT/WAITING_FOR_HF_AUTH"
while ! "$HF" auth whoami >/dev/null 2>&1; do
  sleep 60
done
rm -f "$ROOT/WAITING_FOR_HF_AUTH"

$PYTHON -m worldarena_baseline.publish \
  --archive "$ROOT/submission.tar.gz" \
  --verify-dir "$ROOT/public_verify_cache" | tee "$ROOT/publish-result.json"
touch "$ROOT/COMPLETE"
echo "$(date -Is) complete" | tee -a "$LOGS/campaign.log"
