#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/di/worldarena2_track1_20260815
SOURCE=/home/huazhi/nlh/baseline
PYTHON=$ROOT/venv_reuse/bin/python
RUN=$ROOT/runs/v10-action-relational-native-attention
DATA=$RUN/data
MODEL=$ROOT/models/Wan2.2-TI2V-5B
RELATION=$RUN/relation-cache
OPTIMIZER=$DATA/optimizer-full-action-2060.jsonl
AUDIT=$DATA/audit20.jsonl
REPLAY=$DATA/replay500.jsonl
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
PROBE=$ROOT/probes/v6-gripper/gripper-probe.pt
OBS=$ROOT/probes/v6-gripper/observability-video-v2
TRAJECTORY=$ROOT/runs/v6-gripper-trajectory-r2/calibration.json
PHASE=${1:-dry-run}

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:$SOURCE:/home/huazhi/nlh/Wan2.2

check_gpu6() {
  local uuid memory util
  uuid=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$uuid"; then
    echo "GPU6 has a compute process; refusing v10 launch" >&2; exit 1
  fi
  memory=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits)
  util=$(nvidia-smi -i 6 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if (( memory >= 1024 || util >= 10 )); then
    echo "GPU6 is not idle: memory=$memory MiB util=$util%" >&2; exit 1
  fi
}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=6" \
    "cached latent+text only; T5/VAE forbidden in trainer" \
    "preflight -> smoke3 -> train50 -> audit50" \
    "gated train150 -> audit150 -> gated train300 -> audit300" \
    "gated train500 -> audit500"
  exit 0
fi

cd "$SOURCE"

common=(
  --checkpoint-dir "$MODEL" --optimizer-manifest "$OPTIMIZER"
  --audit-manifest "$AUDIT" --data-receipt "$DATA/data-receipt.json"
  --relation-root "$RELATION" --replay "$REPLAY"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --probe-checkpoint "$PROBE" --trajectory-calibration "$TRAJECTORY"
  --observability-root "$OBS" --output-dir "$RUN/gpu6"
)

run_gpu() {
  check_gpu6
  "$PYTHON" scripts/train_wan_v10_relational.py "${common[@]}" "$@"
}

mkdir -p "$RUN/gpu6"
case "$PHASE" in
  replay)
    "$PYTHON" scripts/build_wan_v10_replay.py --optimizer-manifest "$OPTIMIZER" --output "$REPLAY" ;;
  preflight) run_gpu --mode preflight ;;
  smoke) run_gpu --mode smoke ;;
  train50) run_gpu --mode train --stop-step 50 ;;
  audit50) run_gpu --mode audit --resume "$RUN/gpu6/step-000050.pt" --audit-step 50 ;;
  train150) run_gpu --mode train --resume "$RUN/gpu6/step-000050-gated.pt" --stop-step 150 ;;
  audit150) run_gpu --mode audit --resume "$RUN/gpu6/step-000150.pt" --audit-step 150 ;;
  train300) run_gpu --mode train --resume "$RUN/gpu6/step-000150-gated.pt" --stop-step 300 ;;
  audit300) run_gpu --mode audit --resume "$RUN/gpu6/step-000300.pt" --audit-step 300 ;;
  train500) run_gpu --mode train --resume "$RUN/gpu6/step-000300-gated.pt" --stop-step 500 ;;
  audit500) run_gpu --mode audit --resume "$RUN/gpu6/step-000500.pt" --audit-step 500 ;;
  *) echo "unsupported v10 phase: $PHASE" >&2; exit 2 ;;
esac
