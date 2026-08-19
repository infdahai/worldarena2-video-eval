#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/di/worldarena2_track1_20260815
SOURCE=/home/huazhi/nlh/baseline
PYTHON=$ROOT/venv_reuse/bin/python
RUN=$ROOT/runs/v11-worldarena-balanced-bimanual-world-controller
DATA=$ROOT/runs/v10-action-relational-native-attention/data
MODEL=$ROOT/models/Wan2.2-TI2V-5B
RELATION=$ROOT/runs/v10-action-relational-native-attention/relation-cache
OPTIMIZER=$DATA/optimizer-full-action-2060.jsonl
AUDIT=$DATA/audit20.jsonl
REPLAY=$RUN/replay2060.jsonl
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
OBS=$ROOT/probes/v6-gripper/observability-video-v2
SOURCE_RECEIPT=$RUN/source-receipt.json
OUTPUT=$RUN/gpu6
PHASE=${1:-dry-run}

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:$SOURCE:/home/huazhi/nlh/Wan2.2

validate_source() {
  "$PYTHON" scripts/validate_wan_v11_sync_closure.py \
    --source-root "$SOURCE" --expected-receipt "$SOURCE_RECEIPT" >/dev/null
}

check_gpu6() {
  local uuid memory util
  uuid=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$uuid"; then
    echo "GPU6 has a compute process; refusing v11 launch" >&2; exit 1
  fi
  memory=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits)
  util=$(nvidia-smi -i 6 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if (( memory >= 1024 || util >= 10 )); then
    echo "GPU6 is not idle: memory=$memory MiB util=$util%" >&2; exit 1
  fi
}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "v11 GPU6 world-size1; frozen Wan + frozen clean-gated-step10" \
    "replay -> preflight -> smoke -> train100 -> audit100" \
    "gated train500 -> audit500 -> gated train2060 -> audit2060" \
    "cached latent+text only; no T5/VAE in training hot path"
  exit 0
fi

cd "$SOURCE"
mkdir -p "$OUTPUT"
validate_source

common=(
  --checkpoint-dir "$MODEL" --optimizer-manifest "$OPTIMIZER"
  --audit-manifest "$AUDIT" --data-receipt "$DATA/data-receipt.json"
  --relation-root "$RELATION" --replay "$REPLAY"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --observability-root "$OBS" --source-receipt "$SOURCE_RECEIPT" --output-dir "$OUTPUT"
)

run_gpu() {
  check_gpu6
  "$PYTHON" scripts/train_wan_v11_bimanual.py "${common[@]}" "$@"
}

case "$PHASE" in
  replay) "$PYTHON" scripts/train_wan_v11_bimanual.py "${common[@]}" --mode replay ;;
  preflight) run_gpu --mode preflight ;;
  smoke) run_gpu --mode smoke ;;
  train100) run_gpu --mode train --stop-step 100 ;;
  audit100) run_gpu --mode audit --resume "$OUTPUT/step-000100.pt" --audit-step 100 ;;
  train500) run_gpu --mode train --resume "$OUTPUT/step-000100-gated.pt" --stop-step 500 ;;
  audit500) run_gpu --mode audit --resume "$OUTPUT/step-000500.pt" --audit-step 500 ;;
  train2060) run_gpu --mode train --resume "$OUTPUT/step-000500-gated.pt" --stop-step 2060 ;;
  audit2060) run_gpu --mode audit --resume "$OUTPUT/step-002060.pt" --audit-step 2060 ;;
  *) echo "unsupported v11 phase: $PHASE" >&2; exit 2 ;;
esac
