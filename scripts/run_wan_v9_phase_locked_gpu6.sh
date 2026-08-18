#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/di/worldarena2_track1_20260815
SOURCE=/home/huazhi/nlh/baseline
PYTHON=$ROOT/venv_reuse/bin/python
RUN=$ROOT/runs/v9-phase-locked-action-cross-attention/gpu6
MODEL=$ROOT/models/Wan2.2-TI2V-5B
CACHE=$ROOT/cache/v4-clean-1785-v3
MANIFEST=$CACHE/manifests/clean-1785.cached.jsonl
V8_DATA=$ROOT/runs/v8-direct-action-band/data
V8_REPLAY=$V8_DATA/replay250.jsonl
AUDIT=$V8_DATA/audit20.jsonl
DEV=$ROOT/eval/dev-fast-20-v3/dev-fast-20.jsonl
TRANSITIONS=$RUN/transition-cache
NEGATIVE=$ROOT/runs/v8-direct-action-band/negative-cache/wan_v8_counterfactuals
DATASET=$ROOT/datasets/FlowWAM_RoboTwin_extracted
URDF=$SOURCE/source_inputs/arx5_description_isaac.urdf
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
PROBE=$ROOT/probes/v6-gripper/gripper-probe.pt
OBS=$ROOT/probes/v6-gripper/observability-video-v2
TRAJECTORY_CALIBRATION=$ROOT/runs/v6-gripper-trajectory-r2/calibration.json
REPLAY=$RUN/replay500.jsonl
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=6" \
    "preflight -> smoke3 -> train25 -> audit25 -> train100 -> audit100" \
    "conditional train250 -> audit250 -> conditional phaseT500 -> audit500" \
    "81 frames / 480x640 / cached latent+text / no T5 / no VAE"
  exit 0
fi

check_gpu6() {
  local uuid memory util
  uuid=$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader,nounits | tr -d ' ')
  if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader,nounits | grep -Fxq "$uuid"; then
    echo "GPU6 has a compute process; refusing v9 launch" >&2; exit 1
  fi
  memory=$(nvidia-smi -i 6 --query-gpu=memory.used --format=csv,noheader,nounits)
  util=$(nvidia-smi -i 6 --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if (( memory >= 1024 || util >= 10 )); then
    echo "GPU6 is not idle enough for v9: memory=$memory MiB util=$util%" >&2; exit 1
  fi
}

mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:$SOURCE:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"

validate_source() {
  "$PYTHON" scripts/validate_wan_v9_sync_closure.py --source-root "$SOURCE" --output "$RUN/source-closure.json" >/dev/null
}

build_replay() {
  "$PYTHON" scripts/build_wan_v9_replay.py --v8-replay "$V8_REPLAY" --audit-manifest "$AUDIT" --output "$REPLAY"
}

common=(
  --checkpoint-dir "$MODEL" --cache-root "$CACHE" --manifest "$MANIFEST"
  --transition-root "$TRANSITIONS" --replay "$REPLAY" --audit-manifest "$AUDIT"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --probe-checkpoint "$PROBE" --trajectory-calibration "$TRAJECTORY_CALIBRATION"
  --observability-root "$OBS" --output-dir "$RUN" --source-receipt "$RUN/source-closure.json"
)

run_gpu() {
  check_gpu6
  "$PYTHON" scripts/train_wan_v9_phase_locked.py "${common[@]}" "$@"
}

require_decision() {
  local report=$1 key=$2
  "$PYTHON" -c 'import json,sys; payload=json.load(open(sys.argv[1])); raise SystemExit(0 if payload["decision"].get(sys.argv[2]) is True else 1)' "$report" "$key"
}

validate_source
build_replay
case "$PHASE" in
  all)
    "$0" cache
    "$0" preflight
    "$0" smoke
    "$0" train25
    "$0" audit25
    require_decision "$RUN/audit-025.json" continue
    "$0" train100
    "$0" audit100
    require_decision "$RUN/audit-100.json" continue
    "$0" train250
    "$0" audit250
    require_decision "$RUN/audit-250.json" pass
    "$0" train500
    "$0" audit500
    ;;
  cache)
    "$PYTHON" scripts/cache_wan_v9_transitions.py \
      --clean-manifest "$MANIFEST" --audit-manifest "$AUDIT" --dev-manifest "$DEV" \
      --dataset-root "$DATASET" --cache-root "$CACHE" --negative-root "$NEGATIVE" \
      --urdf "$URDF" --output-root "$TRANSITIONS"
    ;;
  preflight) run_gpu --mode preflight ;;
  smoke) run_gpu --mode smoke ;;
  train25) run_gpu --mode train --stop-step 25 ;;
  audit25) run_gpu --mode audit --resume "$RUN/step-000025.pt" --audit-step 25 ;;
  train100) run_gpu --mode train --resume "$RUN/step-000025-gated.pt" --stop-step 100 ;;
  audit100) run_gpu --mode audit --resume "$RUN/step-000100.pt" --audit-step 100 ;;
  train250) run_gpu --mode train --resume "$RUN/step-000100-gated.pt" --stop-step 250 ;;
  audit250) run_gpu --mode audit --resume "$RUN/step-000250.pt" --audit-step 250 ;;
  train500) run_gpu --mode train --resume "$RUN/step-000250-gated.pt" --stop-step 500 ;;
  audit500) run_gpu --mode audit --resume "$RUN/step-000500.pt" --audit-step 500 ;;
  *) echo "unsupported v9 phase: $PHASE" >&2; exit 2 ;;
esac
