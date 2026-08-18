#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/di/worldarena2_track1_20260815
SOURCE=/home/huazhi/nlh/baseline
PYTHON=$ROOT/venv_reuse/bin/python
RUN=$ROOT/runs/v8-direct-action-band/mechanism-gpu6
MODEL=$ROOT/models/Wan2.2-TI2V-5B
CACHE=$ROOT/cache/v4-clean-1785-v3
MANIFEST=$CACHE/manifests/clean-1785.cached.jsonl
DATA=$ROOT/runs/v8-direct-action-band/data
NEGATIVE=$ROOT/runs/v8-direct-action-band/negative-cache/wan_v8_counterfactuals
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
PROBE=$ROOT/probes/v6-gripper/gripper-probe.pt
SPLIT=$ROOT/probes/v6-gripper/clean1785-probe-split.json
OBS=$ROOT/probes/v6-gripper/observability-video-v2
TRAJ_CAL=$ROOT/runs/v6-gripper-trajectory-r2/calibration.json
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' "GPU6 only" "preflight -> smoke(3 full iterations) -> phase-m25 -> phase-m100 -> audit100 -> conditional phase-t250"
  exit 0
fi

common=(--checkpoint-dir "$MODEL" --cache-root "$CACHE" --manifest "$MANIFEST" --data-receipt "$DATA/receipt.json" --replay "$DATA/replay250.jsonl" --audit-manifest "$DATA/audit20.jsonl" --negative-root "$NEGATIVE" --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')" --probe-checkpoint "$PROBE" --probe-split "$SPLIT" --trajectory-calibration "$TRAJ_CAL" --observability-root "$OBS" --output-dir "$RUN")

if nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -q "$(nvidia-smi -i 6 --query-gpu=uuid --format=csv,noheader)"; then
  echo "GPU6 has a compute process; refusing to launch" >&2; exit 1
fi
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:$SOURCE:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"
run() { "$PYTHON" scripts/train_wan_v8_direct_action_band.py "${common[@]}" "$@"; }
case "$PHASE" in
  preflight) run --mode preflight ;;
  smoke) run --mode smoke ;;
  train25) run --mode phase-m --target-step 25 ;;
  train50) run --mode phase-m --target-step 50 --resume "$RUN/step-000025.pt" ;;
  train100) run --mode phase-m --target-step 100 --resume "$RUN/step-000050.pt" ;;
  audit100) run --mode audit100 --resume "$RUN/step-000100.pt" ;;
  train150) run --mode phase-t --target-step 150 --resume "$RUN/step-000100-gated.pt" ;;
  train200) run --mode phase-t --target-step 200 --resume "$RUN/step-000150.pt" ;;
  train250) run --mode phase-t --target-step 250 --resume "$RUN/step-000200.pt" ;;
  audit250) run --mode audit250 --resume "$RUN/step-000250.pt" ;;
  *) echo "unsupported v8 phase: $PHASE" >&2; exit 2 ;;
esac
