#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/di/worldarena2_track1_20260815
SOURCE=/home/huazhi/nlh/baseline
PYTHON=$ROOT/venv_reuse/bin/python
MODEL=$ROOT/models/Wan2.2-TI2V-5B
CACHE=$ROOT/cache/v4-clean-1785-v3
MANIFEST=$CACHE/manifests/clean-1000.cached.v7-se3.jsonl
SOURCE_MANIFEST=$ROOT/data_selection/v4-clean-scale-20260817-r3/clean-1000.jsonl
LEAKAGE=$ROOT/data_selection/v4-clean-scale-20260817-r3/clean-data-scale-receipt.json
DISCOVERY=$ROOT/eval/v7-se3-discovery-8/discovery-8.jsonl
DEV_FAST20=$ROOT/eval/dev-fast-20-v3/dev-fast-20.jsonl
V6_REPLAY=$ROOT/replay/v6-gripper-clean1000-ws7.json
V7_REPLAY=$ROOT/replay/v7-se3-clean1000-ws7.json
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
PROBE=$ROOT/probes/v6-gripper/gripper-probe.pt
SPLIT=$ROOT/probes/v6-gripper/clean1785-probe-split.json
OBS=$ROOT/probes/v6-gripper/observability-video-v2
RUN=$ROOT/runs/v7-se3-mechanism
SMOKE=$ROOT/runs/v7-se3-mechanism-smoke
PREFLIGHT=$RUN/preflight-gradient-audit.json
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6" \
    "preflight -> $PREFLIGHT" \
    "smoke -> $SMOKE/production-smoke.v7.json" \
    "train10 -> $RUN/step-000010.pt" \
    "train25 -> $RUN/step-000025.pt" \
    "audit25 -> $RUN/audit-step-000025.json" \
    "train50 requires audit25 pass" \
    "train50 -> $RUN/step-000050.pt" \
    "audit50 -> $RUN/audit-step-000050.json"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export PYTHONPATH=$SOURCE/src:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"

common=(
  --checkpoint-dir "$MODEL" --cache-root "$CACHE" --manifest "$MANIFEST"
  --data-source-manifest "$SOURCE_MANIFEST" --data-leakage-receipt "$LEAKAGE"
  --discovery-manifest "$DISCOVERY" --dev-fast20-manifest "$DEV_FAST20"
  --v6-replay "$V6_REPLAY" --v7-replay "$V7_REPLAY"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --probe-checkpoint "$PROBE" --probe-split "$SPLIT" --observability-root "$OBS"
  --preflight-receipt "$PREFLIGHT" --seed 20260818
)

run() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=7 \
    scripts/train_wan_se3_probe_v7_fsdp.py "${common[@]}" "$@"
}

case "$PHASE" in
  preflight)
    mkdir -p "$RUN"
    run --mode preflight --output-dir "$RUN"
    ;;
  smoke)
    mkdir -p "$SMOKE"
    run --mode smoke --output-dir "$SMOKE"
    ;;
  train10)
    mkdir -p "$RUN"
    run --mode train --target-step 10 --output-dir "$RUN"
    ;;
  train25)
    mkdir -p "$RUN"
    run --mode train --target-step 25 --resume "$RUN/step-000010.pt" --output-dir "$RUN"
    ;;
  audit25)
    run --mode audit --resume "$RUN/step-000025.pt" --audit-output "$RUN/audit-step-000025.json" --output-dir "$RUN"
    ;;
  train50)
    "$PYTHON" -c 'import json,sys; from worldarena_baseline.wan_v7_training import v7_discovery_gate; p=json.load(open(sys.argv[1])); assert p.get("gate") == v7_discovery_gate(p.get("metrics", {})); assert p["gate"]["pass"] is True' "$RUN/audit-step-000025.json"
    run --mode train --target-step 50 --resume "$RUN/step-000025.pt" --output-dir "$RUN"
    ;;
  audit50)
    run --mode audit --resume "$RUN/step-000050.pt" --audit-output "$RUN/audit-step-000050.json" --output-dir "$RUN"
    ;;
  *)
    echo "unknown v7 phase: $PHASE" >&2
    exit 2
    ;;
esac
