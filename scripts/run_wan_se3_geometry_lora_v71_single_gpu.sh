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
V7_REPLAY=$ROOT/replay/v7-se3-clean1000-ws1-gpu6.json
PARENT=$ROOT/runs/v6-clean-gated-parent/clean-gated-step10.pt
BASE=$ROOT/runs/s1A_branch_0125_ws7/step-000125.pt
PROBE=$ROOT/probes/v6-gripper/gripper-probe.pt
SPLIT=$ROOT/probes/v6-gripper/clean1785-probe-split.json
OBS=$ROOT/probes/v6-gripper/observability-video-v2
RUN=$ROOT/runs/v71-se3-geometry-lora-single-gpu/mechanism
SMOKE=$ROOT/runs/v71-se3-geometry-lora-single-gpu/smoke
PREFLIGHT=$RUN/preflight-gradient-audit.json
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=6" \
    "architecture=v71-geometry-lora" \
    "blocks=8,16,24 rank=16 q/k/v/o geometry-only" \
    "loss=weighted-fm-only" \
    "preflight -> smoke -> train10 -> audit10 -> train25 -> audit25" \
    "train50 requires positive audit25" \
    "step100 is not launched until step50 improves"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"

common=(
  --architecture v71-geometry-lora --topology single-gpu
  --checkpoint-dir "$MODEL" --cache-root "$CACHE" --manifest "$MANIFEST"
  --data-source-manifest "$SOURCE_MANIFEST" --data-leakage-receipt "$LEAKAGE"
  --discovery-manifest "$DISCOVERY" --dev-fast20-manifest "$DEV_FAST20"
  --v6-replay "$V6_REPLAY" --v7-replay "$V7_REPLAY"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --probe-checkpoint "$PROBE" --probe-split "$SPLIT" --observability-root "$OBS"
  --preflight-receipt "$PREFLIGHT" --seed 20260818
)

run() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/train_wan_se3_probe_v7_fsdp.py "${common[@]}" "$@"
}

validate_source_closure() {
  local output_root=$1
  mkdir -p "$output_root"
  "$PYTHON" scripts/validate_wan_v7_sync_closure.py \
    --repo-root "$SOURCE" --output "$output_root/source-sync-closure.v71.json" >/dev/null
}

require_positive_audit25() {
  "$PYTHON" - "$RUN/audit-step-000025.json" <<'PY'
import json, math, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))["metrics"]
for name in ("reverse", "shift", "swap"):
    value = m["counterfactual"][name]
    if value["finite_pairs"] != 20 or value["wins"] < 11:
        raise SystemExit(f"v7.1 step25 has no positive {name} separation")
    if not (value["position_improvement"] > 0 or value["velocity_improvement"] > 0):
        raise SystemExit(f"v7.1 step25 has no positive {name} probe direction")
PY
}

case "$PHASE" in
  preflight)
    validate_source_closure "$RUN"
    run --mode preflight --output-dir "$RUN"
    ;;
  smoke)
    validate_source_closure "$SMOKE"
    run --mode smoke --output-dir "$SMOKE"
    ;;
  train10)
    validate_source_closure "$RUN"
    run --mode train --target-step 10 --output-dir "$RUN"
    ;;
  audit10)
    validate_source_closure "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000010.pt" \
      --audit-output "$RUN/audit-step-000010.json" --output-dir "$RUN"
    ;;
  train25)
    validate_source_closure "$RUN"
    run --mode train --target-step 25 --resume "$RUN/step-000010.pt" --output-dir "$RUN"
    ;;
  audit25)
    validate_source_closure "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000025.pt" \
      --audit-output "$RUN/audit-step-000025.json" --output-dir "$RUN"
    ;;
  train50)
    validate_source_closure "$RUN"
    require_positive_audit25
    run --mode train --target-step 50 --resume "$RUN/step-000025.pt" --output-dir "$RUN"
    ;;
  audit50)
    validate_source_closure "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000050.pt" \
      --audit-output "$RUN/audit-step-000050.json" --output-dir "$RUN"
    ;;
  *)
    echo "unknown v7.1 phase: $PHASE" >&2
    exit 2
    ;;
esac
