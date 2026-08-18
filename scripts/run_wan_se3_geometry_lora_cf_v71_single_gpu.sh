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
RUN=$ROOT/runs/v71-se3-geometry-lora-cf-single-gpu/mechanism
SMOKE=$ROOT/runs/v71-se3-geometry-lora-cf-single-gpu/smoke
PREFLIGHT=$RUN/preflight-gradient-audit.json
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=6" \
    "architecture=v71-geometry-lora-cf" \
    "fresh-init-from-clean-gated-step10" \
    "blocks=8,16,24 rank=16 q/k/v/o geometry-only" \
    "counterfactual=geometry-only-counterfactual; frozen raster remains correct" \
    "loss=weighted-fm + calibrated smooth pairwise ranking" \
    "negative cycle: reverse -> shift+1 -> swap -> reverse -> shift-1 -> swap" \
    "step25 requires 12/20 for reverse,shift,swap" \
    "step50 requires 14/20 for reverse,shift,swap" \
    "preflight -> smoke -> parent-audit -> train10/audit10 -> train25/audit25 -> conditional train50/audit50"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"

common=(
  --architecture v71-geometry-lora-cf --topology single-gpu
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
    --repo-root "$SOURCE" --output "$output_root/source-sync-closure.v71-cf.json" >/dev/null
}

gate() {
  local step=$1
  "$PYTHON" - "$RUN/audit-step-000000.json" "$RUN/audit-step-$(printf '%06d' "$step").json" "$step" <<'PY'
import json, math, sys
parent = json.load(open(sys.argv[1], encoding="utf-8"))
candidate = json.load(open(sys.argv[2], encoding="utf-8"))
step = int(sys.argv[3])
pm = parent["metrics"]
cm = candidate["metrics"]
minimum = 12 if step == 25 else 14
for name in ("reverse", "shift", "swap"):
    value = cm["counterfactual"][name]
    if value["finite_pairs"] != 20 or value["wins"] < minimum:
        raise SystemExit(f"step{step} {name} wins gate failed")
    if not cm["average_ranking_margin"][name] > 0:
        raise SystemExit(f"step{step} {name} ranking margin gate failed")
def mean(report, metric):
    return sum(row["metrics"]["correct"][metric] for row in report["episodes"]) / 20
position = (mean(parent, "position_error") - mean(candidate, "position_error")) / mean(parent, "position_error")
velocity = (mean(parent, "velocity_error") - mean(candidate, "velocity_error")) / mean(parent, "velocity_error")
fm = (mean(candidate, "fm_loss") - mean(parent, "fm_loss")) / mean(parent, "fm_loss")
if fm > 0.02:
    raise SystemExit(f"step{step} correct FM regression gate failed: {fm}")
if step == 25 and (position < 0 or velocity < 0):
    raise SystemExit(f"step25 parent-relative probe gate failed: pos={position} vel={velocity}")
if step == 50 and (position <= 0.05 or velocity <= 0.05):
    raise SystemExit(f"step50 5-percent probe gate failed: pos={position} vel={velocity}")
if step == 50 and cm.get("routing_retention", 0) < 0.9:
    raise SystemExit("step50 routing retention gate failed")
print(json.dumps({"step": step, "position_improvement": position, "velocity_improvement": velocity, "fm_regression": fm}, sort_keys=True))
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
  audit0)
    validate_source_closure "$RUN"
    run --mode audit --audit-set retirement20 --audit-zero-gate \
      --audit-output "$RUN/audit-step-000000.json" --output-dir "$RUN"
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
  gate25)
    gate 25
    ;;
  train50)
    validate_source_closure "$RUN"
    gate 25
    run --mode train --target-step 50 --resume "$RUN/step-000025.pt" --output-dir "$RUN"
    ;;
  audit50)
    validate_source_closure "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000050.pt" \
      --audit-output "$RUN/audit-step-000050.json" --output-dir "$RUN"
    ;;
  gate50)
    gate 50
    ;;
  *)
    echo "unknown v7.1-CF phase: $PHASE" >&2
    exit 2
    ;;
esac
