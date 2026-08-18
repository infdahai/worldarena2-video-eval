#!/usr/bin/env bash
set -euo pipefail

# Isolated opportunistic lineage.  This launcher intentionally has no switch
# for other devices: physical GPU6 is the sole permitted execution target.
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
RUN=$ROOT/runs/v7-se3-single-gpu/mechanism
SMOKE=$ROOT/runs/v7-se3-single-gpu/smoke
PREFLIGHT=$RUN/preflight-gradient-audit.json
PHASE=${1:-dry-run}

if [[ "$PHASE" == dry-run ]]; then
  printf '%s\n' \
    "CUDA_VISIBLE_DEVICES=6" \
    "torchrun --nproc_per_node=1 --topology single-gpu" \
    "lineage -> v7-se3-single-gpu" \
    "preflight -> $PREFLIGHT" \
    "smoke -> $SMOKE/production-smoke.v7.json" \
    "train10 -> $RUN/step-000010.pt" \
    "train25 -> $RUN/step-000025.pt" \
    "retirement-audit20 -> zero-gate / step10 / step25" \
    "gate-only step50 is retired and cannot be launched here"
  exit 0
fi

export CUDA_VISIBLE_DEVICES=6
export PYTHONPATH=$SOURCE/src:/home/huazhi/nlh/Wan2.2
cd "$SOURCE"

common=(
  --topology single-gpu
  --checkpoint-dir "$MODEL" --cache-root "$CACHE" --manifest "$MANIFEST"
  --data-source-manifest "$SOURCE_MANIFEST" --data-leakage-receipt "$LEAKAGE"
  --discovery-manifest "$DISCOVERY" --dev-fast20-manifest "$DEV_FAST20"
  --v6-replay "$V6_REPLAY" --v7-replay "$V7_REPLAY"
  --parent-checkpoint "$PARENT" --base-parent-sha256 "$(sha256sum "$BASE" | awk '{print $1}')"
  --probe-checkpoint "$PROBE" --probe-split "$SPLIT" --observability-root "$OBS"
  --preflight-receipt "$PREFLIGHT" --seed 20260818
)

build_replay() {
  "$PYTHON" scripts/build_wan_v7_replay.py \
    --topology single-gpu --v6-replay "$V6_REPLAY" \
    --clean-1000-manifest "$SOURCE_MANIFEST" --output "$V7_REPLAY"
}

run() {
  "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=1 \
    scripts/train_wan_se3_probe_v7_fsdp.py "${common[@]}" "$@"
}

validate_source_closure() {
  local output_root=$1
  local receipt=${2:-}
  local closure_receipt=$output_root/source-sync-closure.v7.json
  mkdir -p "$output_root"
  "$PYTHON" scripts/validate_wan_v7_sync_closure.py \
    --repo-root "$SOURCE" --output "$closure_receipt" >/dev/null
  if [[ -n "$receipt" ]]; then
    "$PYTHON" - "$receipt" "$closure_receipt" <<'PY'
import json
import sys

preflight = json.load(open(sys.argv[1], encoding="utf-8"))
closure = json.load(open(sys.argv[2], encoding="utf-8"))
expected = preflight.get("source_hashes", {}).get("source_code_sha256")
actual = closure.get("closure_sha256")
if not isinstance(expected, str) or expected != actual:
    raise SystemExit("v7 single-gpu preflight receipt source closure digest mismatch")
PY
  fi
}

case "$PHASE" in
  preflight)
    validate_source_closure "$RUN"
    build_replay
    run --mode preflight --output-dir "$RUN"
    ;;
  smoke)
    validate_source_closure "$SMOKE"
    build_replay
    run --mode smoke --output-dir "$SMOKE"
    ;;
  train10)
    validate_source_closure "$RUN" "$PREFLIGHT"
    build_replay
    run --mode train --target-step 10 --output-dir "$RUN"
    ;;
  train25)
    validate_source_closure "$RUN" "$PREFLIGHT"
    build_replay
    run --mode train --target-step 25 --resume "$RUN/step-000010.pt" --output-dir "$RUN"
    ;;
  audit20)
    validate_source_closure "$RUN" "$PREFLIGHT"
    build_replay
    run --mode audit --audit-set retirement20 --audit-zero-gate \
      --audit-output "$RUN/retirement-audit20-zero-gate.json" --output-dir "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000010.pt" \
      --audit-output "$RUN/retirement-audit20-step10.json" --output-dir "$RUN"
    run --mode audit --audit-set retirement20 --resume "$RUN/step-000025.pt" \
      --audit-output "$RUN/retirement-audit20-step25.json" --output-dir "$RUN"
    ;;
  *)
    echo "unknown v7 single-gpu phase: $PHASE" >&2
    exit 2
    ;;
esac
