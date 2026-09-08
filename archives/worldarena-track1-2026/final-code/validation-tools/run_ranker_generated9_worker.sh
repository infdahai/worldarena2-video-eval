#!/usr/bin/env bash
# One bounded holdout campaign; wait for this card's generation queue, not all cards.
set -euo pipefail
H=$1; gpu=$2; uuid=$3
A=/data/di/worldarena2_track1_20260815
test "$H" = "$A/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903"
[[ "$gpu" =~ ^[0-6]$ ]]
exec 6>"$H/claims/score-worker$gpu.lock"
flock -n 6 || exit 75
echo "WAIT_GENERATION $(date -Is) gpu=$gpu uuid=$uuid"
exec 9>"$H/claims/gpu$gpu.lock"
flock 9
exec 7>"$A/official_track1_eval/tmp/gpu$gpu.track1-p0-parallel.lock"
flock 7
export PYTHONPATH="/home/huazhi/nlh/baseline/src:$H/tools"
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
tasks=(open_microwave press_stapler put_object_cabinet stack_bowls_three)
while :; do
  completed=0; did_work=0
  for seed in 1 4 2 3; do
    for shard in 0 1 2 3; do
      task=${tasks[$shard]}; job="$task.seed$seed"
      receipt="$H/generated9/seed$seed/shard$shard/package/receipts/holdout16-generated9.complete.json"
      if [[ -s "$receipt" ]]; then completed=$((completed + 1)); continue; fi
      test ! -e "$H/claims/score-$job.failed" || { echo "REFUSED prior failure $job"; exit 1; }
      [[ -s "$H/generation/seed$seed/$task/stage1-only.receipt.json" ]] || continue
      exec 8>"$H/claims/score-$job.lock"
      if ! flock -n 8; then exec 8>&-; continue; fi
      if [[ -s "$receipt" ]]; then exec 8>&-; continue; fi
      test "$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)" = "$uuid"
      apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
      if [[ -n "$apps" ]]; then echo "REFUSED occupied gpu=$gpu pids=$apps"; exit 75; fi
      echo "START $(date -Is) gpu=$gpu job=$job"
      if ! "$A/envs/flowwam-v16/bin/python" -B "$H/tools/score_holdout16_generated9_mvp.py" \
        --seed "$seed" --shard "$shard" --source-root "$H/generation/seed$seed/$task" \
        --score-root "$H/generated9" --manifest "$H/manifest.jsonl" --artifact-root "$A" \
        --dataset "$H/input-staging/input" --ffmpeg "$A/bin/ffmpeg" --gpu "$gpu" \
        >"$H/logs/generated9-$job.log" 2>&1; then
        printf '%s gpu=%s job=%s\n' "$(date -Is)" "$gpu" "$job" >"$H/claims/score-$job.failed"
        echo "FAILED $(date -Is) gpu=$gpu job=$job"; exit 1
      fi
      test -s "$receipt"
      echo "DONE $(date -Is) gpu=$gpu job=$job"
      did_work=1
      exec 8>&-
    done
  done
  if (( completed == 16 )); then echo "ALL_DONE $(date -Is)"; exit 0; fi
  if (( did_work == 0 )); then sleep 15; fi
done
