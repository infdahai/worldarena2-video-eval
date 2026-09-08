#!/usr/bin/env bash
# Bounded copy of the existing holdout per-card lock workflow: generation only.
set -euo pipefail
H=$1; gpu=$2; uuid=$3
A=/data/di/worldarena2_track1_20260815
test "$H" = "$A/runs/flowwam-full15-gap-20260904/inference-ensemble-mvp-20260901/ranker-fresh16-native-20260903"
[[ "$gpu" =~ ^[0-6]$ ]]
exec 6>"$H/claims/generation-worker$gpu.lock"
flock -n 6 || exit 75
exec 9>"$H/claims/gpu$gpu.lock"
flock -n 9 || exit 75
exec 7>"$A/official_track1_eval/tmp/gpu$gpu.track1-p0-parallel.lock"
flock -n 7 || exit 75
export PYTHONPATH=/home/huazhi/nlh/baseline/src
export CUDA_VISIBLE_DEVICES="$gpu" TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
tasks=(open_microwave press_stapler put_object_cabinet stack_bowls_three)
job_index=0
for task in "${tasks[@]}"; do
  for seed in 1 4 2 3; do
    assigned=$((job_index % 7)); job_index=$((job_index + 1))
    (( assigned == gpu )) || continue
    job="$task.seed$seed"
    receipt="$H/generation/seed$seed/$task/stage1-only.receipt.json"
    test ! -e "$H/claims/$job.failed"
    test ! -e "$H/claims/$job.started.json"
    test ! -e "$receipt"
    exec 8>"$H/claims/$job.lock"
    flock -n 8 || exit 75
    test "$(nvidia-smi -i "$gpu" --query-gpu=uuid --format=csv,noheader)" = "$uuid"
    apps=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -n "$apps" ]]; then echo "REFUSED occupied gpu=$gpu pids=$apps"; exit 75; fi
    free=$(df -B1 --output=avail "$A" | tail -1 | tr -d ' ')
    used=$(du -sb "$H" | cut -f1)
    (( free >= 107374182400 && used < 21474836480 )) || exit 76
    printf '{"worker_pid":%s,"gpu":%s,"uuid":"%s","job":"%s","started_at":"%s"}\n' "$$" "$gpu" "$uuid" "$job" "$(date -Is)" >"$H/claims/$job.started.json"
    echo "START $(date -Is) pid=$$ gpu=$gpu uuid=$uuid job=$job"
    if ! "$A/envs/flowwam-v16/bin/python" -B /home/huazhi/nlh/baseline/scripts/run_flowwam_official_stage1.py \
      --artifact-root "$A" --official-source /home/huazhi/nlh/FlowWAM_WorldArena \
      --test-dataset-dir "$H/input-staging/input" --robot-only-dir "$H/input-staging/robot_only" \
      --output-dir "$H/generation/seed$seed/$task" --sample-manifest "$H/jobs/$job.jsonl" \
      --checkpoint "$A/models/FlowWAM/flowwam_worldarena_stage1.safetensors" \
      --local-model-path "$A/runs/flowwam-orb-20260904/model-layout" \
      --expected-count 4 --physical-gpu "$gpu" --seed "$seed" --flow-max-magnitude 20 \
      >"$H/logs/$job.log" 2>&1; then
      printf '%s gpu=%s job=%s\n' "$(date -Is)" "$gpu" "$job" >"$H/claims/$job.failed"
      echo "FAILED $(date -Is) gpu=$gpu job=$job"; exit 1
    fi
    test -s "$receipt"
    echo "DONE $(date -Is) gpu=$gpu job=$job"
    exec 8>&-
  done
done
echo "ALL_DONE $(date -Is) gpu=$gpu"
