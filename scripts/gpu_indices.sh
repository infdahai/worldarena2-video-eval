#!/usr/bin/env bash

parse_gpu_indices() {
  local gpu_indices="$1"
  local gpu

  IFS=',' read -r -a GPU_LIST <<< "$gpu_indices"
  if [ "${#GPU_LIST[@]}" -eq 0 ] || [ -z "${GPU_LIST[0]}" ]; then
    echo "GPU_INDICES must contain at least one GPU index" >&2
    return 2
  fi
  for gpu in "${GPU_LIST[@]}"; do
    if ! [[ "$gpu" =~ ^[0-9]+$ ]]; then
      echo "invalid GPU index in GPU_INDICES: $gpu" >&2
      return 2
    fi
  done
}
