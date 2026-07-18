#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/data/di/worldarena2_track1_baseline}"
BASE_PYTHON="${BASE_PYTHON:-/data/zlj/pi0_training/envs/lerobot/bin/python}"
VENV="$ROOT/venv"
PYTHON="$VENV/bin/python"
PIP="$PYTHON -m pip"
HF="$VENV/bin/hf"
LOGS="$ROOT/logs"
DOWNLOADS="$ROOT/downloads"
OSCAR="$ROOT/oscar-public"
OSCAR_COMMIT="4dea2f657e221b0ff24c895fcc8ab4d46d5a9adb"
export HF_HOME="${HF_HOME:-/data/di/hf_cache}"

mkdir -p "$ROOT" "$LOGS" "$DOWNLOADS" "$ROOT/data" "$ROOT/assets" "$ROOT/controls"

fail() {
  code=$?
  echo "$(date -Is) bootstrap failed with exit code $code" | tee -a "$LOGS/bootstrap.log"
  touch "$ROOT/SETUP_FAILED"
  exit "$code"
}
trap fail ERR
rm -f "$ROOT/SETUP_FAILED"

if [ ! -x "$PYTHON" ]; then
  "$BASE_PYTHON" -m venv "$VENV"
fi

$PIP install --upgrade pip setuptools wheel
$PIP install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu126

if [ ! -d "$OSCAR/.git" ]; then
  git clone https://github.com/wuzy2115/oscar-public.git "$OSCAR"
fi
git -C "$OSCAR" fetch --depth 1 origin "$OSCAR_COMMIT"
git -C "$OSCAR" checkout --detach "$OSCAR_COMMIT"
cp -R "$ROOT/baseline/compat/transformer_engine" "$OSCAR/"

$PIP install -r "$OSCAR/requirements_minimal.txt"
$PIP install -e "$ROOT/baseline"

if [ ! -f "$DOWNLOADS/dataset_track1.tar.gz" ]; then
  "$HF" download WorldArena/WorldArena2.0 dataset_track1.tar.gz \
    --repo-type dataset --local-dir "$DOWNLOADS"
fi
if [ ! -d "$ROOT/data/dataset_track1" ]; then
  tar -xzf "$DOWNLOADS/dataset_track1.tar.gz" -C "$ROOT/data"
fi

if [ ! -f "$DOWNLOADS/embodiments.zip" ]; then
  "$HF" download TianxingChen/RoboTwin2.0 embodiments.zip \
    --repo-type dataset --local-dir "$DOWNLOADS"
fi
if [ ! -f "$ROOT/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf" ]; then
  unzip -q "$DOWNLOADS/embodiments.zip" \
    "embodiments/aloha-agilex/config.yml" \
    "embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf" \
    -d "$ROOT/assets"
fi

"$PYTHON" -m worldarena_baseline.cli inspect \
  --dataset-root "$ROOT/data/dataset_track1" | tee "$ROOT/dataset-report.json"
"$PYTHON" -m worldarena_baseline.cli prepare \
  --dataset-root "$ROOT/data/dataset_track1" \
  --urdf "$ROOT/assets/embodiments/aloha-agilex/urdf/arx5_description_isaac.urdf" \
  --controls-dir "$ROOT/controls" | tee "$LOGS/prepare-controls.log"

if [ ! -f "$ROOT/checkpoints/model/.metadata" ]; then
  "$HF" download zywu2115/OSCAR-2B --local-dir "$ROOT/checkpoints"
fi
if [ ! -f "$ROOT/cosmos-reason1-7b/config.json" ]; then
  "$HF" download nvidia/Cosmos-Reason1-7B \
    --local-dir "$ROOT/cosmos-reason1-7b"
fi

"$PYTHON" - <<'PY'
import torch
import transformer_engine

print({
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "transformer_engine": transformer_engine.__version__,
})
PY

touch "$ROOT/SETUP_COMPLETE"
echo "$(date -Is) bootstrap complete" | tee -a "$LOGS/bootstrap.log"

if ! tmux has-session -t worldarena2-campaign 2>/dev/null; then
  tmux new-session -d -s worldarena2-campaign \
    "ROOT=$ROOT HF_HOME=$HF_HOME bash $ROOT/baseline/scripts/campaign.sh"
fi
