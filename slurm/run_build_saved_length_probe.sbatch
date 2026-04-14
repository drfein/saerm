#!/bin/bash
#SBATCH -J build_len_probe
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
CANDIDATE_FILE="${CANDIDATE_FILE:?CANDIDATE_FILE required}"
REWARD_MODEL="${REWARD_MODEL:?REWARD_MODEL required}"
OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR required}"

mkdir -p "$REPO/logs" "$OUTPUT_DIR"
cd "$REPO"

if [[ -f /sailhome/drfein/miniconda3/etc/profile.d/conda.sh ]]; then
  source /sailhome/drfein/miniconda3/etc/profile.d/conda.sh
elif [[ -f /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh ]]; then
  source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
else
  source /sailhome/drfein/miniconda3/bin/activate "$CONDA_ENV_NAME"
fi
conda activate "$CONDA_ENV_NAME" 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
export HF_HOME="${HF_HOME:-/aal/u/drfein/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python3 scripts/build_length_probe_from_saved_candidates.py \
  --candidate-file "$CANDIDATE_FILE" \
  --reward-model "$REWARD_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --batch-size "${RM_BATCH_SIZE:-4}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --torch-dtype "${TORCH_DTYPE:-bfloat16}" \
  --device "${DEVICE:-cuda}"
