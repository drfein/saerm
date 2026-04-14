#!/bin/bash
#SBATCH -J bon_multi_rm
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
CANDIDATE_FILE="${CANDIDATE_FILE:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
DATASET_NAME="${DATASET_NAME:-}"

if [[ -z "$CANDIDATE_FILE" || -z "$OUTPUT_DIR" || -z "$DATASET_NAME" ]]; then
  echo "Missing required env vars:"
  echo "  CANDIDATE_FILE=${CANDIDATE_FILE:-<unset>}"
  echo "  OUTPUT_DIR=${OUTPUT_DIR:-<unset>}"
  echo "  DATASET_NAME=${DATASET_NAME:-<unset>}"
  exit 2
fi

mkdir -p "$REPO/logs"
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
export TOKENIZERS_PARALLELISM=false
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
export HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

python scripts/score_saved_candidates_multi_rm.py \
  --candidate-file "$CANDIDATE_FILE" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-name "$DATASET_NAME" \
  --deberta-model "${DEBERTA_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}" \
  --deberta-probe-file "${DEBERTA_PROBE_FILE:-$REPO/artifacts/probes/length/length_deberta_gsm8k_correct_only/probe.pt}" \
  --allen-model "${ALLEN_MODEL:-allenai/Llama-3.1-8B-Instruct-RM-RB2}" \
  --allen-probe-file "${ALLEN_PROBE_FILE:-$REPO/artifacts/probes/length/length_allen_gsm8k/probe.pt}" \
  --batch-size "${RM_BATCH_SIZE:-4}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --alpha "${ALPHA:-1.0}" \
  --n-values "${N_VALUES:-1,2,4,8,16,32,64}" \
  --device "${DEVICE:-cuda}"
