#!/bin/bash
#SBATCH -J hh_bon_score
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/hh_length_pipeline}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
REWARD_MODEL="${REWARD_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}"
PROBE_FILE="${PROBE_FILE:-$REPO/artifacts/probes/length/length_deberta_gsm8k_correct_only/probe.pt}"
CANDIDATE_FILE="${CANDIDATE_FILE:-$PIPELINE_DIR/bon/candidate_pool.jsonl}"
BON_OUTDIR="${BON_OUTDIR:-$PIPELINE_DIR/bon/deberta_baseline_vs_lenpen_vs_nulled_saved}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$REPO/logs" "$BON_OUTDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
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
export XDG_CACHE_HOME
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM=false

python scripts/score_saved_candidates_three.py \
  --candidate-file "$CANDIDATE_FILE" \
  --reward-model "$REWARD_MODEL" \
  --probe-file "$PROBE_FILE" \
  --output-dir "$BON_OUTDIR" \
  --batch-size "${RM_BATCH_SIZE:-4}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --length-penalty-max-len "${LENGTH_PENALTY_MAX_LEN:-256}" \
  --length-penalty-scale "${LENGTH_PENALTY_SCALE:-1.0}" \
  --device cuda
