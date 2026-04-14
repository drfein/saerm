#!/bin/bash
#SBATCH -p aal
#SBATCH --account=aal
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH -J shp_qwen3_lenprobe
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

set -euo pipefail

cd /sailhome/drfein/saerm
mkdir -p logs

if [ -f /sailhome/drfein/miniconda3/etc/profile.d/conda.sh ]; then
  source /sailhome/drfein/miniconda3/etc/profile.d/conda.sh
  conda activate saerm
elif [ -f /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh ]; then
  source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
  conda activate saerm
else
  source /nlp/scr/drfein/miniconda3/bin/activate saerm
fi

export HF_HOME="${HF_HOME:-/aal/u/drfein/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$TRANSFORMERS_CACHE"

REWARD_MODEL="${REWARD_MODEL:-/nlp/scr/drfein/saerm/reward_models/shp-qwen3-rm/checkpoint-20982}"
OUTPUT_DIR="${OUTPUT_DIR:-/sailhome/drfein/saerm/artifacts/probes/length/length_shp_qwen3_shp_balanced}"
DATASET_ID="${DATASET_ID:-stanfordnlp/SHP}"
SPLIT="${SPLIT:-train}"
PAIRS_PER_GROUP="${PAIRS_PER_GROUP:-500}"
EVAL_PAIRS_PER_GROUP="${EVAL_PAIRS_PER_GROUP:-500}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"

python scripts/build_shp_length_probe.py \
  --reward-model "$REWARD_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --dataset-id "$DATASET_ID" \
  --split "$SPLIT" \
  --pairs-per-group "$PAIRS_PER_GROUP" \
  --eval-pairs-per-group "$EVAL_PAIRS_PER_GROUP" \
  --batch-size "$BATCH_SIZE" \
  --max-length "$MAX_LENGTH" \
  --device cuda
