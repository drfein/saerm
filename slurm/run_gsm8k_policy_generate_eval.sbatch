#!/bin/bash
#SBATCH -J gsm8k_eval
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
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
POLICY_MODEL="${POLICY_MODEL:?POLICY_MODEL required}"
EVAL_FILE="${EVAL_FILE:?EVAL_FILE required}"
OUTPUT_ROOT="${OUTPUT_ROOT:?OUTPUT_ROOT required}"
MODEL_NAME="${MODEL_NAME:?MODEL_NAME required}"

mkdir -p "$REPO/logs" "$OUTPUT_ROOT" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
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
export XDG_CACHE_HOME HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE

GEN_DIR="$OUTPUT_ROOT/generated"
EVAL_DIR="$OUTPUT_ROOT/gsm8k_eval"

python3 scripts/generate_policy_responses.py \
  --policy-model "$POLICY_MODEL" \
  --eval-file "$EVAL_FILE" \
  --output-dir "$GEN_DIR" \
  --limit "${MAX_EXAMPLES:-512}" \
  --offset "${OFFSET:-0}" \
  --batch-size "${GEN_BATCH_SIZE:-8}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --max-length "${MAX_LENGTH:-2048}" \
  --dtype "${POLICY_DTYPE:-bfloat16}"

python3 scripts/evaluate_saved_gsm8k_generations.py \
  --input-file "$GEN_DIR/details.jsonl" \
  --output-dir "$EVAL_DIR"

python3 - <<'PY'
import json
import os
from pathlib import Path

output_root = Path(os.environ["OUTPUT_ROOT"])
gen = json.loads((output_root / "generated" / "summary.json").read_text())
acc = json.loads((output_root / "gsm8k_eval" / "summary.json").read_text())
summary = {
    "model_name": os.environ["MODEL_NAME"],
    "policy_model": os.environ["POLICY_MODEL"],
    "generated": gen,
    "gsm8k_eval": acc,
}
(output_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
