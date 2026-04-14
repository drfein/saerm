#!/bin/bash
#SBATCH -J hh_openai_wr
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
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
COMPARE_ROOT="${COMPARE_ROOT:-/aal/u/drfein/saerm_runs/hh_length_pipeline/trl_ppo_canary_512_behavior_compare}"
MODEL_A_FILE="${MODEL_A_FILE:-$COMPARE_ROOT/baseline_rm_eval/details.jsonl}"
MODEL_B_FILE="${MODEL_B_FILE:-$COMPARE_ROOT/length_nulled_rm_eval/details.jsonl}"
MODEL_A_NAME="${MODEL_A_NAME:-baseline}"
MODEL_B_NAME="${MODEL_B_NAME:-length_nulled}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
OUTPUT_DIR="${OUTPUT_DIR:-/aal/u/drfein/saerm_runs/hh_length_pipeline/trl_ppo_canary_512_openai_winrate}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
USER_INSTRUCTION="${USER_INSTRUCTION:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-256}"
REASONING_EFFORT="${REASONING_EFFORT:-}"

mkdir -p "$REPO/logs" "$OUTPUT_DIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"
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

for ENV_FILE in "$REPO/.env" "/sailhome/drfein/.env" "/sailhome/drfein/em/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
done

echo "[job] host=$(hostname)"
echo "[job] judge_model=$JUDGE_MODEL"
echo "[job] model_a_file=$MODEL_A_FILE"
echo "[job] model_b_file=$MODEL_B_FILE"
echo "[job] output_dir=$OUTPUT_DIR"

OPENAI_ARGS=(
  --model-a-file "$MODEL_A_FILE"
  --model-b-file "$MODEL_B_FILE"
  --model-a-name "$MODEL_A_NAME"
  --model-b-name "$MODEL_B_NAME"
  --model "$JUDGE_MODEL"
  --output-dir "$OUTPUT_DIR"
  --seed "${SEED:-0}"
  --max-examples "${MAX_EXAMPLES:-512}"
  --sleep-seconds "${SLEEP_SECONDS:-0.0}"
  --max-output-tokens "$MAX_OUTPUT_TOKENS"
)
if [[ -n "$SYSTEM_PROMPT" ]]; then
  OPENAI_ARGS+=(--system-prompt "$SYSTEM_PROMPT")
fi
if [[ -n "$USER_INSTRUCTION" ]]; then
  OPENAI_ARGS+=(--user-instruction "$USER_INSTRUCTION")
fi
if [[ -n "$REASONING_EFFORT" ]]; then
  OPENAI_ARGS+=(--reasoning-effort "$REASONING_EFFORT")
fi

python3 scripts/evaluate_openai_win_rate.py \
  "${OPENAI_ARGS[@]}"
