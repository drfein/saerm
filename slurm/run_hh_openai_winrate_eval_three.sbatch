#!/bin/bash
#SBATCH -J hh_openai_wr3
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
COMPARE_ROOT="${COMPARE_ROOT:-/aal/u/drfein/saerm_runs/hh_length_pipeline/trl_ppo_canary_512_deberta_threeway_behavior_compare}"
OUTPUT_DIR="${OUTPUT_DIR:-/aal/u/drfein/saerm_runs/hh_length_pipeline/trl_ppo_canary_512_deberta_threeway_openai_winrate}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5-mini}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

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

run_pair() {
  local model_a_file="$1"
  local model_b_file="$2"
  local model_a_name="$3"
  local model_b_name="$4"
  local outdir="$5"
  python3 scripts/evaluate_openai_win_rate.py \
    --model-a-file "$model_a_file" \
    --model-b-file "$model_b_file" \
    --model-a-name "$model_a_name" \
    --model-b-name "$model_b_name" \
    --model "$JUDGE_MODEL" \
    --output-dir "$outdir" \
    --seed "${SEED:-0}" \
    --max-examples "${MAX_EXAMPLES:-512}" \
    --sleep-seconds "${SLEEP_SECONDS:-0.0}"
}

run_pair \
  "$COMPARE_ROOT/baseline_rm_eval/details.jsonl" \
  "$COMPARE_ROOT/length_penalty_rm_eval/details.jsonl" \
  "baseline" \
  "length_penalty" \
  "$OUTPUT_DIR/baseline_vs_length_penalty"

run_pair \
  "$COMPARE_ROOT/baseline_rm_eval/details.jsonl" \
  "$COMPARE_ROOT/length_nulled_rm_eval/details.jsonl" \
  "baseline" \
  "length_nulled" \
  "$OUTPUT_DIR/baseline_vs_length_nulled"

run_pair \
  "$COMPARE_ROOT/length_penalty_rm_eval/details.jsonl" \
  "$COMPARE_ROOT/length_nulled_rm_eval/details.jsonl" \
  "length_penalty" \
  "length_nulled" \
  "$OUTPUT_DIR/length_penalty_vs_length_nulled"

python3 scripts/summarize_openai_winrate_roundrobin.py \
  --baseline-vs-length-penalty "$OUTPUT_DIR/baseline_vs_length_penalty/summary.json" \
  --baseline-vs-length-nulled "$OUTPUT_DIR/baseline_vs_length_nulled/summary.json" \
  --length-penalty-vs-length-nulled "$OUTPUT_DIR/length_penalty_vs_length_nulled/summary.json" \
  --output-file "$OUTPUT_DIR/comparison_summary.json"
