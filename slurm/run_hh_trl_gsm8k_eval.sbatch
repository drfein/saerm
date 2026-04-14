#!/bin/bash
#SBATCH -J hh_trl_gsm8k
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
HH_PIPELINE_DIR="${HH_PIPELINE_DIR:-$ARTIFACT_ROOT/hh_length_pipeline}"
PPO_ROOT="${PPO_ROOT:-$HH_PIPELINE_DIR/trl_ppo_canary_512}"
COMPARE_OUTDIR="${COMPARE_OUTDIR:-$HH_PIPELINE_DIR/trl_ppo_canary_512_gsm8k_eval}"
BASELINE_POLICY="${BASELINE_POLICY:-$PPO_ROOT/skywork_baseline/final}"
NULLED_POLICY="${NULLED_POLICY:-$PPO_ROOT/skywork_length_nulled/final}"
GSM8K_EVAL_FILE="${GSM8K_EVAL_FILE:-$ARTIFACT_ROOT/gsm8k_length_pipeline/eval/test.jsonl}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"

mkdir -p "$REPO/logs" "$COMPARE_OUTDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR"
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
export TRITON_CACHE_DIR
export TOKENIZERS_PARALLELISM=false

for ENV_FILE in "$REPO/.env" "/sailhome/drfein/.env" "/sailhome/drfein/em/.env"; do
  if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
  fi
done
if [[ -z "${HF_TOKEN:-}" && -n "${HF_KEY:-}" ]]; then
  export HF_TOKEN="$HF_KEY"
fi
if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi

echo "[job] host=$(hostname)"
echo "[job] compare_outdir=$COMPARE_OUTDIR"
echo "[job] baseline_policy=$BASELINE_POLICY"
echo "[job] nulled_policy=$NULLED_POLICY"
echo "[job] gsm8k_eval_file=$GSM8K_EVAL_FILE"

python scripts/evaluate_gsm8k_policy.py \
  --model-id "$BASELINE_POLICY" \
  --eval-file "$GSM8K_EVAL_FILE" \
  --output-dir "$COMPARE_OUTDIR/baseline_gsm8k_eval" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --seed "${SEED:-0}" \
  ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"}

python scripts/evaluate_gsm8k_policy.py \
  --model-id "$NULLED_POLICY" \
  --eval-file "$GSM8K_EVAL_FILE" \
  --output-dir "$COMPARE_OUTDIR/length_nulled_gsm8k_eval" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --temperature "${TEMPERATURE:-0.0}" \
  --top-p "${TOP_P:-1.0}" \
  --seed "${SEED:-0}" \
  ${EVAL_LIMIT:+--limit "$EVAL_LIMIT"}

python scripts/summarize_gsm8k_policy_comparison.py \
  --baseline-summary "$COMPARE_OUTDIR/baseline_gsm8k_eval/summary.json" \
  --nulled-summary "$COMPARE_OUTDIR/length_nulled_gsm8k_eval/summary.json" \
  --output-file "$COMPARE_OUTDIR/comparison_summary.json"
