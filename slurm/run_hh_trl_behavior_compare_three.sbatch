#!/bin/bash
#SBATCH -J hh_trl_cmp3
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:3
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/hh_length_pipeline}"
COMPARE_OUTDIR="${COMPARE_OUTDIR:-$PIPELINE_DIR/trl_ppo_canary_512_deberta_threeway_behavior_compare}"
BASELINE_POLICY="${BASELINE_POLICY:-$PIPELINE_DIR/trl_ppo_canary_512_deberta_correct_only_r4/deberta_baseline/final}"
LENGTH_PENALTY_POLICY="${LENGTH_PENALTY_POLICY:-$PIPELINE_DIR/trl_ppo_canary_512_deberta_lenpen_r1/deberta_lenpen_baseline/final}"
LENGTH_NULLED_POLICY="${LENGTH_NULLED_POLICY:-$PIPELINE_DIR/trl_ppo_canary_512_deberta_correct_only_r4/deberta_length_nulled/final}"
EVAL_FILE="${EVAL_FILE:-$PIPELINE_DIR/eval/test.jsonl}"
REWARD_MODEL="${REWARD_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}"
PROBE_FILE="${PROBE_FILE:-$REPO/artifacts/probes/length/length_deberta_gsm8k_correct_only/probe.pt}"
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

run_rm_eval() {
  local gpu_id="$1"
  local policy_path="$2"
  local outdir="$3"
  CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/evaluate_assistant_rm.py \
    --policy-model "$policy_path" \
    --reward-model "$REWARD_MODEL" \
    --probe-file "$PROBE_FILE" \
    --eval-file "$EVAL_FILE" \
    --output-dir "$outdir" \
    --limit "${EVAL_LIMIT:-512}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --batch-size "${RM_BATCH_SIZE:-8}" \
    --generation-batch-size "${GENERATION_BATCH_SIZE:-8}" \
    --policy-device cuda:0 \
    --reward-device cuda:0 \
    --seed "${SEED:-0}"
}

run_len_eval() {
  local gpu_id="$1"
  local policy_path="$2"
  local outdir="$3"
  CUDA_VISIBLE_DEVICES="$gpu_id" python scripts/measure_generation_length.py \
    --policy-model "$policy_path" \
    --eval-file "$EVAL_FILE" \
    --output-dir "$outdir" \
    --limit "${LENGTH_LIMIT:-512}" \
    --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
    --batch-size "${LENGTH_BATCH_SIZE:-16}"
}

run_rm_eval 0 "$BASELINE_POLICY" "$COMPARE_OUTDIR/baseline_rm_eval" &
pid_rm_baseline=$!
run_rm_eval 1 "$LENGTH_PENALTY_POLICY" "$COMPARE_OUTDIR/length_penalty_rm_eval" &
pid_rm_penalty=$!
run_rm_eval 2 "$LENGTH_NULLED_POLICY" "$COMPARE_OUTDIR/length_nulled_rm_eval" &
pid_rm_nulled=$!

wait "$pid_rm_baseline" "$pid_rm_penalty" "$pid_rm_nulled"

run_len_eval 0 "$BASELINE_POLICY" "$COMPARE_OUTDIR/baseline_length" &
pid_len_baseline=$!
run_len_eval 1 "$LENGTH_PENALTY_POLICY" "$COMPARE_OUTDIR/length_penalty_length" &
pid_len_penalty=$!
run_len_eval 2 "$LENGTH_NULLED_POLICY" "$COMPARE_OUTDIR/length_nulled_length" &
pid_len_nulled=$!

wait "$pid_len_baseline" "$pid_len_penalty" "$pid_len_nulled"

python scripts/summarize_behavior_comparison_three.py \
  --baseline-rm-summary "$COMPARE_OUTDIR/baseline_rm_eval/summary.json" \
  --baseline-length-summary "$COMPARE_OUTDIR/baseline_length/summary.json" \
  --length-penalty-rm-summary "$COMPARE_OUTDIR/length_penalty_rm_eval/summary.json" \
  --length-penalty-length-summary "$COMPARE_OUTDIR/length_penalty_length/summary.json" \
  --length-nulled-rm-summary "$COMPARE_OUTDIR/length_nulled_rm_eval/summary.json" \
  --length-nulled-length-summary "$COMPARE_OUTDIR/length_nulled_length/summary.json" \
  --output-file "$COMPARE_OUTDIR/comparison_summary.json"
