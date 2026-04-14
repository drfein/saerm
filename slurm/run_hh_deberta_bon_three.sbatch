#!/bin/bash
#SBATCH -J hh_deb_bon3
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/hh_length_pipeline}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
POLICY_MODEL="${POLICY_MODEL:-$PIPELINE_DIR/checkpoints/llama31_8b_hh_sft}"
REWARD_MODEL="${REWARD_MODEL:-OpenAssistant/reward-model-deberta-v3-large-v2}"
PROBE_FILE="${PROBE_FILE:-$REPO/artifacts/probes/length/length_deberta_gsm8k_correct_only/probe.pt}"
BON_OUTDIR="${BON_OUTDIR:-$PIPELINE_DIR/bon/deberta_baseline_vs_lenpen_vs_nulled_sft8b}"
PROMPTS_FILE="${PROMPTS_FILE:-$PIPELINE_DIR/eval/test.jsonl}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"

mkdir -p "$REPO/logs" "$BON_OUTDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR"
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

python scripts/evaluate_best_of_n_three.py \
  --policy-model "$POLICY_MODEL" \
  --reward-model "$REWARD_MODEL" \
  --probe-file "$PROBE_FILE" \
  --prompts "$PROMPTS_FILE" \
  --output-dir "$BON_OUTDIR" \
  --num-prompts "${NUM_PROMPTS:-512}" \
  --n "${N_CANDIDATES:-8}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --length-penalty-max-len "${LENGTH_PENALTY_MAX_LEN:-256}" \
  --length-penalty-scale "${LENGTH_PENALTY_SCALE:-1.0}" \
  --policy-device "${POLICY_DEVICE:-cuda:0}" \
  --reward-device "${REWARD_DEVICE:-cuda:1}" \
  --batch-size "${RM_BATCH_SIZE:-8}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-32}" \
  ${USE_VLLM:+--use-vllm} \
  --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --seed "${SEED:-0}"
