#!/bin/bash
#SBATCH -J hh_bon_gen
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
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/hh_length_pipeline}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
POLICY_MODEL="${POLICY_MODEL:-$PIPELINE_DIR/checkpoints/llama31_8b_hh_sft}"
PROMPTS_FILE="${PROMPTS_FILE:-$PIPELINE_DIR/eval/test.jsonl}"
CANDIDATE_FILE="${CANDIDATE_FILE:-$PIPELINE_DIR/bon/candidate_pool.jsonl}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"

mkdir -p "$REPO/logs" "$(dirname "$CANDIDATE_FILE")" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR"
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

USE_VLLM_FLAG=()
if [[ "${USE_VLLM:-0}" != "0" ]]; then
  USE_VLLM_FLAG+=(--use-vllm)
fi

echo "POLICY_MODEL=$POLICY_MODEL"
echo "PROMPTS_FILE=$PROMPTS_FILE"
echo "CANDIDATE_FILE=$CANDIDATE_FILE"
echo "USE_VLLM=${USE_VLLM:-0}"
echo "PROMPT_BATCH_SIZE=${PROMPT_BATCH_SIZE:-16}"
echo "MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}"

python scripts/generate_bon_candidates.py \
  --policy-model "$POLICY_MODEL" \
  --prompts "$PROMPTS_FILE" \
  --output-file "$CANDIDATE_FILE" \
  --offset "${OFFSET:-0}" \
  --num-prompts "${NUM_PROMPTS:-512}" \
  --prompt-key "${PROMPT_KEY:-prompt}" \
  --messages-key "${MESSAGES_KEY:-messages}" \
  --answer-key "${ANSWER_KEY:-answer}" \
  --n "${N_CANDIDATES:-8}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --prompt-batch-size "${PROMPT_BATCH_SIZE:-16}" \
  "${USE_VLLM_FLAG[@]}" \
  --vllm-tensor-parallel-size "${VLLM_TENSOR_PARALLEL_SIZE:-1}" \
  --vllm-gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.85}" \
  --vllm-max-model-len "${VLLM_MAX_MODEL_LEN:-4096}" \
  --seed "${SEED:-0}"
