#!/bin/bash
#SBATCH -J gsm8k_len_bon
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
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/gsm8k_length_pipeline}"
SFT_OUTDIR="${SFT_OUTDIR:-$PIPELINE_DIR/checkpoints/llama31_8b_gsm8k_sft}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
POLICY_MODEL="${POLICY_MODEL:-$SFT_OUTDIR}"
REWARD_MODEL="${REWARD_MODEL:-Skywork/Skywork-Reward-V2-Llama-3.1-8B}"
PROBE_FILE="${PROBE_FILE:-$REPO/artifacts/probes/length/length_skywork_gsm8k/probe.pt}"
BON_OUTDIR="${BON_OUTDIR:-$PIPELINE_DIR/bon/skywork_baseline_vs_length_nulled}"
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

echo "[job] host=$(hostname)"
echo "[job] artifact_root=$ARTIFACT_ROOT"
echo "[job] policy_model=$POLICY_MODEL"
echo "[job] reward_model=$REWARD_MODEL"
echo "[job] probe_file=$PROBE_FILE"
echo "[job] output_dir=$BON_OUTDIR"

python scripts/evaluate_best_of_n.py \
  --policy-model "$POLICY_MODEL" \
  --reward-model "$REWARD_MODEL" \
  --probe-file "$PROBE_FILE" \
  --prompts "$PIPELINE_DIR/eval/test.jsonl" \
  --output-dir "$BON_OUTDIR" \
  --num-prompts "${NUM_PROMPTS:-1319}" \
  --n "${N_CANDIDATES:-8}" \
  --temperature "${TEMPERATURE:-0.8}" \
  --top-p "${TOP_P:-0.95}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-256}" \
  --seed "${SEED:-0}"
