#!/bin/bash
#SBATCH -J trl_rm_ppo
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --account=aal
#SBATCH --partition=aal
#SBATCH --nodelist=aal1

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:?PIPELINE_DIR required}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
POLICY_MODEL="${POLICY_MODEL:-meta-llama/Llama-3.2-1B-Instruct}"
REWARD_MODEL="${REWARD_MODEL:?REWARD_MODEL required}"
PROBE_FILE="${PROBE_FILE:-}"
VARIANT="${VARIANT:?VARIANT required}"
REWARD_TAG="${REWARD_TAG:?REWARD_TAG required}"
PPO_OUTDIR_ROOT="${PPO_OUTDIR_ROOT:?PPO_OUTDIR_ROOT required}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"

OUTDIR="$PPO_OUTDIR_ROOT/${REWARD_TAG}_${VARIANT}"

mkdir -p "$REPO/logs" "$OUTDIR" "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$TRITON_CACHE_DIR"
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
export XDG_CACHE_HOME HF_HOME HUGGINGFACE_HUB_CACHE HF_DATASETS_CACHE TRANSFORMERS_CACHE TRITON_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export TRL_EXPERIMENTAL_SILENCE=1

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

python - <<'PY'
import importlib.util
missing = [m for m in ("trl", "datasets", "transformers") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing required packages in current env: {missing}")
print("Verified TRL runtime imports.")
PY

LAUNCH_ARGS=(
  --train-file "$PIPELINE_DIR/ppo/train.jsonl"
  --eval-file "$PIPELINE_DIR/ppo/val.jsonl"
  --policy-model "$POLICY_MODEL"
  --reward-model "$REWARD_MODEL"
  --output-dir "$OUTDIR"
  --experiment-name "${REWARD_TAG}_${VARIANT}"
  --reward-dtype "${REWARD_DTYPE:-bfloat16}"
  --train-steps "${TRAIN_STEPS:-512}"
  --max-prompt-length "${MAX_PROMPT_LENGTH:-768}"
  --per-device-train-batch-size "${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
  --per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS:-1}"
  --num-mini-batches "${NUM_MINI_BATCHES:-1}"
  --local-rollout-forward-batch-size "${ROLLOUT_FORWARD_BATCH_SIZE:-1}"
  --temperature "${TEMPERATURE:-0.7}"
  --top-p "${TOP_P:-0.95}"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
  --learning-rate "${LR:-5e-7}"
  --num-ppo-epochs "${NUM_PPO_EPOCHS:-2}"
  --max-grad-norm "${MAX_GRAD_NORM:-0.3}"
  --kl-coef "${KL_COEFF:-0.08}"
  --eval-steps "${EVAL_EVERY:-64}"
  --save-steps "${SAVE_STEPS:-64}"
  --logging-steps "${LOGGING_STEPS:-10}"
  --attn-implementation "$ATTN_IMPLEMENTATION"
)

if [[ -n "${LENGTH_PENALTY_MAX_LEN:-}" ]]; then
  LAUNCH_ARGS+=(--length-penalty-max-len "${LENGTH_PENALTY_MAX_LEN}")
fi
if [[ -n "${LENGTH_PENALTY_EMA:-}" ]]; then
  LAUNCH_ARGS+=(--length-penalty-ema "${LENGTH_PENALTY_EMA}")
fi
if [[ "${REWARD_LOAD_IN_4BIT:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--reward-load-in-4bit)
fi
if [[ "${REWARD_LOAD_IN_8BIT:-0}" == "1" ]]; then
  LAUNCH_ARGS+=(--reward-load-in-8bit)
fi

if [[ "$VARIANT" != "baseline" && -n "$PROBE_FILE" ]]; then
  LAUNCH_ARGS+=(--probe-file "$PROBE_FILE")
fi

python scripts/train_trl_rm_ppo.py "${LAUNCH_ARGS[@]}"
