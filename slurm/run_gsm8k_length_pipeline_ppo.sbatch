#!/bin/bash
#SBATCH -J gsm8k_len_ppo
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=80G
#SBATCH --time=48:00:00
#SBATCH --account=aal
#SBATCH --partition=aal
#SBATCH --array=0-1%1

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/aal/u/drfein/saerm_runs}"
PIPELINE_DIR="${PIPELINE_DIR:-$ARTIFACT_ROOT/gsm8k_length_pipeline}"
SFT_OUTDIR="${SFT_OUTDIR:-$PIPELINE_DIR/checkpoints/llama31_8b_gsm8k_sft}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
POLICY_MODEL="${POLICY_MODEL:-$SFT_OUTDIR}"
REWARD_MODEL="${REWARD_MODEL:-Skywork/Skywork-Reward-V2-Llama-3.1-8B}"
PROBE_FILE="${PROBE_FILE:-$REPO/artifacts/probes/length/length_skywork_gsm8k/probe.pt}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
HF_HOME="${HF_HOME:-$XDG_CACHE_HOME/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$XDG_CACHE_HOME/triton}"

VARIANTS=("baseline" "length_nulled")
VARIANT="${VARIANTS[${SLURM_ARRAY_TASK_ID:-0}]}"
OUTDIR="$PIPELINE_DIR/ppo/skywork_${VARIANT}"

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

python - <<'PY'
import importlib.util, sys
missing = [m for m in ("verl", "ray") if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(f"Missing required packages in current env: {missing}")
print("Verified VeRL runtime imports.")
PY

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME
export HF_HOME
export HUGGINGFACE_HUB_CACHE
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TRITON_CACHE_DIR
export TOKENIZERS_PARALLELISM=false
export VERL_FILE_LOGGER_PATH="$OUTDIR/metrics.jsonl"
export VERL_EXTRA_OVERRIDES="+ray_kwargs.ray_init.address=auto"
export RAY_raylet_start_wait_time_s=240
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0

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

unset ROCR_VISIBLE_DEVICES
ray stop --force 2>/dev/null || true
sleep 10

_RAY_STARTED=0
for _attempt in 1 2 3 4 5; do
  echo "[job] ray start --head attempt ${_attempt}..."
  if ray start --head \
       --num-cpus="${SLURM_CPUS_PER_TASK:-8}" \
       --num-gpus=2 \
       --object-store-memory=8589934592 \
       --include-dashboard=false \
       --disable-usage-stats; then
    _RAY_STARTED=1
    break
  fi
  ray stop --force 2>/dev/null || true
  sleep 15
done
if [[ $_RAY_STARTED -eq 0 ]]; then
  echo "[job] ray start failed after retries"
  exit 1
fi

echo "[job] host=$(hostname) variant=$VARIANT"
echo "[job] artifact_root=$ARTIFACT_ROOT"
echo "[job] policy_model=$POLICY_MODEL"
echo "[job] reward_model=$REWARD_MODEL"
echo "[job] hf_home=$HF_HOME"
echo "[job] output_dir=$OUTDIR"

LAUNCH_ARGS=(
  --base-config "$REPO/experiments/downstream/verl_ppo_base.yaml"
  --train-files "$PIPELINE_DIR/ppo/train.parquet"
  --val-files "$PIPELINE_DIR/ppo/val.parquet"
  --policy-model "$POLICY_MODEL"
  --reward-model "$REWARD_MODEL"
  --output-dir "$OUTDIR"
  --experiment-name "skywork_${VARIANT}"
  --reward-device "${REWARD_DEVICE:-cuda:1}"
  --reward-dtype "${REWARD_DTYPE:-bfloat16}"
  --reward-batch-size "${REWARD_BATCH_SIZE:-4}"
  --reward-max-length "${REWARD_MAX_LENGTH:-2048}"
  --train-steps "${TRAIN_STEPS:-200}"
  --batch-size-prompts "${BATCH_SIZE_PROMPTS:-8}"
  --group-size "${GROUP_SIZE:-4}"
  --temperature "${TEMPERATURE:-0.8}"
  --top-p "${TOP_P:-0.95}"
  --max-new-tokens "${MAX_NEW_TOKENS:-256}"
  --n-gpus-per-node 1
  --lr "${LR:-1e-6}"
  --kl-coeff "${KL_COEFF:-0.0}"
  --eval-every "${EVAL_EVERY:-50}"
  --reward-num-workers 1
)

if [[ "$VARIANT" == "length_nulled" ]]; then
  LAUNCH_ARGS+=(--probe-file "$PROBE_FILE")
fi

python scripts/launch_verl_rm_ppo.py "${LAUNCH_ARGS[@]}"
