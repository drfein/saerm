#!/bin/bash
#SBATCH -J length_all
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --account=aal
#SBATCH --partition=aal
#SBATCH --array=0-4   # 5 models

# Length Bias experiments for 5 models
#
# Models:
#   skywork, allen, skywork_qwen3, skywork_qwen-smallest, deberta

set -euo pipefail
cd /sailhome/drfein/saerm

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

# Models
declare -a MODELS=(
    "skywork|Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    "allen|allenai/Llama-3.1-8B-Instruct-RM-RB2"
    "skywork_qwen3|Skywork/Skywork-Reward-V2-Qwen3-8B"
    "skywork_qwen-smallest|Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    "deberta|OpenAssistant/reward-model-deberta-v3-large-v2"
)

# Parse model
IFS='|' read -r MODEL_NAME MODEL_PATH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

EXPERIMENT_NAME="length_${MODEL_NAME}_gsm8k"

echo "============================================================"
echo "Job ${SLURM_ARRAY_TASK_ID}: ${EXPERIMENT_NAME}"
echo "  Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "  Bias: length"
echo "============================================================"

# Note: Using larger batch size/length for consistency with other GSM8K runs
python experiments/run_experiment.py \
    --bias-type length \
    --name "${EXPERIMENT_NAME}" \
    --model "${MODEL_PATH}" \
    --dataset-source /sailhome/drfein/saerm/data/gsm8k_soln.json \
    --dataset gsm8k \
    --artifacts-dir artifacts \
    --plots-dir plots \
    --device cuda \
    --trust-remote-code \
    --batch-size 8 \
    --max-length 2048 \
    --probe-size 500

echo "Done: ${EXPERIMENT_NAME}"
