#!/bin/bash
#SBATCH -J calibration
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --account=aal
#SBATCH --partition=aal
#SBATCH --array=0-4   # 5 models

# Calibration (confidence probe) experiments for all models
#
# Tests if reward models are biased by verbalized confidence scores
# Uses manual confidence manipulation on Math 500 dataset
#
# Models:
#   skywork, allen, skywork_qwen3, skywork_qwen-smallest, deberta
#
# Usage:
#   sbatch slurm/run_calibration_all.sh

set -euo pipefail
cd /sailhome/drfein/saerm

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

# Models (5)
declare -a MODELS=(
    "skywork|Skywork/Skywork-Reward-V2-Llama-3.1-8B"
    "allen|allenai/Llama-3.1-8B-Instruct-RM-RB2"
    "skywork_qwen3|Skywork/Skywork-Reward-V2-Qwen3-8B"
    "skywork_qwen-smallest|Skywork/Skywork-Reward-V2-Qwen3-0.6B"
    "deberta|OpenAssistant/reward-model-deberta-v3-large-v2"
)

# Parse model
IFS='|' read -r MODEL_NAME MODEL_PATH <<< "${MODELS[$SLURM_ARRAY_TASK_ID]}"

EXPERIMENT_NAME="calibration_${MODEL_NAME}_math500"

echo "============================================================"
echo "Job ${SLURM_ARRAY_TASK_ID}: ${EXPERIMENT_NAME}"
echo "  Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "  Bias: calibration (confidence probe)"
echo "  Dataset: Math 500"
echo "============================================================"

# Set batch size (smaller for deberta)
BATCH_SIZE=8
if [[ "${MODEL_NAME}" == "deberta" ]]; then
    BATCH_SIZE=4
fi

python experiments/run_experiment.py \
    --bias-type calibration \
    --name "${EXPERIMENT_NAME}" \
    --model "${MODEL_PATH}" \
    --dataset-source /sailhome/drfein/saerm/data/math500_uncertainty.json \
    --dataset math500 \
    --artifacts-dir artifacts \
    --plots-dir plots \
    --device cuda \
    --trust-remote-code \
    --batch-size "${BATCH_SIZE}" \
    --max-length 2048 \
    --probe-size 250 \
    --max-test-examples 250 \
    --extra-dataset_class calibration \
    --extra-probe_conf_high "[10]" \
    --extra-probe_conf_low "[1]" \
    --extra-eval_conf_levels "[1, 3, 5, 7, 10]" \
    --extra-min_rollouts 5

echo "Done: ${EXPERIMENT_NAME}"
