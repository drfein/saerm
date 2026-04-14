#!/bin/bash
#SBATCH -J calibration_alpha
#SBATCH -o logs/%A_%a.out
#SBATCH -e logs/%A_%a.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -t 6:00:00
#SBATCH --account=aal
#SBATCH --partition=aal
#SBATCH --array=0-4   # 5 models

# Calibration experiments with alpha sweep
#
# Tests both confidence probe and uncertainty probe with multiple alpha values
# Alpha values: 0.1, 0.5, 0.75, 1.0, 1.5
#
# Models:
#   skywork, allen, skywork_qwen3, skywork_qwen-smallest, deberta
#
# Usage:
#   sbatch slurm/run_calibration_alpha_sweep.sh

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

echo "============================================================"
echo "Job ${SLURM_ARRAY_TASK_ID}: Calibration Alpha Sweep - ${MODEL_NAME}"
echo "  Model: ${MODEL_NAME} (${MODEL_PATH})"
echo "  Probes: confidence, uncertainty"
echo "  Alpha values: 0.1, 0.5, 0.75, 1.0, 1.5"
echo "============================================================"

# Create temporary config files with model-specific settings
CALIBRATION_CONFIG="/tmp/calibration_${MODEL_NAME}_${SLURM_JOB_ID}.yaml"
UNCERTAINTY_CONFIG="/tmp/uncertainty_${MODEL_NAME}_${SLURM_JOB_ID}.yaml"

# Set batch size (smaller for deberta)
BATCH_SIZE=8
if [[ "${MODEL_NAME}" == "deberta" ]]; then
    BATCH_SIZE=4
fi

# Create calibration config
cat > "${CALIBRATION_CONFIG}" <<EOF
name: calibration_${MODEL_NAME}_math500
bias_type: calibration

model_path: ${MODEL_PATH}
trust_remote_code: true

dataset_source: /sailhome/drfein/saerm/data/math500_uncertainty.json
dataset_class: calibration
probe_size: 250
max_test_examples: 250
split_seed: 42

batch_size: ${BATCH_SIZE}
max_length: 2048
device: cuda

save_probe: true

extra:
  probe_conf_high: [10]
  probe_conf_low: [1]
  eval_conf_levels: [1, 3, 5, 7, 10]
  min_rollouts: 5
EOF

# Create uncertainty config (for comparison)
cat > "${UNCERTAINTY_CONFIG}" <<EOF
name: uncertainty_${MODEL_NAME}_plausibleqa
bias_type: uncertainty

model_path: ${MODEL_PATH}
trust_remote_code: true

dataset_source: /sailhome/drfein/saerm/data/plausibleqa.json
dataset_class: uncertainty
probe_size: 500
max_test_examples: null
split_seed: 42

batch_size: ${BATCH_SIZE}
max_length: 2048
device: cuda

save_probe: true

extra:
  min_plaus_gap: 0.0
EOF

echo ""
echo "Running calibration experiments with alpha sweep..."
echo ""

python experiments/run_calibration_alpha_sweep.py \
    --calibration-config "${CALIBRATION_CONFIG}" \
    --uncertainty-config "${UNCERTAINTY_CONFIG}" \
    --alpha-values 0.1 0.5 0.75 1.0 1.5 \
    --output-dir artifacts/results/calibration_alpha_sweep

# Clean up temporary configs
rm -f "${CALIBRATION_CONFIG}" "${UNCERTAINTY_CONFIG}"

echo ""
echo "Done: calibration_${MODEL_NAME}_alpha_sweep"
