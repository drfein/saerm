#!/bin/bash
#SBATCH -J length_deberta
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -t 4:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

# Run length bias experiment for DeBERTa model

set -euo pipefail
cd /sailhome/drfein/saerm

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

echo "============================================================"
echo "Length Bias Experiment: DeBERTa"
echo "  Model: OpenAssistant/reward-model-deberta-v3-large-v2"
echo "============================================================"

python experiments/run_experiment.py \
    --bias-type length \
    --name length_deberta_gsm8k \
    --model OpenAssistant/reward-model-deberta-v3-large-v2 \
    --dataset-source /sailhome/drfein/saerm/data/gsm8k_soln.json \
    --dataset gsm8k \
    --artifacts-dir artifacts \
    --plots-dir plots \
    --device cuda \
    --batch-size 4 \
    --max-length 1024 \
    --probe-size 500

echo "Done: length_deberta_gsm8k"


