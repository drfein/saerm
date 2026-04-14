#!/bin/bash
#SBATCH --job-name=length_gsm8k
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.err
#SBATCH --partition=aal
#SBATCH --account=aal
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --array=0-4

# GSM8K length bias re-run (after indentation fix)
MODELS=(allen deberta skywork_qwen-smallest skywork_qwen3 skywork)
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}

cd /sailhome/drfein/saerm
mkdir -p logs

source /sailhome/drfein/miniconda3/bin/activate saerm
export PYTHONPATH=$PYTHONPATH:.

echo "Running GSM8K length bias for $MODEL"
python experiments/run_experiment.py --config experiments/configs/length_${MODEL}_gsm8k.yaml
