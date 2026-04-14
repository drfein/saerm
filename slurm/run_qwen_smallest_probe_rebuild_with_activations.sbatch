#!/bin/bash
#SBATCH -J qwen06b_probe_rebuild
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH -t 12:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail
cd /sailhome/drfein/saerm

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

python scripts/rebuild_qwen_smallest_probes_with_activations.py \
  --artifacts-dir "${ARTIFACTS_DIR:-artifacts_qwen_smallest_probe_rebuild}" \
  --plots-dir "${PLOTS_DIR:-plots_qwen_smallest_probe_rebuild}" \
  --device "${DEVICE:-cuda}"
