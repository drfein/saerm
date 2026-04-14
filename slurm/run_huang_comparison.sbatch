#!/bin/bash
#SBATCH -J huang_lwr_compare
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH -t 4:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail
REPO=/sailhome/drfein/saerm
cd "$REPO"

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-/aal/u/drfein/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

OUT_DIR="${OUT_DIR:-artifacts/results/huang_comparison}"
mkdir -p "$OUT_DIR"

echo "=== Huang et al. (LWR) vs Ours: Allen RM on GSM8K ==="
python scripts/compare_saved_debias_vs_lwr_probefit.py \
  --raw-data-files artifacts/raw_data/length/length_allen_gsm8k/raw_data.json \
  --source-file data/gsm8k_soln.json \
  --output-file "$OUT_DIR/huang_comparison_allen_gsm8k.json" \
  --device cuda \
  --batch-size 4 \
  --max-length 1024 \
  --lwr-frac 0.5 \
  --lwr-alpha 1.0

echo "=== Huang et al. (LWR) vs Ours: DeBERTa RM on GSM8K ==="
python scripts/compare_saved_debias_vs_lwr_probefit.py \
  --raw-data-files artifacts/raw_data/length/length_deberta_gsm8k/raw_data.json \
  --source-file data/gsm8k_soln.json \
  --output-file "$OUT_DIR/huang_comparison_deberta_gsm8k.json" \
  --device cuda \
  --batch-size 4 \
  --max-length 1024 \
  --lwr-frac 0.5 \
  --lwr-alpha 1.0

echo "=== Done. Results in $OUT_DIR ==="
