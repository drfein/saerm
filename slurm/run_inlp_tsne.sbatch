#!/bin/bash
#SBATCH -J inlp_tsne
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH -t 6:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail
cd /sailhome/drfein/saerm

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

export PYTHONUNBUFFERED=1
export PYTHONPATH="/sailhome/drfein/saerm${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/aal/u/drfein/.cache}"
export HF_HOME="${HF_HOME:-/aal/u/drfein/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

ARTIFACTS_DIR="${ARTIFACTS_DIR:-artifacts_inlp_rebuild}"
PLOTS_DIR="${PLOTS_DIR:-plots_inlp_rebuild}"

echo "=== Step 1: Rebuild probes with activations ==="
python scripts/rebuild_allen_deberta_probes_with_activations.py \
  --artifacts-dir "$ARTIFACTS_DIR" \
  --plots-dir "$PLOTS_DIR" \
  --device cuda

echo "=== Step 2: INLP table + t-SNE panel ==="
python scripts/plot_inlp_tsne.py \
  --probe-root "$ARTIFACTS_DIR/probes" \
  --output-json "$ARTIFACTS_DIR/inlp_results.json" \
  --output-plot "$PLOTS_DIR/inlp_tsne_panel.png" \
  --max-iters 5 \
  --seed 42

echo "=== Done. Results in $ARTIFACTS_DIR/inlp_results.json ==="
echo "=== Plot in $PLOTS_DIR/inlp_tsne_panel.png ==="
