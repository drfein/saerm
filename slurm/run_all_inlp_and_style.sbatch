#!/bin/bash
#SBATCH -J inlp_style_all
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH -t 48:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail
REPO=/sailhome/drfein/saerm
cd "$REPO"

source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
conda activate saerm

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export SAERM_ROOT="$REPO"
export HF_HOME="${HF_HOME:-/aal/u/drfein/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"

BIAS_ARTIFACTS="${BIAS_ARTIFACTS:-artifacts_inlp_rebuild}"
BIAS_PLOTS="${BIAS_PLOTS:-plots_inlp_rebuild}"
STYLE_ARTIFACTS="${STYLE_ARTIFACTS:-artifacts_style_probe}"
STYLE_PLOTS="${STYLE_PLOTS:-plots_style_probe}"

mkdir -p logs "$BIAS_ARTIFACTS" "$STYLE_ARTIFACTS"

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [1/5] Rebuild bias probes with activations (all 5 RMs) ==="
python scripts/rebuild_all_rms_probes_with_activations.py \
  --artifacts-dir "$BIAS_ARTIFACTS" \
  --plots-dir "$BIAS_PLOTS" \
  --device cuda

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [2/5] Compute NLL scores for all 10 generative LMs ==="
python scripts/compute_style_nll_scores.py \
  --dataset allenai/tulu-3-wildchat-reused-on-policy-8b \
  --n-prompts 2400 \
  --seed 10 \
  --output "$STYLE_ARTIFACTS/nll_scores.json" \
  --batch-size 2 \
  --device cuda

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [3/5] Build style probe activations for all 5 RMs ==="

declare -a RM_NAMES=(
  "allen"
  "skywork"
  "skywork_qwen3"
  "skywork_qwsm"
  "deberta"
)
declare -a RM_PATHS=(
  "allenai/Llama-3.1-8B-Instruct-RM-RB2"
  "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
  "Skywork/Skywork-Reward-V2-Qwen3-8B"
  "Skywork/Skywork-Reward-V2-Qwen3-0.6B"
  "OpenAssistant/reward-model-deberta-v3-large-v2"
)

for i in "${!RM_NAMES[@]}"; do
  rm_name="${RM_NAMES[$i]}"
  rm_path="${RM_PATHS[$i]}"
  echo "  Building style probe for $rm_name ($rm_path)"
  python scripts/build_style_probe_per_rm.py \
    --nll-scores "$STYLE_ARTIFACTS/nll_scores.json" \
    --reward-model "$rm_path" \
    --output-dir "$STYLE_ARTIFACTS/probes/style/${rm_name}" \
    --lm-a "Qwen/Qwen3-8B" \
    --lm-b "meta-llama/Llama-2-7b-chat-hf" \
    --batch-size 4 \
    --device cuda
done

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [4/5] INLP table + t-SNE for bias probes ==="
python scripts/plot_inlp_tsne.py \
  --probe-root "$BIAS_ARTIFACTS/probes" \
  --output-json "$BIAS_ARTIFACTS/inlp_results.json" \
  --output-plot "$BIAS_PLOTS/inlp_tsne_panel.png" \
  --max-iters 5 \
  --seed 42

# ─────────────────────────────────────────────────────────────────────────────
echo "=== [5/5] INLP table + t-SNE for style probes ==="
python scripts/plot_inlp_tsne.py \
  --probe-root "$STYLE_ARTIFACTS/probes" \
  --output-json "$STYLE_ARTIFACTS/inlp_results.json" \
  --output-plot "$STYLE_PLOTS/inlp_tsne_panel.png" \
  --max-iters 5 \
  --seed 42

echo "=== Done ==="
echo "Bias INLP results:  $BIAS_ARTIFACTS/inlp_results.json"
echo "Style INLP results: $STYLE_ARTIFACTS/inlp_results.json"
echo "Bias tSNE plot:     $BIAS_PLOTS/inlp_tsne_panel.png"
echo "Style tSNE plot:    $STYLE_PLOTS/inlp_tsne_panel.png"
