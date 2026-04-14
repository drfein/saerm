#!/bin/bash
#SBATCH -J bon_sig
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

REPO="${REPO:-/sailhome/drfein/saerm}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-saerm}"
DETAILS_FILE="${DETAILS_FILE:-}"
SUMMARY_FILE="${SUMMARY_FILE:-}"
OUTPUT_FILE="${OUTPUT_FILE:-}"

if [[ -z "$DETAILS_FILE" || -z "$SUMMARY_FILE" || -z "$OUTPUT_FILE" ]]; then
  echo "Missing env vars"
  echo "DETAILS_FILE=${DETAILS_FILE:-<unset>}"
  echo "SUMMARY_FILE=${SUMMARY_FILE:-<unset>}"
  echo "OUTPUT_FILE=${OUTPUT_FILE:-<unset>}"
  exit 2
fi

cd "$REPO"

if [[ -f /sailhome/drfein/miniconda3/etc/profile.d/conda.sh ]]; then
  source /sailhome/drfein/miniconda3/etc/profile.d/conda.sh
elif [[ -f /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh ]]; then
  source /nlp/scr/drfein/miniconda3/etc/profile.d/conda.sh
else
  source /sailhome/drfein/miniconda3/bin/activate "$CONDA_ENV_NAME"
fi
conda activate "$CONDA_ENV_NAME" 2>/dev/null || true

export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"

python scripts/analyze_multi_rm_bon_significance.py \
  --details-file "$DETAILS_FILE" \
  --summary-file "$SUMMARY_FILE" \
  --output "$OUTPUT_FILE"
