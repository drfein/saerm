#!/bin/bash
#SBATCH -J ppo_check
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=01:00:00
#SBATCH --account=aal
#SBATCH --partition=aal

set -euo pipefail

PARENT_JOB_ID="${PARENT_JOB_ID:?PARENT_JOB_ID required}"
PARENT_STDOUT="${PARENT_STDOUT:?PARENT_STDOUT required}"
PARENT_STDERR="${PARENT_STDERR:?PARENT_STDERR required}"
FALLBACK_WRAPPER="${FALLBACK_WRAPPER:?FALLBACK_WRAPPER required}"
FALLBACK_OUTDIR_ROOT="${FALLBACK_OUTDIR_ROOT:?FALLBACK_OUTDIR_ROOT required}"

state="$(sacct -X -n -P -j "$PARENT_JOB_ID" --format=State | head -n1 | cut -d'|' -f1 | tr -d ' ')"
collapse=0
reason="ok"

if [[ "$state" != "COMPLETED" ]]; then
  collapse=1
  reason="state:$state"
fi

if grep -Eiq '(^|[^a-z])nan([^a-z]|$)|OutOfMemoryError|device-side assert|Traceback|probability tensor contains either inf' "$PARENT_STDOUT" "$PARENT_STDERR" 2>/dev/null; then
  collapse=1
  if [[ "$reason" == "ok" ]]; then
    reason="log_signature"
  else
    reason="${reason}+log_signature"
  fi
fi

mkdir -p "$FALLBACK_OUTDIR_ROOT"
printf '{\n  "parent_job_id": "%s",\n  "parent_state": "%s",\n  "collapse": %s,\n  "reason": "%s"\n}\n' \
  "$PARENT_JOB_ID" "$state" "$([[ $collapse -eq 1 ]] && echo true || echo false)" "$reason" \
  > "$FALLBACK_OUTDIR_ROOT/check_result.json"

if [[ "$collapse" -eq 0 ]]; then
  echo "No collapse detected for $PARENT_JOB_ID"
  exit 0
fi

sbatch --export=ALL,\
PIPELINE_DIR="$PIPELINE_DIR",\
POLICY_MODEL="$POLICY_MODEL",\
REWARD_MODEL="$REWARD_MODEL",\
PROBE_FILE="$PROBE_FILE",\
VARIANT="$VARIANT",\
REWARD_TAG="$REWARD_TAG",\
PPO_OUTDIR_ROOT="$FALLBACK_OUTDIR_ROOT",\
ATTN_IMPLEMENTATION="$ATTN_IMPLEMENTATION",\
TRAIN_STEPS="$FALLBACK_TRAIN_STEPS",\
MAX_NEW_TOKENS="$MAX_NEW_TOKENS",\
LR="$LR",\
NUM_PPO_EPOCHS="$NUM_PPO_EPOCHS",\
KL_COEFF="$KL_COEFF",\
EVAL_EVERY="$FALLBACK_EVAL_EVERY",\
SAVE_STEPS="$FALLBACK_SAVE_STEPS",\
LOGGING_STEPS="$LOGGING_STEPS" \
  "$FALLBACK_WRAPPER"
