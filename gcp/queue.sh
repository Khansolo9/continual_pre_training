#!/usr/bin/env bash
# Sequential queue runner — runs ON the L4 instance, typically inside a
# tmux session so it survives SSH disconnects. Reads a queue file of
# one-run-per-line specs and executes each, pushing to GCS on success.
#
# Queue file format (one run per line, # for comments):
#
#   <run_id> <method_config> <model_config> <seed>
#
# Example queue file (gcp/queue_stage1.txt):
#
#   qwen3_rq2_mer25_s42_cuda    configs/methods/mer25.yaml    configs/models/qwen3_0.6b.yaml    42
#   gemma3_rq2_replay25_s42_cuda configs/methods/replay25.yaml configs/models/gemma3_1b.yaml   42
#
# Usage (on the instance):
#   cd ~/continual_pre_training
#   tmux new-session -d -s queue "bash gcp/queue.sh gcp/queue_stage1.txt"
#   tmux attach -t queue   # to watch
#
# Failure handling:
# - One run's failure does not stop the queue.
# - Failed run IDs are appended to gcp/batch_log_failures.txt for review.
# - Per-run logs remain at experiments/runs/<run_id>/{stdout,run}.log even
#   on failure.

set -uo pipefail

QUEUE_FILE="${1:-}"
if [[ -z "$QUEUE_FILE" || ! -f "$QUEUE_FILE" ]]; then
  echo "Usage: $0 <queue_file>" >&2
  echo "Queue file must exist; one run per line: <run_id> <method_cfg> <model_cfg> <seed>" >&2
  exit 1
fi

PROJECT_DIR="$HOME/continual_pre_training"
cd "$PROJECT_DIR"
# shellcheck disable=SC1091
source .cpt-env/bin/activate

BATCH_LOG="gcp/batch_log.txt"
FAILURES="gcp/batch_log_failures.txt"

log() {
  local line="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
  echo "$line" | tee -a "$BATCH_LOG"
}

log "=================================================================="
log "Queue start: $QUEUE_FILE"
log "=================================================================="

while IFS= read -r line || [[ -n "$line" ]]; do
  # Skip blanks and comments
  [[ -z "${line// }" || "${line:0:1}" == "#" ]] && continue

  # Parse columns (whitespace-separated)
  read -r RUN_ID METHOD_CFG MODEL_CFG SEED <<< "$line"
  if [[ -z "${RUN_ID:-}" || -z "${METHOD_CFG:-}" || -z "${MODEL_CFG:-}" || -z "${SEED:-}" ]]; then
    log "SKIP malformed line: $line"
    continue
  fi

  RUN_DIR="experiments/runs/$RUN_ID"
  if [[ -f "$RUN_DIR/metrics.json" ]]; then
    log "SKIP $RUN_ID (already has metrics.json — assumed complete)"
    continue
  fi

  log "------------------------------------------------------------------"
  log "START $RUN_ID  method=$METHOD_CFG  model=$MODEL_CFG  seed=$SEED"
  log "------------------------------------------------------------------"

  mkdir -p "$RUN_DIR"
  START_EPOCH=$(date +%s)

  if python src/run_experiment.py \
       --run-id "$RUN_ID" \
       --config "$METHOD_CFG" \
       --model-config "$MODEL_CFG" \
       --seed "$SEED" 2>&1 | tee "$RUN_DIR/stdout.log"; then
    ELAPSED=$(( $(date +%s) - START_EPOCH ))
    log "DONE  $RUN_ID  ${ELAPSED}s"
    if ./gcp/sync_artifacts.sh push "$RUN_ID" >> "$BATCH_LOG" 2>&1; then
      log "PUSH  $RUN_ID -> GCS OK"
    else
      log "PUSH  $RUN_ID -> GCS FAILED (artifacts remain on instance)"
      echo "$RUN_ID push_failed" >> "$FAILURES"
    fi
  else
    ELAPSED=$(( $(date +%s) - START_EPOCH ))
    log "FAIL  $RUN_ID  exit=$?  ${ELAPSED}s"
    echo "$RUN_ID run_failed" >> "$FAILURES"
    # Still try to push whatever partial artifacts exist (training_log,
    # run.log) so we can debug from the laptop without SSHing back in.
    ./gcp/sync_artifacts.sh push "$RUN_ID" >> "$BATCH_LOG" 2>&1 || true
  fi
done < "$QUEUE_FILE"

log "=================================================================="
log "Queue end"
if [[ -s "$FAILURES" ]]; then
  log "Failures recorded in $FAILURES:"
  cat "$FAILURES" | tee -a "$BATCH_LOG"
fi
log "=================================================================="
