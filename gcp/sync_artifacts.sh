#!/usr/bin/env bash
# Push a finished run's artifacts to GCS (on the instance), or pull them
# back to the laptop (on the laptop). Run dirs typically contain:
#   metrics.json, run_env.json, training_log.jsonl, runpack_*.md,
#   theta_AB.pt (optional, ~2-4 GB for 1B models — gated by KEEP_CHECKPOINTS).
#
# Usage:
#   ./gcp/sync_artifacts.sh push <run_id>      # instance -> GCS
#   ./gcp/sync_artifacts.sh pull <run_id>      # GCS -> laptop
#   ./gcp/sync_artifacts.sh push-all           # instance -> GCS (everything in experiments/runs/)
#
# By default checkpoints (*.pt, *.safetensors) are NOT pushed — they're large
# and reproducible from the seed + code. Set KEEP_CHECKPOINTS=1 to include.

set -euo pipefail

BUCKET="${BUCKET:-continual-pre-training-25544659304}"

CMD="${1:-}"
RUN_ID="${2:-}"

if [[ -z "$CMD" ]]; then
  echo "Usage: $0 {push|pull|push-all} [run_id]" >&2
  exit 1
fi

EXCLUDES=( )
if [[ "${KEEP_CHECKPOINTS:-0}" != "1" ]]; then
  EXCLUDES+=( -x '\.pt$|\.safetensors$|\.bin$' )
fi

case "$CMD" in
  push)
    [[ -z "$RUN_ID" ]] && { echo "push needs <run_id>"; exit 1; }
    src="experiments/runs/$RUN_ID/"
    dst="gs://$BUCKET/runs/$RUN_ID/"
    [[ ! -d "$src" ]] && { echo "missing $src"; exit 1; }
    echo "==> push $src -> $dst"
    gsutil -m rsync -r "${EXCLUDES[@]}" "$src" "$dst"
    ;;
  pull)
    [[ -z "$RUN_ID" ]] && { echo "pull needs <run_id>"; exit 1; }
    src="gs://$BUCKET/runs/$RUN_ID/"
    dst="experiments/runs/$RUN_ID/"
    mkdir -p "$dst"
    echo "==> pull $src -> $dst"
    gsutil -m rsync -r "$src" "$dst"
    ;;
  push-all)
    echo "==> push-all experiments/runs/ -> gs://$BUCKET/runs/"
    gsutil -m rsync -r "${EXCLUDES[@]}" experiments/runs/ "gs://$BUCKET/runs/"
    ;;
  *)
    echo "Unknown command: $CMD" >&2
    exit 1
    ;;
esac
