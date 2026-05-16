#!/usr/bin/env bash
# Laptop-side polling helper: scan GCS for completed runs not yet on the
# laptop, run post_run.sh for each (which pulls + updates registry +
# regenerates summary_table), and report new completions.
#
# Usage:
#   ./gcp/pull_completed.sh                    # one-shot scan
#   ./gcp/pull_completed.sh --watch [interval] # poll every <interval>s (default 600)
#
# Detection: a run is "completed in GCS" if it has metrics.json under
# gs://<BUCKET>/runs/<run_id>/. We treat that as the source-of-truth
# signal (consistent with what the cloud queue.sh writes after each
# python run_experiment.py returns 0).

set -euo pipefail

BUCKET="${BUCKET:-continual-pre-training-25544659304}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

scan_once() {
  echo "[$(date '+%H:%M:%S')] scanning gs://$BUCKET/runs/ for completed runs..."

  # List all metrics.json files in the bucket. Format:
  #   gs://<bucket>/runs/<run_id>/metrics.json
  REMOTE_RUNS=$(gsutil ls "gs://$BUCKET/runs/*/metrics.json" 2>/dev/null \
    | awk -F'/' '{print $(NF-1)}' \
    | sort)

  if [[ -z "$REMOTE_RUNS" ]]; then
    echo "  (no completed runs in bucket yet)"
    return 0
  fi

  NEW_COUNT=0
  for RUN_ID in $REMOTE_RUNS; do
    if [[ -f "experiments/runs/$RUN_ID/metrics.json" ]]; then
      continue  # already pulled
    fi
    echo "  NEW: $RUN_ID"
    if ./gcp/post_run.sh "$RUN_ID" >/dev/null 2>&1; then
      echo "    pulled + registry updated"
      NEW_COUNT=$((NEW_COUNT + 1))
    else
      echo "    post_run.sh FAILED for $RUN_ID (see git status for partial state)"
    fi
  done

  if [[ "$NEW_COUNT" -gt 0 ]]; then
    echo "[$(date '+%H:%M:%S')] pulled $NEW_COUNT new run(s). Review:"
    git status --short experiments/run_registry.csv experiments/summary_table.csv experiments/summary_pack.md
    cat <<EOF

Suggested commit:
  git add experiments/run_registry.csv experiments/summary_table.csv experiments/summary_pack.md
  git commit -m "Registry + summary_table: capture <run_ids>"

(Or commit per-run for finer granularity.)
EOF
  else
    echo "[$(date '+%H:%M:%S')] no new completed runs."
  fi
}

if [[ "${1:-}" == "--watch" ]]; then
  INTERVAL="${2:-600}"
  echo "Watching every ${INTERVAL}s. Ctrl-C to stop."
  while true; do
    scan_once || true
    sleep "$INTERVAL"
  done
else
  scan_once
fi
