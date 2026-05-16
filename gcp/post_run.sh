#!/usr/bin/env bash
# Post-run pipeline (run on the laptop) after a cloud run completes.
# Brings a cloud-run's artifacts + registry row + summary-table row back
# to the laptop so the project state matches what it would be if the run
# had executed locally.
#
# Usage:
#   ./gcp/post_run.sh <run_id> [instance_name]
#
#   instance_name defaults to cpt-l4-smoke. Override for multi-instance
#   sessions (e.g., if Stage 1 + Stage 3b are on separate instances).
#
# Pipeline (each step idempotent):
#   1. Push run dir from instance -> GCS (excludes checkpoints by default)
#   2. Pull run dir from GCS -> laptop (so experiments/runs/<run_id>/ exists locally)
#   3. Pull instance's run_registry.csv -> /tmp; append any rows whose
#      run_id is not yet in the laptop's registry. Append-only per
#      research-integrity rule #3.
#   4. Regenerate experiments/summary_table.csv + summary_pack.md on the
#      laptop from the now-up-to-date registry + per-run metrics.json.
#   5. Print git status hints so you can review + commit.

set -euo pipefail

RUN_ID="${1:-}"
INSTANCE="${2:-cpt-l4-smoke}"
ZONE="${ZONE:-us-central1-a}"
BUCKET="${BUCKET:-continual-pre-training-25544659304}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "$RUN_ID" ]]; then
  echo "Usage: $0 <run_id> [instance_name]" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

echo "==> [1/5] Push $RUN_ID instance -> GCS"
gcloud compute ssh "$INSTANCE" --zone="$ZONE" --command="cd ~/continual_pre_training && ./gcp/sync_artifacts.sh push $RUN_ID"

echo "==> [2/5] Pull $RUN_ID GCS -> laptop"
./gcp/sync_artifacts.sh pull "$RUN_ID"

echo "==> [3/5] Merge new registry rows instance -> laptop"
INSTANCE_REG=$(mktemp)
gcloud compute scp --zone="$ZONE" "${INSTANCE}:~/continual_pre_training/experiments/run_registry.csv" "$INSTANCE_REG" >/dev/null

# Find rows in INSTANCE_REG whose run_id isn't already in laptop registry.
# run_id is the first column. Header line is always present in both.
LAPTOP_REG="experiments/run_registry.csv"
LAPTOP_IDS=$(mktemp)
awk -F',' 'NR>1 {print $1}' "$LAPTOP_REG" | sort > "$LAPTOP_IDS"

NEW_ROWS=$(mktemp)
awk -F',' -v laptop="$LAPTOP_IDS" '
  BEGIN {
    while ((getline line < laptop) > 0) seen[line] = 1
  }
  NR>1 && !($1 in seen) { print }
' "$INSTANCE_REG" > "$NEW_ROWS"

NEW_COUNT=$(wc -l < "$NEW_ROWS" | tr -d ' ')
if [[ "$NEW_COUNT" -gt 0 ]]; then
  cat "$NEW_ROWS" >> "$LAPTOP_REG"
  echo "  appended $NEW_COUNT new row(s):"
  awk -F',' '{print "    " $1}' "$NEW_ROWS"
else
  echo "  no new rows to append (laptop already has $RUN_ID's row)"
fi
rm -f "$INSTANCE_REG" "$LAPTOP_IDS" "$NEW_ROWS"

echo "==> [4/5] Enrich registry row from metrics.json + regenerate summary table/pack"
if [[ -d ".cpt-env" ]]; then
  # shellcheck disable=SC1091
  source .cpt-env/bin/activate
fi
# The runner only writes status + timestamps to the registry. Backfill the
# metadata columns (research_question, method, seed, model_id, etc.) from
# the run's metrics.json so cloud rows match MPS-era rows.
python src/adhoc/enrich_registry_row.py "$RUN_ID"
python src/adhoc/generate_summary_pack.py --write

echo "==> [5/5] Git status (review before commit)"
git status --short experiments/run_registry.csv experiments/summary_table.csv experiments/summary_pack.md experiments/runs/"$RUN_ID"/ || true

cat <<MSG

Done. Suggested next steps:

  # Inspect the new run
  cat experiments/runs/$RUN_ID/metrics.json | python -m json.tool | head -40

  # Commit the registry + table updates (per-stage batching is fine)
  git add experiments/run_registry.csv experiments/summary_table.csv experiments/summary_pack.md
  git commit -m "Registry + summary_table: capture $RUN_ID results"
  git push origin main

Note: experiments/runs/$RUN_ID/ is gitignored, so the raw artifacts stay
on disk + in GCS as the audit trail. The registry/table is the
git-tracked record that they exist and what they show.
MSG
