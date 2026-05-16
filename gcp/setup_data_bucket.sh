#!/usr/bin/env bash
# One-time setup: create the project GCS bucket and upload tokenized data,
# manifests, and eval prompts so cloud instances can pull them on bootstrap.
#
# Run from the project root on the laptop:
#   ./gcp/setup_data_bucket.sh
#
# Idempotent: re-running just rsyncs any new/changed files into the bucket.
# Skips the bucket-create step if it already exists.

set -euo pipefail

PROJECT_ID="${PROJECT_ID:-continual-pre-training}"
BUCKET="${BUCKET:-continual-pre-training-25544659304}"
REGION="${REGION:-us-central1}"

if [[ ! -d data/processed || ! -d data/manifests ]]; then
  echo "ERROR: run from project root (data/processed/ and data/manifests/ must exist)" >&2
  exit 1
fi

echo "==> Bucket: gs://$BUCKET (region $REGION)"
if gsutil ls -b "gs://$BUCKET" >/dev/null 2>&1; then
  echo "  exists"
else
  gsutil mb -p "$PROJECT_ID" -c standard -l "$REGION" -b on "gs://$BUCKET"
  echo "  created"
fi

# Lifecycle: auto-delete snapshots/ contents after 90 days. Tokenized data and
# runs/ artifacts stay forever (they're tiny).
LIFECYCLE_JSON=$(mktemp)
cat >"$LIFECYCLE_JSON" <<'JSON'
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {"age": 90, "matchesPrefix": ["snapshots/"]}
      }
    ]
  }
}
JSON
gsutil lifecycle set "$LIFECYCLE_JSON" "gs://$BUCKET" >/dev/null
rm "$LIFECYCLE_JSON"

echo "==> Sync data/ → gs://$BUCKET/data/ (excludes data/raw/ and data/cache/)"
gsutil -m rsync -r -x '^raw/|^cache/' data/ "gs://$BUCKET/data/"

echo "==> Done. Bucket contents:"
gsutil du -sh "gs://$BUCKET/data/processed" "gs://$BUCKET/data/manifests" "gs://$BUCKET/data/eval" 2>/dev/null || true
