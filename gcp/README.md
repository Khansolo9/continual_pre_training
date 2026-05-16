# GCP cloud execution (Stage 1 / Stage 3b reruns)

Scripts that provision an NVIDIA L4 spot instance on GCP, bootstrap the
continual-pretraining environment on it, and sync run artifacts back to
Cloud Storage. Used for post-AMENDMENT-004 reruns where laptop MPS is
the bottleneck. Paper 2 EWC profiling stays on the laptop (MPS) — see
`docs/plans/GCP_MIGRATION_PLAN.md`.

## Prerequisites (one-time, already done)

- `gcloud` SDK installed and authenticated (`gcloud auth login`,
  `gcloud auth application-default login`).
- Project set: `continual-pre-training` (project number `25544659304`).
- Default region/zone: `us-central1` / `us-central1-a`.
- Quotas in `us-central1`:
  - `NVIDIA_L4_GPUS = 1`
  - `PREEMPTIBLE_NVIDIA_L4_GPUS = 1` (spot tier)
- APIs enabled: `compute.googleapis.com`, `storage.googleapis.com`.

## Bucket layout

- `gs://${BUCKET}/data/` — tokenized data + manifests + eval prompts, uploaded once via `setup_data_bucket.sh`.
- `gs://${BUCKET}/runs/` — `experiments/runs/<run_id>/` artifacts pushed by `sync_artifacts.sh push` after a run.
- `gs://${BUCKET}/snapshots/` — optional rolling checkpoint snapshots if a run is killed mid-Domain-B.

`${BUCKET}` defaults to `continual-pre-training-25544659304` (project ID + project number) for global uniqueness.

## Typical session

```bash
# 1. (Once per project) Create the bucket + upload data
./gcp/setup_data_bucket.sh

# 2. Provision the spot instance (~30s to RUNNING, ~3 min until SSH-ready)
./gcp/provision_l4.sh cpt-l4-spot-1

# 3. SSH in and bootstrap
gcloud compute ssh cpt-l4-spot-1 -- 'curl -fsSL https://raw.githubusercontent.com/Khansolo9/continual_pre_training/main/gcp/bootstrap_instance.sh | bash'

# 4. Launch a run on the instance
gcloud compute ssh cpt-l4-spot-1 -- 'cd continual_pre_training && source .cpt-env/bin/activate && python src/run_experiment.py --config configs/methods/replay25.yaml --model-config configs/models/qwen3.yaml --run-id qwen3_rq2_replay25_s42_cuda'

# 5. After the run finishes, push artifacts back
gcloud compute ssh cpt-l4-spot-1 -- 'cd continual_pre_training && ./gcp/sync_artifacts.sh push qwen3_rq2_replay25_s42_cuda'

# 6. Pull to the laptop and tear down
./gcp/sync_artifacts.sh pull qwen3_rq2_replay25_s42_cuda
gcloud compute instances delete cpt-l4-spot-1 --quiet
```

A typical 1B replay25 run takes ~25–35 min on L4. Spot pricing is $0.10/hr.

## Cost guardrails

- **Always delete the instance after the run**: spot is cheap but a forgotten instance still bills overnight at $0.10/hr.
- **Use spot for non-EWC**, on-demand for EWC profiling (rare, only on laptop now anyway).
- **GCS storage**: ~$0.02/GB/month standard tier. Total project artifacts likely <10 GB → <$0.20/mo.

## What the scripts do NOT do

- They do not run multi-instance / multi-region jobs. One L4 at a time.
- They do not auto-launch runs. Each run is started by hand after `bootstrap_instance.sh` completes, so we can confirm parity smokes before any rerun.
- They do not delete instances automatically. Manual `gcloud compute instances delete` after artifact sync.
