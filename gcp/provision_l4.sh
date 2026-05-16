#!/usr/bin/env bash
# Provision an NVIDIA L4 spot instance for Stage 1 / Stage 3b reruns.
#
# Usage:
#   ./gcp/provision_l4.sh                       # default name cpt-l4-spot
#   ./gcp/provision_l4.sh my-name               # custom name
#   ON_DEMAND=1 ./gcp/provision_l4.sh ondemand  # on-demand tier ($0.40/hr)
#
# After RUNNING, the instance takes ~2-3 min to finish DLVM first-boot driver
# install. SSH availability lags VM creation; gcloud retries until ready.
#
# Spot instances may be preempted; for our use (single-run, <1 hr), the
# preemption probability is acceptable. If preempted, just re-provision and
# resume from the rolling snapshot (AMENDMENT-003).

set -euo pipefail

NAME="${1:-cpt-l4-spot}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-2-9-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_SIZE_GB="${DISK_SIZE_GB:-150}"

if [[ "${ON_DEMAND:-0}" == "1" ]]; then
  PROV_FLAGS=( )
  echo "Provisioning ON-DEMAND L4 ($0.40/hr — use only for EWC profiling smokes)..."
else
  PROV_FLAGS=( --provisioning-model=SPOT --instance-termination-action=STOP )
  echo "Provisioning SPOT L4 ($0.10/hr)..."
fi

gcloud compute instances create "$NAME" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --accelerator="type=nvidia-l4,count=1" \
  "${PROV_FLAGS[@]}" \
  --image-family="$IMAGE_FAMILY" \
  --image-project="$IMAGE_PROJECT" \
  --boot-disk-size="${DISK_SIZE_GB}GB" \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --metadata="install-nvidia-driver=True" \
  --scopes=cloud-platform \
  --shielded-secure-boot \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --quiet

echo ""
echo "Instance '$NAME' created in zone $ZONE."
echo ""
echo "Wait ~2 min for the DLVM driver install to finish, then SSH:"
echo "  gcloud compute ssh $NAME --zone=$ZONE"
echo ""
echo "Or bootstrap the project env in one shot:"
echo "  gcloud compute ssh $NAME --zone=$ZONE -- \\"
echo "    'curl -fsSL https://raw.githubusercontent.com/Khansolo9/continual_pre_training/main/gcp/bootstrap_instance.sh | bash'"
echo ""
echo "Delete when done:"
echo "  gcloud compute instances delete $NAME --zone=$ZONE --quiet"
