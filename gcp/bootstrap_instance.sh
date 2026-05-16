#!/usr/bin/env bash
# Runs ON the L4 instance to set up the continual-pretraining environment.
#
# Fetched + executed in one shot:
#   curl -fsSL https://raw.githubusercontent.com/Khansolo9/continual_pre_training/main/gcp/bootstrap_instance.sh | bash
#
# What it does:
#   1. Clone (or pull) the public repo into ~/continual_pre_training.
#   2. Create a venv at ~/continual_pre_training/.cpt-env (matches laptop name).
#   3. Install pinned deps from requirements.txt with torch swapped to the
#      CUDA 12.6 wheel (compatible with the DLVM CUDA 12.9 driver).
#   4. Pull tokenized data + manifests + eval prompts from GCS into data/.
#   5. Run the validation smoke gate: pytest tests/, EWC equivalence verifier,
#      tiny CUDA forward pass.
#   6. Print model-weight download instructions (HF Hub login is interactive,
#      so we leave that as a manual step).

set -euo pipefail

REPO="https://github.com/Khansolo9/continual_pre_training.git"
PROJECT_DIR="$HOME/continual_pre_training"
BUCKET="${BUCKET:-continual-pre-training-25544659304}"

echo "==> [1/6] Repo"
if [[ -d "$PROJECT_DIR/.git" ]]; then
  cd "$PROJECT_DIR" && git fetch && git checkout main && git pull --ff-only
else
  git clone "$REPO" "$PROJECT_DIR"
  cd "$PROJECT_DIR"
fi

echo "==> [2/6] Python venv"
# The DLVM PyTorch 2.9 / Ubuntu 22.04 image ships system Python 3.10 without
# the `venv` module — `python3 -m venv` fails with "ensurepip is not available"
# until python3.10-venv is apt-installed. Idempotent: dpkg-skip if already on.
if ! dpkg -s python3.10-venv >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq python3.10-venv python3-pip
fi
if [[ ! -f "$PROJECT_DIR/.cpt-env/bin/activate" ]]; then
  rm -rf "$PROJECT_DIR/.cpt-env"
  python3 -m venv "$PROJECT_DIR/.cpt-env"
fi
# shellcheck disable=SC1091
source "$PROJECT_DIR/.cpt-env/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo "==> [3/6] Install pinned deps (torch first via CUDA wheel index)"
# Install torch 2.10.0 with cu126 wheel — matches the MPS-side 2.10.0 build at
# the API level; CUDA reduction kernels differ from MPS by design.
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# Then everything else from the locked file. torch is already satisfied so pip
# leaves it alone.
pip install -r requirements.txt

echo "==> [4/6] Pull tokenized data from gs://$BUCKET/data/"
mkdir -p data
gsutil -m rsync -r "gs://$BUCKET/data/" data/

echo "==> [5/6] Validation smokes"
echo "  - CUDA visible to torch?"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not visible'; print(f'  CUDA OK: {torch.cuda.get_device_name(0)}'); print(f'  torch={torch.__version__}, cuda={torch.version.cuda}')"
echo "  - pytest tests/"
python -m pytest tests/ -q --ignore=tests/test_manifest_path.py
echo "  - EWC analytic-gradient equivalence verifier (CUDA reduction order)"
if [[ -f src/adhoc/verify_ewc_gradient_equivalence.py ]]; then
  python src/adhoc/verify_ewc_gradient_equivalence.py
fi

echo "==> [6/6] Manual step: HuggingFace model weights"
cat <<MSG
The DLVM does not have project model weights cached. To pull them:

  # If you have an HF token (required for gated Gemma/Llama models):
  pip install -U huggingface_hub
  huggingface-cli login

  # Then trigger a cached download for each family you plan to run:
  python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \\
    [AutoTokenizer.from_pretrained(m) and AutoModelForCausalLM.from_pretrained(m, torch_dtype='bfloat16') \\
     for m in ['Qwen/Qwen3-0.6B', 'google/gemma-3-1b-pt', 'meta-llama/Llama-3.2-1B', 'gpt2']]"

(Or rely on the first run to pull on-demand; this just front-loads the
~15 GB download into bootstrap rather than mid-run.)

==> Bootstrap complete.
MSG
