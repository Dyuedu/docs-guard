#!/usr/bin/env bash
# Bootstrap a rented Ubuntu 22.04 GPU box (T4) for training + benchmarking.
#
#   REPO_URL=<git url> DATA_URL=<tar.gz url of data/> bash scripts/setup_cloud.sh
#
# Assumes the NVIDIA driver is preinstalled (most GPU images ship it); check with nvidia-smi.
set -euo pipefail

: "${REPO_URL:?set REPO_URL}"
DATA_URL="${DATA_URL:-}"
WORKDIR="${WORKDIR:-$HOME/doc-guard}"

nvidia-smi || { echo "NVIDIA driver missing: install it before running this script" >&2; exit 1; }

# Docker + NVIDIA Container Toolkit (needed for Triton and trtexec builds)
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
fi
if ! dpkg -s nvidia-container-toolkit >/dev/null 2>&1; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
fi

# Code + Python env
[ -d "$WORKDIR/.git" ] || git clone "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"
sudo apt-get install -y python3-venv
python3 -m venv .venv
. .venv/bin/activate
pip install -U pip
pip install -e ".[train,export,dev]"

# Data (packed locally with: tar czf data.tgz data/)
if [ -n "$DATA_URL" ] && [ ! -d data/processed ]; then
  curl -fL "$DATA_URL" -o /tmp/data.tgz && tar xzf /tmp/data.tgz && rm /tmp/data.tgz
fi

pytest -q
echo "Ready. Record versions for the report:"
nvidia-smi --query-gpu=name,driver_version --format=csv
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
