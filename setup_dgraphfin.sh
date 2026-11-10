#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential

python3.11 -m venv .venv-pyg
source .venv-pyg/bin/activate
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
python -m pip install torch-geometric==2.6.1
python -m pip install pyg_lib torch_scatter torch_sparse \
  -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
python -m pip install numpy==1.26.4 scipy==1.13.1 scikit-learn==1.5.2 \
  pandas==2.2.3 matplotlib==3.9.2 psutil==6.0.0 tqdm==4.66.5

python - <<'PY'
import torch
import torch_geometric
import pyg_lib
print("PyTorch:", torch.__version__)
print("PyG:", torch_geometric.__version__)
print("pyg-lib:", pyg_lib.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
print("GPU:", torch.cuda.get_device_name(0))
PY

mkdir -p data/dgraphfin/raw results/dgraphfin embeddings/dgraphfin figures/dgraphfin logs/dgraphfin
python -m pip freeze > requirements-dgraphfin-lock.txt

echo
echo "Environment ready. Place DGraphFin.zip at:"
echo "$PROJECT_DIR/data/dgraphfin/raw/DGraphFin.zip"
