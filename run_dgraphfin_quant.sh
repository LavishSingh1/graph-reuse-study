#!/usr/bin/env bash
# Run from the repo root (same folder as run_all_dgraphfin.sh). Reuses the existing .venv-pyg.
set -euo pipefail
cd "$(dirname "$0")"
source .venv-pyg/bin/activate
[[ -f data/dgraphfin/raw/DGraphFin.zip || -d data/dgraphfin/processed ]] || { echo "Missing DGraphFin data"; exit 1; }
mkdir -p logs/dgraphfin_quant results/dgraphfin_quant embeddings/dgraphfin_quant
nvidia-smi > logs/dgraphfin_quant/nvidia_smi.txt
for seed in 42 123 717; do
  python run_dgraphfin_quantization.py --seed "$seed" --epochs 50 --patience 8 \
    --graph-batch-size 4096 --inference-batch-size 65536 --workers 8 \
    2>&1 | tee "logs/dgraphfin_quant/seed_${seed}.log"
done
python analyze_dgraphfin_quantization.py 2>&1 | tee logs/dgraphfin_quant/analysis.log
echo "Done. Summary: results/dgraphfin_quant/dgraphfin_quant_summary.csv"
