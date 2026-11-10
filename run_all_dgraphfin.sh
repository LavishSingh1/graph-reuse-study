#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -d .venv-pyg ]]; then
  echo "Missing .venv-pyg. Run: bash setup_dgraphfin.sh"
  exit 1
fi
source .venv-pyg/bin/activate

ZIP="data/dgraphfin/raw/DGraphFin.zip"
if [[ ! -f "$ZIP" ]]; then
  echo "Missing $ZIP"
  echo "Download DGraphFin.zip from https://dgraph.xinye.com/dataset and place it at $ZIP"
  exit 1
fi

mkdir -p logs/dgraphfin results/dgraphfin embeddings/dgraphfin figures/dgraphfin
nvidia-smi > logs/dgraphfin/nvidia_smi.txt
python -m pip freeze > logs/dgraphfin/requirements.txt

for seed in 42 123 717; do
  echo "Starting DGraph-Fin seed $seed"
  /usr/bin/time -v python run_dgraphfin_pipeline.py \
    --seed "$seed" \
    --epochs 50 \
    --patience 8 \
    --graph-batch-size 4096 \
    --inference-batch-size 65536 \
    --workers 8 \
    2>&1 | tee "logs/dgraphfin/seed_${seed}.log"
done

python analyze_dgraphfin.py 2>&1 | tee logs/dgraphfin/analysis.log

echo "DGraph-Fin pipeline completed."
echo "Summary: results/dgraphfin/dgraphfin_summary.csv"
echo "Figure: figures/dgraphfin/dgraphfin_cumulative_cost.png"
