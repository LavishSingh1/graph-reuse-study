# The Cost of Reuse: When and at What Scale Do Precomputed Graph Representations Pay Off?

Code, raw measurements, logs and figures for the HiPC 2026 Student Research Symposium paper
*"The Cost of Reuse: When and at What Scale Do Precomputed Graph Representations Pay Off?"*
(Lavish Singh, Pranjal K. Yadav, Rishabh Vats).

We treat a supervised GraphSAGE embedding matrix as a **reusable artifact** and account for the full
lifecycle cost: encoder training (C_e), exact full-graph materialization + serialization (C_m), loading (C_l),
downstream fitting (C_p), against a reference pipeline (C_b):

```
C_reuse(k) = C_e + C_m + k (C_l + C_p),   C_ref(k) = k C_b,   k* = (C_e + C_m) / (C_b - C_l - C_p)   (finite only if C_b > C_l + C_p)
```

| Dataset | Nodes | Reuse pipeline vs. reference | Outcome (paper) |
|---|---|---|---|
| YelpChi (shared-user relation, RUR) | 45,954 | raw + frozen vs. end-to-end GraphSAGE | k* ~ 2 |
| Amazon (net_upu) | 11,944 | raw + frozen vs. raw-feature MLP | k* ~ 4 |
| DGraph-Fin | 3.7 M | frozen vs. neighbor-sampled end-to-end | no finite break-even |

> Wall-clock break-even points depend on hardware, implementation and stopping behaviour. All reported
> numbers were measured on one NVIDIA RTX A5000 (24 GB), Intel i9-14900K (32 logical cores), 62 GB RAM,
> Ubuntu 22.04. Raw per-seed measurements are in `results/` so every table and figure can be recomputed
> without re-running anything.

---

## 1. Repository layout

```
.
├── README.md
├── requirements-lock.txt             # DGL environment (YelpChi, Amazon)           [pip freeze]
├── requirements-dgraphfin-lock.txt   # PyG environment (DGraph-Fin)                [pip freeze]
├── setup_dgraphfin.sh                # creates .venv-pyg (torch 2.4.1+cu124, PyG 2.6.1)
├── system_info/                      # lscpu, nvidia-smi, OS, library versions of the measurement machine
│
├── src/                              # YelpChi + Amazon experiments (DGL)
│   ├── check_dataset.py              #   downloads / inspects YelpChi
│   ├── run_experiment.py             #   label-fraction sweep: raw MLP / end-to-end SAGE / precomputed SAGE+MLP
│   ├── run_matched_ablation.py       #   capacity-matched ablation, fixed vs. retrained encoder (encoder vs. downstream label fraction)
│   ├── run_relation_subset_edge_budget.py   # relation subsets (RUR/RTR/RSR/...) and equal-edge-budget comparison
│   ├── run_random_relation_experiments.py   # random-relation / random-projection / random-GraphSAGE controls
│   ├── run_amazon_experiments.py     #   Amazon: inspect | relation scan | reuse study
│   ├── run_amazon_robustness.py      #   Amazon end-to-end learning-rate / epoch robustness
│   └── analyze_*.py                  #   one analysis script per experiment -> figures/ and summary CSVs
│
├── run_dgraphfin_pipeline.py         # DGraph-Fin: raw MLP, sampled end-to-end SAGE, exact layer-wise materialization, frozen MLP
├── run_all_dgraphfin.sh              # 3 seeds + analysis
├── analyze_dgraphfin.py
├── run_dgraphfin_quantization.py     # DGraph-Fin FP32 / FP16 / INT8 embedding experiment (reviewer-requested)
├── analyze_dgraphfin_quantization.py
├── run_dgraphfin_quant.sh
│
├── results/                          # raw per-seed CSVs + summary CSVs (the source of every number in the paper)
├── logs/                             # full stdout of every run, nvidia-smi, pip freeze
├── figures/                          # generated figures (PNG + PDF)
├── data/                             # NOT tracked (see Section 3)
└── embeddings/                       # NOT tracked (regenerated; up to ~1 GB)
```

## 2. Environments

Two environments were used (different graph libraries).

**A. YelpChi / Amazon (DGL)** — Python 3.11, PyTorch 2.4.0+cu121, DGL 2.4.0, NumPy 1.26.4, scikit-learn 1.5.2, pandas 2.2.3.
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt \
    -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html   # DGL wheel is not on PyPI; match the CUDA tag to your driver
```

**B. DGraph-Fin (PyG)** — Python 3.11, PyTorch 2.4.1+cu124, PyG 2.6.1, pyg-lib, torch-scatter/sparse.
```bash
bash setup_dgraphfin.sh        # creates .venv-pyg and verifies CUDA (needs sudo for apt packages)
```
A CUDA GPU is required. All scripts must be launched **from the repository root**.

## 3. Data

| Dataset | How to get it |
|---|---|
| YelpChi | Downloaded automatically by DGL (`FraudYelpDataset`) into `data/` on first use (`python src/check_dataset.py`). |
| Amazon | Downloaded automatically by DGL (`FraudAmazonDataset`) on first use of `src/run_amazon_experiments.py --mode inspect`. |
| DGraph-Fin | Manual: download `DGraphFin.zip` from <https://dgraph.xinye.com/dataset> and place it at `data/dgraphfin/raw/DGraphFin.zip`. PyG processes it on first run. |

Datasets are not redistributed here; follow their own licenses/terms.

## 4. Reproducing the experiments

Seeds: YelpChi and Amazon use five seeds (42, 123, 717, 2026, 31415); the YelpChi label-fraction sweep and DGraph-Fin use
three (42, 123, 717). Hyperparameters (all datasets): 64-d hidden/embedding, 2-layer mean GraphSAGE, dropout 0.3, Adam, lr 1e-3,
weight decay 1e-4, FP32 embeddings. YelpChi/Amazon: full-graph training, <=300 epochs, patience 30.
DGraph-Fin: fanouts (15, 10), batch 4096, <=50 epochs, patience 8, 8 DataLoader workers, exact layer-wise materialization with full neighborhoods.
PR-AUC is the primary metric.

### 4.1 YelpChi (environment A)
```bash
python src/check_dataset.py

# label-fraction sweep (raw MLP, end-to-end GraphSAGE, precomputed GraphSAGE+MLP)
for s in 42 123 717; do for f in 0.01 0.05 0.10 0.25 0.50 0.75 1.00; do
  python src/run_experiment.py --seed $s --label-fraction $f --pipeline all
done; done
python src/analyze_results.py                     # -> figures/cumulative_cost.*, label_fraction_pr_auc.*, results/cumulative_cost.csv

# capacity-matched ablation: fixed vs. retrained encoder
python src/run_matched_ablation.py --seed 42 --encoder-label-fraction 1.0 --downstream-label-fraction 0.10   # repeat over seeds/fractions
python src/analyze_ablation.py                    # -> figures/ablation/

# relation subsets and equal-edge-budget comparison (source of the RUR / RTR / RSR numbers)
for s in 42 123 717 2026 31415; do python src/run_relation_subset_edge_budget.py --experiment all --seed $s; done
python src/analyze_relation_subset_edge_budget.py # -> figures/relation_subsets/, figures/edge_budgets/

# random-relation / random-projection / random-GraphSAGE controls
for s in 42 123 717 2026 31415; do python src/run_random_relation_experiments.py --experiment all --seed $s; done
python src/analyze_random_relation_experiments.py # -> figures/next_experiments/
```

### 4.2 Amazon (environment A)
```bash
python src/run_amazon_experiments.py --mode inspect
for s in 42 123 717 2026 31415; do
  python src/run_amazon_experiments.py --mode scan  --seed $s                       # relation scan
  python src/run_amazon_experiments.py --mode reuse --reuse-relation net_upu --seed $s
  python src/run_amazon_robustness.py  --seed $s --relation net_upu                 # end-to-end LR/epoch robustness
done
python src/analyze_amazon_experiments.py --reuse-relation net_upu   # -> figures/amazon/, results/amazon/*summary.csv
python src/analyze_amazon_robustness.py
```

### 4.3 DGraph-Fin (environment B)
```bash
bash run_all_dgraphfin.sh        # seeds 42/123/717 -> results/dgraphfin/, then analyze_dgraphfin.py
```
Writes `results/dgraphfin/dgraphfin_summary.csv` and `figures/dgraphfin/dgraphfin_cumulative_cost.*`.
Expected (paper): sampled end-to-end ~13.9 s, exact materialization ~82 s, 903 MB FP32 artifact, frozen downstream fit ~40 s,
no finite break-even.

### 4.4 DGraph-Fin embedding quantization (environment B)
```bash
bash run_dgraphfin_quant.sh      # -> results/dgraphfin_quant/dgraphfin_quant_summary.csv
```
Per seed: trains the sampled encoder once, materializes exact FP32 embeddings once, then for each of FP32 / FP16 / INT8
(symmetric, per-column scale) quantizes, serializes, loads and fits the same MLP head. Each precision is fit in two modes:
`dataloader` (identical to the paper's pipeline: CPU tensor + `torch.utils.data.DataLoader`) and `gpu` (embeddings resident on the GPU
in their stored dtype, batches built on-device; separates DataLoader overhead from true dense-feature consumption cost).
The analysis prints storage, PR-AUC, per-seed and mean k*, and relative embedding RMSE.

## 5. Outputs and where the paper's numbers come from

| Paper element | Source files |
|---|---|
| Table I, YelpChi row; RUR vs. all-relations numbers | `results/relation_subsets/relation_subset_summary.csv`, `results/edge_budgets/edge_budget_summary.csv` |
| Table I, Amazon row | `results/amazon/amazon_reuse_net_upu_summary.csv`, `results/amazon/amazon_cumulative_cost_net_upu.csv` |
| Table I, DGraph-Fin row; Fig. 1 | `results/dgraphfin/dgraphfin_summary.csv`, `results/dgraphfin/dgraphfin_cumulative_cost.csv` |
| Fixed-encoder / label-efficiency ablation | `results/ablation/` |
| Random-projection / random-GraphSAGE controls | `results/next_experiments/random_relation_summary.csv` |
| Quantization results | `results/dgraphfin_quant/` |

`run_*.py` emit one CSV per seed/configuration (training time, peak GPU memory, ROC-AUC, PR-AUC, F1, inference time,
encoder / extraction / storage time and size); `analyze_*.py` aggregate them to mean +/- std and draw the figures.
Per-run stdout, `nvidia-smi` and `pip freeze` are in `logs/`.

## 6. Notes on timing methodology

- Wall-clock times are measured with `time.perf_counter()` after `torch.cuda.synchronize()`; peak GPU memory via `torch.cuda.max_memory_allocated()`.
- C_m includes exact all-node layer-wise extraction **and** serialization (`torch.save`); C_l is `torch.load`.
- Break-even points use component times from the same run family; run-to-run variance is not negligible for sub-second fits
  (Amazon), so report ranges, not single points.
- Absolute times are hardware-specific; the structural claim (retaining utility != saving compute) is what the study supports.
- Rerunning on other hardware will move k* but not the cost decomposition.

## 7. Limitations (as in the paper)

Transductive, supervised two-layer mean GraphSAGE; fraud-oriented node classification; static graphs; same-task reuse
(no cross-task transfer); single GPU; FP32 (FP16/INT8 only for DGraph-Fin, Section 4.4).

## 8. Citation

```bibtex
@inproceedings{singh2026costofreuse,
  title     = {The Cost of Reuse: When and at What Scale Do Precomputed Graph Representations Pay Off?},
  author    = {Singh, Lavish and Yadav, Pranjal K. and Vats, Rishabh},
  booktitle = {HiPC 2026 Student Research Symposium},
  year      = {2026}
}
```

## 9. License

Code: Apache License 2.0 (see `LICENSE` and `NOTICE`). Datasets (YelpChi, Amazon, DGraph-Fin) are not
redistributed and are subject to their original licenses.
