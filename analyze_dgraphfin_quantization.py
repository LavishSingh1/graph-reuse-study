"""Summarize results/dgraphfin_quant/*.csv -> mean/std table and break-even k* per (precision, fit_mode).

Cost model (paper Eq. 1-2): C_reuse(k) = (C_e + C_m) + k (C_l + C_p),  C_ref(k) = k C_b
  C_e + C_m = encoder + exact extraction + quantize + save ; C_l = load(+decode/transfer) ; C_p = fit
k* = (C_e+C_m)/(C_b - C_l - C_p) if C_b > C_l + C_p else none.
Per-seed k* is computed from that seed's own C_b; the table reports mean±std of components and k* from the means.
"""
import glob
from pathlib import Path
import numpy as np
import pandas as pd

files = sorted(glob.glob("results/dgraphfin_quant/dgraphfin_quant_seed_*.csv"))
if not files:
    raise SystemExit("No results found in results/dgraphfin_quant/")
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
df["k_star_seed"] = np.where(df.reference_e2e_seconds > df.per_workload_seconds,
                             df.upfront_seconds / (df.reference_e2e_seconds - df.per_workload_seconds), np.nan)
cols = ["pr_auc", "roc_auc", "storage_mb", "quantize_seconds", "save_seconds", "load_seconds",
        "training_seconds", "upfront_seconds", "per_workload_seconds", "reference_e2e_seconds",
        "embedding_rel_rmse", "peak_gpu_memory_mb"]
g = df.groupby(["fit_mode", "precision"], sort=False)
mean, std = g[cols].mean(), g[cols].std()
out = mean.add_suffix("_mean").join(std.add_suffix("_std"))
out["n_seeds"] = g.size()
m = mean
out["k_star_from_means"] = np.where(m.reference_e2e_seconds > m.per_workload_seconds,
                                    m.upfront_seconds / (m.reference_e2e_seconds - m.per_workload_seconds), np.nan)
out["cost_at_k10_reuse_over_ref"] = (m.upfront_seconds + 10 * m.per_workload_seconds) / (10 * m.reference_e2e_seconds)
Path("results/dgraphfin_quant").mkdir(parents=True, exist_ok=True)
out.to_csv("results/dgraphfin_quant/dgraphfin_quant_summary.csv")

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 50)
show = pd.DataFrame({
    "PR-AUC": mean.pr_auc.round(4).astype(str) + "±" + std.pr_auc.round(4).astype(str),
    "store MB": mean.storage_mb.round(0),
    "load s": mean.load_seconds.round(2),
    "fit Cp s": (mean.training_seconds.round(1).astype(str) + "±" + std.training_seconds.round(1).astype(str)),
    "upfront s": mean.upfront_seconds.round(1),
    "ref Cb s": mean.reference_e2e_seconds.round(1),
    "k*": out.k_star_from_means.round(1),
    "cost@k=10 (reuse/ref)": out.cost_at_k10_reuse_over_ref.round(2),
    "rel RMSE": mean.embedding_rel_rmse.map("{:.1e}".format),
})
print(show.to_string())
print("\nper-seed k* (NaN = no finite break-even):")
print(df.pivot_table(index=["fit_mode", "precision"], columns="seed", values="k_star_seed", sort=False, dropna=False).round(2).to_string())
