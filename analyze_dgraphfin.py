from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def flatten(frame):
    frame = frame.copy()
    frame.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col for col in frame.columns]
    return frame


files = sorted(Path("results/dgraphfin").glob("dgraphfin_seed_*.csv"))
if not files:
    raise FileNotFoundError("No DGraph-Fin result CSV files found")
results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
metrics = ["roc_auc", "pr_auc", "f1", "training_seconds", "inference_seconds", "first_run_seconds", "peak_gpu_memory_mb", "one_time_encoder_seconds", "embedding_extraction_seconds", "embedding_save_seconds", "embedding_load_seconds", "embedding_storage_mb", "max_process_rss_mb"]
metrics = [m for m in metrics if m in results.columns]
summary = flatten(results.groupby("pipeline")[metrics].agg(["mean", "std", "count"]).reset_index())
summary_path = Path("results/dgraphfin/dgraphfin_summary.csv")
summary.to_csv(summary_path, index=False)
print(summary.to_string(index=False))

means = results.groupby("pipeline").mean(numeric_only=True)
means = means.fillna(0.0)
counts = np.arange(1, 21)
rows = []
plt.figure(figsize=(8, 5))
labels = {"raw_features_mlp": "Raw features + MLP", "end_to_end_graphsage_mlp": "End-to-end GraphSAGE + MLP", "frozen_graphsage_mlp": "Frozen GraphSAGE + MLP"}
for pipeline, row in means.iterrows():
    one_time = row.get("one_time_encoder_seconds", 0) + row.get("embedding_extraction_seconds", 0) + row.get("embedding_save_seconds", 0)
    repeated = row["training_seconds"] + row.get("embedding_load_seconds", 0)
    costs = one_time + counts * repeated
    plt.plot(counts, costs, marker="o", label=f"{labels.get(pipeline,pipeline)} (PR-AUC {row['pr_auc']:.3f})")
    rows.extend({"pipeline": pipeline, "reuse_count": int(k), "cumulative_seconds": float(c)} for k, c in zip(counts, costs))
pd.DataFrame(rows).to_csv("results/dgraphfin/dgraphfin_cumulative_cost.csv", index=False)
plt.xlabel("Number of downstream runs")
plt.ylabel("Cumulative measured cost (seconds)")
plt.title("DGraph-Fin representation reuse")
plt.legend()
plt.tight_layout()
Path("figures/dgraphfin").mkdir(parents=True, exist_ok=True)
plt.savefig("figures/dgraphfin/dgraphfin_cumulative_cost.png", dpi=300, bbox_inches="tight")
plt.savefig("figures/dgraphfin/dgraphfin_cumulative_cost.pdf", bbox_inches="tight")
plt.close()
print("saved", summary_path)
