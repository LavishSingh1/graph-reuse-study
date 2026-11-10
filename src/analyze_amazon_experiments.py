from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LABELS = {
    "raw_features_mlp": "Raw features + MLP",
    "end_to_end_graphsage_mlp": "End-to-end GraphSAGE + MLP",
    "frozen_graphsage_mlp": "Frozen GraphSAGE + MLP",
    "raw_plus_frozen_embedding_mlp": "Raw + frozen embedding + MLP",
}


def load_files(pattern):
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")
    return pd.concat([pd.read_csv(path) for path in files], ignore_index=True)


def flatten(frame):
    frame = frame.copy()
    frame.columns = ["_".join(str(part) for part in column if str(part)) if isinstance(column, tuple) else column for column in frame.columns]
    return frame


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reuse-relation", required=True)
    args = parser.parse_args()

    output = Path("results/amazon")
    figures = Path("figures/amazon")
    figures.mkdir(parents=True, exist_ok=True)

    scan = load_files("results/amazon/relation_scan/amazon_relation_scan_seed_*.csv")
    reuse = load_files(f"results/amazon/reuse/amazon_reuse_{args.reuse_relation}_seed_*.csv")

    scan_summary = flatten(scan.groupby("graph_name")[["roc_auc", "pr_auc", "f1", "training_seconds", "peak_gpu_memory_mb"]].agg(["mean", "std", "count"]).reset_index())
    reuse_summary = flatten(reuse.groupby("pipeline")[["roc_auc", "pr_auc", "f1", "training_seconds", "first_run_seconds", "peak_gpu_memory_mb", "one_time_encoder_seconds", "embedding_extraction_seconds", "embedding_storage_mb"]].agg(["mean", "std", "count"]).reset_index())
    scan_summary.to_csv(output / "amazon_relation_scan_summary.csv", index=False)
    reuse_summary.to_csv(output / f"amazon_reuse_{args.reuse_relation}_summary.csv", index=False)

    print("\nRelation scan")
    print(scan_summary.to_string(index=False))
    print("\nReuse study")
    print(reuse_summary.to_string(index=False))

    scan_plot = scan.groupby("graph_name")["pr_auc"].agg(["mean", "std"]).sort_values("mean")
    plt.figure(figsize=(8, 5))
    plt.barh(np.arange(len(scan_plot)), scan_plot["mean"], xerr=scan_plot["std"].fillna(0), capsize=4)
    plt.yticks(np.arange(len(scan_plot)), scan_plot.index)
    plt.xlabel("Test PR-AUC")
    plt.title("Amazon relation scan")
    plt.tight_layout()
    plt.savefig(figures / "amazon_relation_scan.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures / "amazon_relation_scan.pdf", bbox_inches="tight")
    plt.close()

    means = reuse.groupby("pipeline").mean(numeric_only=True)
    counts = np.arange(1, 21)
    plt.figure(figsize=(8, 5))
    cumulative_rows = []
    for pipeline, row in means.iterrows():
        one_time = row.get("one_time_encoder_seconds", 0.0) + row.get("embedding_extraction_seconds", 0.0)
        costs = one_time + counts * row["training_seconds"]
        plt.plot(counts, costs, marker="o", label=LABELS.get(pipeline, pipeline))
        for count, cost in zip(counts, costs):
            cumulative_rows.append({"pipeline": pipeline, "reuse_count": count, "cumulative_seconds": cost})
    pd.DataFrame(cumulative_rows).to_csv(output / f"amazon_cumulative_cost_{args.reuse_relation}.csv", index=False)
    plt.xlabel("Number of downstream runs")
    plt.ylabel("Cumulative training cost (seconds)")
    plt.title(f"Amazon reuse cost on {args.reuse_relation}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures / f"amazon_cumulative_cost_{args.reuse_relation}.png", dpi=300, bbox_inches="tight")
    plt.savefig(figures / f"amazon_cumulative_cost_{args.reuse_relation}.pdf", bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
