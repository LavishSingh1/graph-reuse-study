from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def flatten_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [
        "_".join(str(part) for part in column if str(part))
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]
    return frame


def main() -> None:
    files = sorted(
        Path("results/amazon/robustness").glob(
            "amazon_e2e_robustness_net_upu_seed_*.csv"
        )
    )
    if not files:
        raise FileNotFoundError("No Amazon robustness CSV files were found")

    results = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    summary = flatten_columns(
        results.groupby("learning_rate")[[
            "roc_auc",
            "pr_auc",
            "f1",
            "training_seconds",
            "best_epoch",
            "peak_gpu_memory_mb",
        ]]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    output = Path("results/amazon/amazon_e2e_robustness_summary.csv")
    summary.to_csv(output, index=False)

    print("Loaded files:", len(files))
    print("Loaded rows:", len(results))
    print("\nRobustness summary")
    print(summary.to_string(index=False))

    best_rate = summary.loc[summary["pr_auc_mean"].idxmax(), "learning_rate"]
    print("\nBest learning rate by mean PR-AUC:", best_rate)

    plot_data = results.groupby("learning_rate")["pr_auc"].agg(["mean", "std"])
    positions = np.arange(len(plot_data))
    labels = [f"{rate:g}" for rate in plot_data.index]

    figure_directory = Path("figures/amazon")
    figure_directory.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4.5))
    plt.bar(
        positions,
        plot_data["mean"],
        yerr=plot_data["std"].fillna(0.0),
        capsize=5,
    )
    plt.xticks(positions, labels)
    plt.xlabel("Learning rate")
    plt.ylabel("Test PR-AUC")
    plt.title("Amazon end-to-end GraphSAGE robustness")
    plt.tight_layout()
    plt.savefig(
        figure_directory / "amazon_e2e_learning_rate_robustness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.savefig(
        figure_directory / "amazon_e2e_learning_rate_robustness.pdf",
        bbox_inches="tight",
    )
    plt.close()

    print("Saved:", output)


if __name__ == "__main__":
    main()
