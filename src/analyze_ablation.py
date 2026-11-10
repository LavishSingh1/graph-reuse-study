from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIRECTORY = Path("results/ablation")
FIGURE_DIRECTORY = Path("figures/ablation")

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


PIPELINE_LABELS = {
    "raw_features_linear":
        "Raw features + linear",
    "raw_features_mlp":
        "Raw features + MLP",
    "end_to_end_graphsage_linear":
        "End-to-end GraphSAGE + linear",
    "end_to_end_graphsage_mlp":
        "End-to-end GraphSAGE + MLP",
    "frozen_supervised_graphsage_linear":
        "Frozen GraphSAGE + linear",
    "frozen_supervised_graphsage_mlp":
        "Frozen GraphSAGE + MLP",
    "raw_plus_embedding_mlp":
        "Raw + embedding + MLP",
    "random_graphsage_embedding_mlp":
        "Random GraphSAGE + MLP",
    "neighbor_mean_features_mlp":
        "Neighbor mean + MLP",
}


def load_results() -> pd.DataFrame:
    files = sorted(
        RESULT_DIRECTORY.glob(
            "matched_ablation_seed_*.csv"
        )
    )

    if not files:
        raise FileNotFoundError(
            "No ablation CSV files were found."
        )

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    print("Loaded files:", len(files))
    print("Loaded rows:", len(results))

    return results


def aggregate_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "training_seconds",
        "inference_seconds",
        "peak_gpu_memory_mb",
        "best_epoch",
        "trainable_parameters",
        "encoder_training_seconds",
        "embedding_extraction_seconds",
        "embedding_storage_mb",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in results.columns
    ]

    summary = (
        results.groupby(
            [
                "encoder_label_fraction",
                "downstream_label_fraction",
                "pipeline",
            ]
        )[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    summary.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part) != ""
        )
        if isinstance(column, tuple)
        else column
        for column in summary.columns
    ]

    return summary


def plot_full_label_pr_auc(
    results: pd.DataFrame,
) -> None:
    subset = results[
        np.isclose(
            results["encoder_label_fraction"],
            1.0,
        )
        & np.isclose(
            results["downstream_label_fraction"],
            1.0,
        )
    ]

    summary = (
        subset.groupby("pipeline")["pr_auc"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    labels = [
        PIPELINE_LABELS.get(
            pipeline,
            pipeline,
        )
        for pipeline in summary.index
    ]

    positions = np.arange(len(summary))

    plt.figure(figsize=(10, 7))

    plt.barh(
        positions,
        summary["mean"],
        xerr=summary["std"].fillna(0.0),
        capsize=4,
    )

    plt.yticks(
        positions,
        labels,
    )

    plt.xlabel("Test PR-AUC")
    plt.title(
        "Matched-head and representation ablations"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "matched_ablation_pr_auc.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "matched_ablation_pr_auc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_full_label_cost_quality(
    results: pd.DataFrame,
) -> None:
    subset = results[
        np.isclose(
            results["encoder_label_fraction"],
            1.0,
        )
        & np.isclose(
            results["downstream_label_fraction"],
            1.0,
        )
    ]

    summary = (
        subset.groupby("pipeline")
        .agg(
            pr_auc=("pr_auc", "mean"),
            downstream_seconds=(
                "training_seconds",
                "mean",
            ),
            encoder_seconds=(
                "encoder_training_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    summary["first_run_seconds"] = (
        summary["downstream_seconds"]
        + summary["encoder_seconds"]
    )

    plt.figure(figsize=(10, 7))

    for _, row in summary.iterrows():
        label = PIPELINE_LABELS.get(
            row["pipeline"],
            row["pipeline"],
        )

        plt.scatter(
            row["first_run_seconds"],
            row["pr_auc"],
            s=70,
        )

        plt.annotate(
            label,
            (
                row["first_run_seconds"],
                row["pr_auc"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )

    plt.xlabel(
        "First-run training cost (seconds)"
    )
    plt.ylabel("Test PR-AUC")
    plt.title(
        "Predictive utility versus first-run cost"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "quality_vs_first_run_cost.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "quality_vs_first_run_cost.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_downstream_label_efficiency(
    results: pd.DataFrame,
) -> None:
    subset = results[
        np.isclose(
            results["encoder_label_fraction"],
            1.0,
        )
    ]

    selected_pipelines = [
        "raw_features_mlp",
        "end_to_end_graphsage_mlp",
        "frozen_supervised_graphsage_mlp",
        "raw_plus_embedding_mlp",
    ]

    subset = subset[
        subset["pipeline"].isin(
            selected_pipelines
        )
    ]

    summary = (
        subset.groupby(
            [
                "downstream_label_fraction",
                "pipeline",
            ]
        )["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(9, 6))

    for pipeline in selected_pipelines:
        pipeline_data = summary[
            summary["pipeline"] == pipeline
        ].sort_values(
            "downstream_label_fraction"
        )

        if pipeline_data.empty:
            continue

        plt.errorbar(
            pipeline_data[
                "downstream_label_fraction"
            ] * 100,
            pipeline_data["mean"],
            yerr=pipeline_data[
                "std"
            ].fillna(0.0),
            marker="o",
            capsize=4,
            label=PIPELINE_LABELS.get(
                pipeline,
                pipeline,
            ),
        )

    plt.xlabel(
        "Available downstream labels (%)"
    )
    plt.ylabel("Test PR-AUC")
    plt.title(
        "Fixed supervised representation under "
        "downstream label scarcity"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "fixed_embedding_label_efficiency.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "fixed_embedding_label_efficiency.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    results = load_results()
    summary = aggregate_results(results)

    summary_file = (
        RESULT_DIRECTORY
        / "matched_ablation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\nSummary")
    print(summary.to_string(index=False))
    print("\nSaved:", summary_file)

    plot_full_label_pr_auc(results)
    plot_full_label_cost_quality(results)
    plot_downstream_label_efficiency(results)

    print(
        "\nFigures saved in figures/ablation/."
    )


if __name__ == "__main__":
    main()
