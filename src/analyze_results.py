from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIRECTORY = Path("results")
FIGURE_DIRECTORY = Path("figures")

FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)


def load_results() -> pd.DataFrame:
    files = sorted(RESULT_DIRECTORY.glob("run_seed_*_fraction_*.csv"))

    if not files:
        raise FileNotFoundError(
            "No result CSV files were found in results/."
        )

    frames = [pd.read_csv(file) for file in files]
    results = pd.concat(frames, ignore_index=True)

    required_columns = {
        "pipeline",
        "seed",
        "label_fraction",
        "roc_auc",
        "pr_auc",
        "f1",
        "training_time_seconds",
        "embedding_training_seconds",
        "embedding_extraction_seconds",
    }

    missing = required_columns.difference(results.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    return results


def aggregate_results(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "training_time_seconds",
        "inference_time_seconds",
        "peak_gpu_memory_mb",
        "embedding_training_seconds",
        "embedding_extraction_seconds",
        "embedding_storage_mb",
    ]

    available_metrics = [
        metric for metric in metrics if metric in results.columns
    ]

    summary = (
        results.groupby(["label_fraction", "pipeline"])[available_metrics]
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


def plot_predictive_utility(results: pd.DataFrame) -> None:
    summary = (
        results.groupby(["label_fraction", "pipeline"])["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    pipelines = summary["pipeline"].unique()

    for pipeline in pipelines:
        subset = summary[summary["pipeline"] == pipeline].sort_values(
            "label_fraction"
        )

        error_values = subset["std"].fillna(0.0).to_numpy()

        plt.errorbar(
            subset["label_fraction"] * 100,
            subset["mean"],
            yerr=error_values,
            marker="o",
            capsize=4,
            label=pipeline,
        )

    plt.xlabel("Available training labels (%)")
    plt.ylabel("Test PR-AUC")
    plt.title("Predictive utility under label scarcity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        FIGURE_DIRECTORY / "label_fraction_pr_auc.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        FIGURE_DIRECTORY / "label_fraction_pr_auc.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_cumulative_cost(results: pd.DataFrame) -> pd.DataFrame:
    full_label_results = results[
        np.isclose(results["label_fraction"], 1.0)
    ]

    means = full_label_results.groupby("pipeline").mean(
        numeric_only=True
    )

    required_pipelines = {
        "raw_mlp",
        "end_to_end_graphsage",
        "precomputed_graphsage_mlp",
    }

    missing = required_pipelines.difference(means.index)

    if missing:
        raise ValueError(
            f"Missing pipelines for cumulative analysis: {sorted(missing)}"
        )

    reuse_counts = np.arange(1, 21)

    raw_per_run = means.loc[
        "raw_mlp",
        "training_time_seconds",
    ]

    graph_per_run = means.loc[
        "end_to_end_graphsage",
        "training_time_seconds",
    ]

    precomputed_downstream_per_run = means.loc[
        "precomputed_graphsage_mlp",
        "training_time_seconds",
    ]

    embedding_training = means.loc[
        "precomputed_graphsage_mlp",
        "embedding_training_seconds",
    ]

    embedding_extraction = means.loc[
        "precomputed_graphsage_mlp",
        "embedding_extraction_seconds",
    ]

    one_time_precomputation = (
        embedding_training + embedding_extraction
    )

    raw_cost = reuse_counts * raw_per_run
    graph_cost = reuse_counts * graph_per_run
    precomputed_cost = (
        one_time_precomputation
        + reuse_counts * precomputed_downstream_per_run
    )

    cost_data = pd.DataFrame(
        {
            "reuse_count": reuse_counts,
            "raw_mlp_seconds": raw_cost,
            "end_to_end_graphsage_seconds": graph_cost,
            "precomputed_graphsage_mlp_seconds": precomputed_cost,
        }
    )

    cost_data.to_csv(
        RESULT_DIRECTORY / "cumulative_cost.csv",
        index=False,
    )

    plt.plot(
        reuse_counts,
        raw_cost,
        marker="o",
        label="Raw-feature MLP",
    )
    plt.plot(
        reuse_counts,
        graph_cost,
        marker="s",
        label="End-to-end GraphSAGE",
    )
    plt.plot(
        reuse_counts,
        precomputed_cost,
        marker="^",
        label="Precomputed GraphSAGE + MLP",
    )

    plt.xlabel("Number of downstream runs")
    plt.ylabel("Cumulative training cost (seconds)")
    plt.title("Cumulative computational cost under representation reuse")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY / "cumulative_cost.pdf",
        bbox_inches="tight",
    )
    plt.savefig(
        FIGURE_DIRECTORY / "cumulative_cost.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    if raw_per_run > precomputed_downstream_per_run:
        break_even_raw = one_time_precomputation / (
            raw_per_run - precomputed_downstream_per_run
        )
        print(
            "Break-even against raw MLP:",
            break_even_raw,
            "downstream runs",
        )
    else:
        print(
            "No finite break-even against raw MLP because the cached "
            "embedding MLP is not cheaper per run."
        )

    if graph_per_run > precomputed_downstream_per_run:
        break_even_graph = one_time_precomputation / (
            graph_per_run - precomputed_downstream_per_run
        )
        print(
            "Break-even against end-to-end GraphSAGE:",
            break_even_graph,
            "downstream runs",
        )
    else:
        print(
            "No finite break-even against end-to-end GraphSAGE."
        )

    return cost_data


def main() -> None:
    results = load_results()
    summary = aggregate_results(results)

    summary.to_csv(
        RESULT_DIRECTORY / "summary_mean_std.csv",
        index=False,
    )

    print("\nAggregated results")
    print(summary.to_string(index=False))

    plot_predictive_utility(results)
    cumulative_cost = plot_cumulative_cost(results)

    print("\nCumulative cost")
    print(cumulative_cost.to_string(index=False))

    print("\nFigures saved in figures/.")


if __name__ == "__main__":
    main()
