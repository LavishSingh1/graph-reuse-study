from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUBSET_DIRECTORY = Path(
    "results/relation_subsets"
)

BUDGET_DIRECTORY = Path(
    "results/edge_budgets"
)

SUBSET_FIGURE_DIRECTORY = Path(
    "figures/relation_subsets"
)

BUDGET_FIGURE_DIRECTORY = Path(
    "figures/edge_budgets"
)

SUBSET_FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)

BUDGET_FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


PIPELINE_LABELS = {
    "end_to_end_graphsage_mlp":
        "End-to-end GraphSAGE",
    "frozen_graphsage_mlp":
        "Frozen GraphSAGE",
    "raw_plus_frozen_embedding_mlp":
        "Raw + frozen embedding",
}


GRAPH_LABELS = {
    "net_rsr":
        "RSR",
    "net_rtr":
        "RTR",
    "net_rur":
        "RUR",
    "net_rsr__net_rtr":
        "RSR + RTR",
    "net_rsr__net_rur":
        "RSR + RUR",
    "net_rtr__net_rur":
        "RTR + RUR",
    "net_rsr__net_rtr__net_rur":
        "All relations",
}


def load_results(
    directory: Path,
    pattern: str,
) -> pd.DataFrame:
    files = sorted(
        directory.glob(pattern)
    )

    if not files:
        raise FileNotFoundError(
            f"No files found in {directory} "
            f"matching {pattern}"
        )

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    results = pd.concat(
        frames,
        ignore_index=True,
    )

    print(
        f"Loaded {len(files)} files "
        f"and {len(results)} rows from "
        f"{directory}"
    )

    return results


def flatten_columns(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    frame = frame.copy()

    frame.columns = [
        "_".join(
            str(part)
            for part in column
            if str(part) != ""
        )
        if isinstance(column, tuple)
        else column
        for column in frame.columns
    ]

    return frame


def summarize_subset_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "training_seconds",
        "inference_seconds",
        "first_run_seconds",
        "peak_gpu_memory_mb",
        "one_time_encoder_seconds",
        "representation_extraction_seconds",
        "embedding_storage_mb",
        "graph_edges_before_self_loops",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in results.columns
    ]

    summary = (
        results.groupby(
            [
                "graph_name",
                "pipeline",
            ]
        )[available_metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return flatten_columns(summary)


def summarize_budget_results(
    results: pd.DataFrame,
) -> pd.DataFrame:
    metrics = [
        "roc_auc",
        "pr_auc",
        "f1",
        "training_seconds",
        "inference_seconds",
        "peak_gpu_memory_mb",
        "graph_edges_after_self_loops",
    ]

    summary = (
        results.groupby(
            [
                "relations",
                "edge_budget",
            ]
        )[metrics]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    return flatten_columns(summary)


def plot_relation_subsets(
    results: pd.DataFrame,
) -> None:
    summary = (
        results.groupby(
            [
                "graph_name",
                "pipeline",
            ]
        )["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    graph_order = [
        "net_rur",
        "net_rtr",
        "net_rsr",
        "net_rtr__net_rur",
        "net_rsr__net_rur",
        "net_rsr__net_rtr",
        "net_rsr__net_rtr__net_rur",
    ]

    graph_order = [
        graph_name
        for graph_name in graph_order
        if graph_name in set(summary["graph_name"])
    ]

    pipelines = [
        "end_to_end_graphsage_mlp",
        "frozen_graphsage_mlp",
        "raw_plus_frozen_embedding_mlp",
    ]

    positions = np.arange(len(graph_order))
    bar_width = 0.25

    plt.figure(figsize=(12, 6))

    for pipeline_index, pipeline in enumerate(pipelines):
        pipeline_data = (
            summary[
                summary["pipeline"] == pipeline
            ]
            .set_index("graph_name")
            .reindex(graph_order)
        )

        offsets = (
            pipeline_index
            - (len(pipelines) - 1) / 2
        ) * bar_width

        plt.bar(
            positions + offsets,
            pipeline_data["mean"],
            width=bar_width,
            yerr=pipeline_data[
                "std"
            ].fillna(0.0),
            capsize=3,
            label=PIPELINE_LABELS[pipeline],
        )

    plt.xticks(
        positions,
        [
            GRAPH_LABELS.get(
                graph_name,
                graph_name,
            )
            for graph_name in graph_order
        ],
        rotation=20,
    )

    plt.xlabel("Relation subset")
    plt.ylabel("Test PR-AUC")
    plt.title(
        "Predictive utility across YelpChi relation subsets"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        SUBSET_FIGURE_DIRECTORY
        / "relation_subset_pr_auc.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        SUBSET_FIGURE_DIRECTORY
        / "relation_subset_pr_auc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_subset_cost_quality(
    results: pd.DataFrame,
) -> None:
    summary = (
        results.groupby(
            [
                "graph_name",
                "pipeline",
            ]
        )
        .agg(
            pr_auc=("pr_auc", "mean"),
            first_run_seconds=(
                "first_run_seconds",
                "mean",
            ),
            downstream_seconds=(
                "training_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    plt.figure(figsize=(11, 7))

    markers = {
        "end_to_end_graphsage_mlp": "o",
        "frozen_graphsage_mlp": "s",
        "raw_plus_frozen_embedding_mlp": "^",
    }

    for pipeline, pipeline_data in summary.groupby(
        "pipeline"
    ):
        plt.scatter(
            pipeline_data["first_run_seconds"],
            pipeline_data["pr_auc"],
            marker=markers.get(pipeline, "o"),
            s=80,
            label=PIPELINE_LABELS.get(
                pipeline,
                pipeline,
            ),
        )

        for _, row in pipeline_data.iterrows():
            plt.annotate(
                GRAPH_LABELS.get(
                    row["graph_name"],
                    row["graph_name"],
                ),
                (
                    row["first_run_seconds"],
                    row["pr_auc"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    plt.xlabel("First-run cost (seconds)")
    plt.ylabel("Test PR-AUC")
    plt.title(
        "Relation-subset utility versus first-run cost"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        SUBSET_FIGURE_DIRECTORY
        / "relation_subset_cost_quality.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        SUBSET_FIGURE_DIRECTORY
        / "relation_subset_cost_quality.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_edge_budget_pr_auc(
    results: pd.DataFrame,
) -> None:
    summary = (
        results.groupby(
            [
                "relations",
                "edge_budget",
            ]
        )["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for relation_name, relation_data in summary.groupby(
        "relations"
    ):
        relation_data = relation_data.sort_values(
            "edge_budget"
        )

        plt.errorbar(
            relation_data["edge_budget"],
            relation_data["mean"],
            yerr=relation_data[
                "std"
            ].fillna(0.0),
            marker="o",
            capsize=4,
            label=relation_name,
        )

    plt.xscale("log")
    plt.xlabel("Stored directed-edge budget")
    plt.ylabel("Test PR-AUC")
    plt.title(
        "Predictive utility under matched edge budgets"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_pr_auc.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_pr_auc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_edge_budget_memory(
    results: pd.DataFrame,
) -> None:
    summary = (
        results.groupby(
            [
                "relations",
                "edge_budget",
            ]
        )["peak_gpu_memory_mb"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for relation_name, relation_data in summary.groupby(
        "relations"
    ):
        relation_data = relation_data.sort_values(
            "edge_budget"
        )

        plt.plot(
            relation_data["edge_budget"],
            relation_data["mean"],
            marker="o",
            label=relation_name,
        )

    plt.xscale("log")
    plt.xlabel("Stored directed-edge budget")
    plt.ylabel("Peak allocated GPU memory (MB)")
    plt.title(
        "GPU memory under matched edge budgets"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_gpu_memory.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_gpu_memory.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_edge_budget_time(
    results: pd.DataFrame,
) -> None:
    summary = (
        results.groupby(
            [
                "relations",
                "edge_budget",
            ]
        )["training_seconds"]
        .agg(["mean", "std"])
        .reset_index()
    )

    plt.figure(figsize=(10, 6))

    for relation_name, relation_data in summary.groupby(
        "relations"
    ):
        relation_data = relation_data.sort_values(
            "edge_budget"
        )

        plt.plot(
            relation_data["edge_budget"],
            relation_data["mean"],
            marker="o",
            label=relation_name,
        )

    plt.xscale("log")
    plt.xlabel("Stored directed-edge budget")
    plt.ylabel("Training time (seconds)")
    plt.title(
        "GraphSAGE training time under matched edge budgets"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_training_time.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        BUDGET_FIGURE_DIRECTORY
        / "edge_budget_training_time.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def print_matched_budget_table(
    budget_results: pd.DataFrame,
    target_budget: int = 98_630,
) -> None:
    matched = budget_results[
        budget_results["edge_budget"] == target_budget
    ]

    if matched.empty:
        print(
            f"\nNo results found for budget "
            f"{target_budget:,}."
        )
        return

    summary = (
        matched.groupby("relations")
        .agg(
            pr_auc_mean=("pr_auc", "mean"),
            pr_auc_std=("pr_auc", "std"),
            training_seconds_mean=(
                "training_seconds",
                "mean",
            ),
            peak_gpu_memory_mb_mean=(
                "peak_gpu_memory_mb",
                "mean",
            ),
        )
        .sort_values(
            "pr_auc_mean",
            ascending=False,
        )
    )

    print(
        f"\nMatched-budget comparison at "
        f"{target_budget:,} stored directed edges"
    )

    print(summary.to_string())


def main() -> None:
    subset_results = load_results(
        SUBSET_DIRECTORY,
        "relation_subsets_seed_*.csv",
    )

    budget_results = load_results(
        BUDGET_DIRECTORY,
        "edge_budgets_seed_*.csv",
    )

    subset_summary = summarize_subset_results(
        subset_results
    )

    budget_summary = summarize_budget_results(
        budget_results
    )

    subset_summary_path = (
        SUBSET_DIRECTORY
        / "relation_subset_summary.csv"
    )

    budget_summary_path = (
        BUDGET_DIRECTORY
        / "edge_budget_summary.csv"
    )

    subset_summary.to_csv(
        subset_summary_path,
        index=False,
    )

    budget_summary.to_csv(
        budget_summary_path,
        index=False,
    )

    print("\nRelation-subset summary")
    print(subset_summary.to_string(index=False))

    print("\nEdge-budget summary")
    print(budget_summary.to_string(index=False))

    print_matched_budget_table(
        budget_results,
        target_budget=98_630,
    )

    plot_relation_subsets(
        subset_results
    )

    plot_subset_cost_quality(
        subset_results
    )

    plot_edge_budget_pr_auc(
        budget_results
    )

    plot_edge_budget_memory(
        budget_results
    )

    plot_edge_budget_time(
        budget_results
    )

    print("\nSaved:", subset_summary_path)
    print("Saved:", budget_summary_path)
    print(
        "Figures saved under "
        "figures/relation_subsets and "
        "figures/edge_budgets."
    )


if __name__ == "__main__":
    main()
