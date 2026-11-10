from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_ROOT = Path(
    "results/next_experiments"
)

FIGURE_DIRECTORY = Path(
    "figures/next_experiments"
)

FIGURE_DIRECTORY.mkdir(
    parents=True,
    exist_ok=True,
)


REPRESENTATION_LABELS = {
    "raw_features":
        "Raw features",
    "random_linear_projection":
        "Random linear projection",
    "random_nonlinear_mlp":
        "Random nonlinear MLP",
    "random_graphsage":
        "Random GraphSAGE",
    "joint_supervised_graphsage":
        "End-to-end GraphSAGE",
    "frozen_supervised_graphsage":
        "Frozen GraphSAGE",
    "raw_plus_supervised_graphsage":
        "Raw + frozen embedding",
}


def load_csv_directory(
    directory: Path,
) -> pd.DataFrame:
    files = sorted(
        directory.glob(
            "random_relation_seed_*.csv"
        )
    )

    if not files:
        return pd.DataFrame()

    frames = [
        pd.read_csv(file)
        for file in files
    ]

    return pd.concat(
        frames,
        ignore_index=True,
    )


def aggregate(
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
        "one_time_training_seconds",
        "representation_extraction_seconds",
        "representation_storage_mb",
    ]

    available_metrics = [
        metric
        for metric in metrics
        if metric in results.columns
    ]

    summary = (
        results.groupby(
            [
                "experiment_group",
                "relation_name",
                "representation_name",
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


def plot_random_controls(
    random_results: pd.DataFrame,
) -> None:
    if random_results.empty:
        return

    filtered = random_results[
        random_results[
            "experiment_group"
        ] == "random_controls"
    ]

    summary = (
        filtered.groupby(
            "representation_name"
        )["pr_auc"]
        .agg(["mean", "std"])
        .sort_values("mean")
    )

    labels = [
        REPRESENTATION_LABELS.get(
            name,
            name,
        )
        for name in summary.index
    ]

    positions = np.arange(
        len(summary)
    )

    plt.figure(figsize=(9, 5))

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
        "Random representation controls"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "random_representation_controls.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "random_representation_controls.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_relation_ablation(
    relation_results: pd.DataFrame,
) -> None:
    if relation_results.empty:
        return

    filtered = relation_results[
        relation_results[
            "experiment_group"
        ] == "relation_ablation"
    ]

    summary = (
        filtered.groupby(
            [
                "relation_name",
                "representation_name",
            ]
        )["pr_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )

    relations = sorted(
        summary["relation_name"].unique()
    )

    representations = [
        "joint_supervised_graphsage",
        "frozen_supervised_graphsage",
        "raw_plus_supervised_graphsage",
        "random_graphsage",
    ]

    x_positions = np.arange(
        len(relations)
    )

    width = 0.19

    plt.figure(figsize=(12, 6))

    for index, representation in enumerate(
        representations
    ):
        subset = summary[
            summary["representation_name"]
            == representation
        ].set_index(
            "relation_name"
        ).reindex(
            relations
        )

        plt.bar(
            x_positions
            + (
                index
                - (len(representations) - 1) / 2
            ) * width,
            subset["mean"],
            width=width,
            yerr=subset["std"].fillna(0.0),
            capsize=3,
            label=REPRESENTATION_LABELS.get(
                representation,
                representation,
            ),
        )

    plt.xticks(
        x_positions,
        relations,
        rotation=15,
    )

    plt.ylabel("Test PR-AUC")
    plt.xlabel("YelpChi relation")
    plt.title(
        "Predictive utility by graph relation"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "relation_ablation_pr_auc.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "relation_ablation_pr_auc.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def plot_relation_cost_quality(
    relation_results: pd.DataFrame,
) -> None:
    if relation_results.empty:
        return

    filtered = relation_results[
        relation_results[
            "experiment_group"
        ] == "relation_ablation"
    ]

    summary = (
        filtered.groupby(
            [
                "relation_name",
                "representation_name",
            ]
        )
        .agg(
            mean_pr_auc=(
                "pr_auc",
                "mean",
            ),
            mean_first_run_seconds=(
                "first_run_seconds",
                "mean",
            ),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 7))

    for relation in summary[
        "relation_name"
    ].unique():
        subset = summary[
            summary["relation_name"]
            == relation
        ]

        plt.scatter(
            subset[
                "mean_first_run_seconds"
            ],
            subset["mean_pr_auc"],
            s=75,
            label=relation,
        )

        for _, row in subset.iterrows():
            short_name = (
                REPRESENTATION_LABELS.get(
                    row[
                        "representation_name"
                    ],
                    row[
                        "representation_name"
                    ],
                )
            )

            plt.annotate(
                short_name,
                (
                    row[
                        "mean_first_run_seconds"
                    ],
                    row["mean_pr_auc"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

    plt.xlabel(
        "First-run cost (seconds)"
    )

    plt.ylabel("Test PR-AUC")

    plt.title(
        "Relation-specific utility versus cost"
    )

    plt.legend(
        title="Relation"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "relation_cost_quality.pdf",
        bbox_inches="tight",
    )

    plt.savefig(
        FIGURE_DIRECTORY
        / "relation_cost_quality.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    random_results = load_csv_directory(
        RESULT_ROOT
        / "random_controls"
    )

    relation_results = load_csv_directory(
        RESULT_ROOT
        / "relation_ablation"
    )

    print(
        "Random-control rows:",
        len(random_results),
    )

    print(
        "Relation-ablation rows:",
        len(relation_results),
    )

    frames = [
        frame
        for frame in [
            random_results,
            relation_results,
        ]
        if not frame.empty
    ]

    if not frames:
        raise FileNotFoundError(
            "No result files were found."
        )

    all_results = pd.concat(
        frames,
        ignore_index=True,
    )

    summary = aggregate(
        all_results
    )

    summary_file = (
        RESULT_ROOT
        / "random_relation_summary.csv"
    )

    summary.to_csv(
        summary_file,
        index=False,
    )

    print("\nSummary")
    print(
        summary.to_string(
            index=False
        )
    )

    plot_random_controls(
        random_results
    )

    plot_relation_ablation(
        relation_results
    )

    plot_relation_cost_quality(
        relation_results
    )

    print("\nSaved:", summary_file)

    print(
        "Figures saved in",
        FIGURE_DIRECTORY,
    )


if __name__ == "__main__":
    main()
