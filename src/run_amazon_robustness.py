from __future__ import annotations

import argparse
import gc
from pathlib import Path

import pandas as pd
import torch

from run_amazon_experiments import (
    AmazonConfiguration,
    build_graph,
    check_dataset,
    edge_dictionary,
    load_amazon,
    train_e2e,
    training_configuration,
)
from run_matched_ablation import set_seed

LEARNING_RATES = (1e-3, 5e-4, 1e-4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--relation", default="net_upu")
    parser.add_argument("--maximum-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--embedding-dimension", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    heterograph, features, labels, train_mask, val_mask, test_mask = load_amazon(
        Path("data/amazon")
    )
    check_dataset(labels, train_mask, val_mask, test_mask)
    edges = edge_dictionary(heterograph)

    if args.relation == "all_relations":
        selected_relations = ("net_upu", "net_usu", "net_uvu")
    elif args.relation in edges:
        selected_relations = (args.relation,)
    else:
        raise ValueError(
            f"Unknown relation {args.relation}. Available: {sorted(edges)} or all_relations"
        )

    device = torch.device("cuda:0")
    graph = build_graph(edges, selected_relations, heterograph.num_nodes()).to(device)
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    rows = []

    for learning_rate in LEARNING_RATES:
        print("\n" + "=" * 72)
        print(
            f"Amazon robustness run: relation={args.relation}, seed={args.seed}, "
            f"learning_rate={learning_rate}, patience={args.patience}, "
            f"maximum_epochs={args.maximum_epochs}"
        )
        print("=" * 72)

        experiment_config = AmazonConfiguration(
            seed=args.seed,
            hidden_dimension=args.hidden_dimension,
            embedding_dimension=args.embedding_dimension,
            learning_rate=learning_rate,
            maximum_epochs=args.maximum_epochs,
            patience=args.patience,
        )
        config = training_configuration(experiment_config)
        set_seed(args.seed)

        result, model = train_e2e(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask,
            config=config,
            device=device,
            head_type="mlp",
        )

        rows.append(
            {
                "dataset": "amazon",
                "relation": args.relation,
                "relations_used": "|".join(selected_relations),
                "seed": args.seed,
                "learning_rate": learning_rate,
                "maximum_epochs": args.maximum_epochs,
                "patience": args.patience,
                "pipeline": "end_to_end_graphsage_mlp",
                "stored_relation_edges": sum(
                    edges[name][0].numel() for name in selected_relations
                ),
                "graph_edges_with_self_loops": graph.num_edges(),
                **result,
            }
        )

        del model
        gc.collect()
        torch.cuda.empty_cache()

    output_directory = Path("results/amazon/robustness")
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"amazon_e2e_robustness_{args.relation}_seed_{args.seed}.csv"
    )

    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)

    print("\nCompleted configurations")
    print(
        frame[
            [
                "learning_rate",
                "roc_auc",
                "pr_auc",
                "f1",
                "training_seconds",
                "best_epoch",
                "peak_gpu_memory_mb",
            ]
        ].to_string(index=False)
    )
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
