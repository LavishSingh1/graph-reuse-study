from __future__ import annotations

import argparse
import gc
import itertools
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import dgl
import pandas as pd
import torch
from dgl.data import FraudAmazonDataset

from run_matched_ablation import (
    Configuration,
    EndToEndGraphSAGE,
    extract_embeddings,
    parameter_count,
    set_seed,
    train_feature_classifier,
    train_model,
    evaluate_model,
)

RELATIONS = ("net_upu", "net_usu", "net_uvu")


@dataclass
class AmazonConfiguration:
    seed: int = 42
    hidden_dimension: int = 64
    embedding_dimension: int = 64
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 300
    patience: int = 30


def training_configuration(config: AmazonConfiguration) -> Configuration:
    return Configuration(
        seed=config.seed,
        hidden_dimension=config.hidden_dimension,
        embedding_dimension=config.embedding_dimension,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        maximum_epochs=config.maximum_epochs,
        patience=config.patience,
        encoder_label_fraction=1.0,
        downstream_label_fraction=1.0,
    )


def load_amazon(raw_dir: Path):
    dataset = FraudAmazonDataset(
        raw_dir=str(raw_dir),
        random_seed=717,
        train_size=0.70,
        val_size=0.10,
        force_reload=False,
        verbose=True,
    )
    graph = dataset[0]
    return (
        graph,
        graph.ndata["feature"].float(),
        graph.ndata["label"].long().view(-1),
        graph.ndata["train_mask"].bool().view(-1),
        graph.ndata["val_mask"].bool().view(-1),
        graph.ndata["test_mask"].bool().view(-1),
    )


def check_dataset(labels, train_mask, val_mask, test_mask) -> None:
    masks = {"train": train_mask, "validation": val_mask, "test": test_mask}
    print("\nSplit and class checks")
    for name, mask in masks.items():
        selected = labels[mask]
        positives = int((selected == 1).sum())
        negatives = int((selected == 0).sum())
        invalid = int(((selected != 0) & (selected != 1)).sum())
        print(
            f"{name}: n={int(mask.sum()):,}, positive={positives:,}, "
            f"negative={negatives:,}, invalid={invalid:,}"
        )
        if invalid:
            raise ValueError(f"{name} mask contains non-binary or unlabeled nodes")
    if bool((train_mask & val_mask).any()) or bool((train_mask & test_mask).any()) or bool((val_mask & test_mask).any()):
        raise ValueError("Train, validation, and test masks overlap")


def edge_dictionary(heterograph) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    edges = {}
    print("\nCanonical Amazon relations")
    for canonical_type in heterograph.canonical_etypes:
        relation = canonical_type[1]
        source, destination = heterograph.edges(etype=canonical_type)
        edges[relation] = (source.cpu(), destination.cpu())
        print(f"{canonical_type}: {source.numel():,} stored directed edges")
    missing = set(RELATIONS) - set(edges)
    if missing:
        raise ValueError(f"Missing expected relations: {sorted(missing)}")
    return edges


def build_graph(edges, selected_relations, number_of_nodes):
    sources = torch.cat([edges[name][0] for name in selected_relations])
    destinations = torch.cat([edges[name][1] for name in selected_relations])
    graph = dgl.graph((sources, destinations), num_nodes=number_of_nodes)
    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)
    return graph


def relation_sets():
    return [(name,) for name in RELATIONS] + [RELATIONS]


def graph_name(selected_relations) -> str:
    return "__".join(selected_relations) if len(selected_relations) > 1 else selected_relations[0]


def train_e2e(graph, features, labels, train_mask, val_mask, test_mask, config, device, head_type="mlp"):
    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=config.hidden_dimension,
        embedding_dimension=config.embedding_dimension,
        dropout=config.dropout,
        head_type=head_type,
    ).to(device)
    count = parameter_count(model)
    model, train_info = train_model(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        train_mask=train_mask,
        validation_mask=val_mask,
        configuration=config,
        device=device,
    )
    metrics = evaluate_model(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        validation_mask=val_mask,
        test_mask=test_mask,
        device=device,
    )
    return {"trainable_parameters": count, **train_info, **metrics}, model


def save_embeddings(path, embeddings, metadata) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"embeddings": embeddings.detach().cpu().contiguous(), "metadata": metadata},
        path,
    )
    return path.stat().st_size / (1024 ** 2)


def scan_relations(edges, number_of_nodes, features, labels, train_mask, val_mask, test_mask, cfg, device):
    rows = []
    for selected in relation_sets():
        name = graph_name(selected)
        graph = build_graph(edges, selected, number_of_nodes).to(device)
        stored_edges = sum(edges[relation][0].numel() for relation in selected)
        print(f"\nRelation scan: {name}; stored edges={stored_edges:,}")
        set_seed(cfg.seed)
        result, model = train_e2e(
            graph, features, labels, train_mask, val_mask, test_mask, cfg, device, "mlp"
        )
        rows.append({
            "experiment": "relation_scan",
            "dataset": "amazon",
            "seed": cfg.seed,
            "graph_name": name,
            "relations": "|".join(selected),
            "pipeline": "end_to_end_graphsage_mlp",
            "stored_relation_edges": stored_edges,
            "graph_edges_with_self_loops": graph.num_edges(),
            **result,
        })
        del model, graph
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def reuse_study(relation_name, edges, number_of_nodes, features, labels, train_mask, val_mask, test_mask, cfg, device):
    if relation_name == "all_relations":
        selected = RELATIONS
    elif relation_name in RELATIONS:
        selected = (relation_name,)
    else:
        raise ValueError(f"Unknown relation {relation_name}; choose from {RELATIONS} or all_relations")

    graph = build_graph(edges, selected, number_of_nodes).to(device)
    stored_edges = sum(edges[name][0].numel() for name in selected)
    rows = []

    def common(pipeline, extra):
        return {
            "experiment": "reuse_study",
            "dataset": "amazon",
            "seed": cfg.seed,
            "graph_name": relation_name,
            "relations": "|".join(selected),
            "pipeline": pipeline,
            "stored_relation_edges": stored_edges,
            "graph_edges_with_self_loops": graph.num_edges(),
            **extra,
        }

    print("\nRaw features + MLP")
    set_seed(cfg.seed)
    raw = train_feature_classifier(
        feature_matrix=features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=val_mask,
        test_mask=test_mask,
        configuration=cfg,
        device=device,
        head_type="mlp",
        pipeline_name="raw_features_mlp",
    )
    rows.append(common("raw_features_mlp", {
        **raw,
        "one_time_encoder_seconds": 0.0,
        "embedding_extraction_seconds": 0.0,
        "embedding_storage_mb": 0.0,
        "first_run_seconds": raw["training_seconds"],
    }))

    print("\nEnd-to-end GraphSAGE + MLP")
    set_seed(cfg.seed)
    e2e, e2e_model = train_e2e(
        graph, features, labels, train_mask, val_mask, test_mask, cfg, device, "mlp"
    )
    rows.append(common("end_to_end_graphsage_mlp", {
        **e2e,
        "one_time_encoder_seconds": 0.0,
        "embedding_extraction_seconds": 0.0,
        "embedding_storage_mb": 0.0,
        "first_run_seconds": e2e["training_seconds"],
    }))
    del e2e_model
    gc.collect()
    torch.cuda.empty_cache()

    print("\nTrain reusable GraphSAGE encoder with linear head")
    set_seed(cfg.seed)
    encoder_info, encoder_model = train_e2e(
        graph, features, labels, train_mask, val_mask, test_mask, cfg, device, "linear"
    )
    embeddings, extraction_info = extract_embeddings(
        encoder=encoder_model.encoder,
        graph=graph,
        features=features,
        device=device,
    )
    embedding_path = Path("embeddings/amazon") / f"{relation_name}_seed_{cfg.seed}.pt"
    storage_mb = save_embeddings(
        embedding_path,
        embeddings,
        {
            "dataset": "amazon",
            "relation": relation_name,
            "seed": cfg.seed,
            "encoder_training_seconds": encoder_info["training_seconds"],
        },
    )
    del encoder_model
    gc.collect()
    torch.cuda.empty_cache()

    cached = torch.load(embedding_path, map_location=device, weights_only=False)["embeddings"].to(device)

    print("\nFrozen GraphSAGE embeddings + MLP")
    set_seed(cfg.seed)
    frozen = train_feature_classifier(
        feature_matrix=cached,
        labels=labels,
        train_mask=train_mask,
        validation_mask=val_mask,
        test_mask=test_mask,
        configuration=cfg,
        device=device,
        head_type="mlp",
        pipeline_name="frozen_graphsage_mlp",
    )
    one_time = encoder_info["training_seconds"]
    extraction = extraction_info["embedding_extraction_seconds"]
    rows.append(common("frozen_graphsage_mlp", {
        **frozen,
        "one_time_encoder_seconds": one_time,
        "embedding_extraction_seconds": extraction,
        "embedding_storage_mb": storage_mb,
        "first_run_seconds": one_time + extraction + frozen["training_seconds"],
    }))

    print("\nRaw features + frozen embeddings + MLP")
    combined = torch.cat([features, cached], dim=1)
    set_seed(cfg.seed)
    combined_result = train_feature_classifier(
        feature_matrix=combined,
        labels=labels,
        train_mask=train_mask,
        validation_mask=val_mask,
        test_mask=test_mask,
        configuration=cfg,
        device=device,
        head_type="mlp",
        pipeline_name="raw_plus_frozen_embedding_mlp",
    )
    rows.append(common("raw_plus_frozen_embedding_mlp", {
        **combined_result,
        "one_time_encoder_seconds": one_time,
        "embedding_extraction_seconds": extraction,
        "embedding_storage_mb": storage_mb,
        "first_run_seconds": one_time + extraction + combined_result["training_seconds"],
    }))

    del graph, embeddings, cached, combined
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def save_rows(rows, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_path, index=False)
    print("\nResults")
    columns = ["graph_name", "pipeline", "roc_auc", "pr_auc", "f1", "training_seconds", "peak_gpu_memory_mb"]
    print(frame[[column for column in columns if column in frame.columns]].to_string(index=False))
    print("Saved:", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["inspect", "scan", "reuse"], required=True)
    parser.add_argument("--reuse-relation", default="all_relations")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maximum-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-dimension", type=int, default=64)
    parser.add_argument("--embedding-dimension", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    config = AmazonConfiguration(
        seed=args.seed,
        hidden_dimension=args.hidden_dimension,
        embedding_dimension=args.embedding_dimension,
        maximum_epochs=args.maximum_epochs,
        patience=args.patience,
    )
    cfg = training_configuration(config)
    set_seed(config.seed)

    heterograph, features, labels, train_mask, val_mask, test_mask = load_amazon(Path("data/amazon"))
    check_dataset(labels, train_mask, val_mask, test_mask)
    edges = edge_dictionary(heterograph)
    print("\nConfiguration")
    print(json.dumps(asdict(config), indent=2))
    print("Nodes:", f"{heterograph.num_nodes():,}")
    print("Features:", tuple(features.shape))
    print("Label values:", torch.unique(labels, return_counts=True))

    if args.mode == "inspect":
        return

    device = torch.device("cuda:0")
    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    val_mask = val_mask.to(device)
    test_mask = test_mask.to(device)

    if args.mode == "scan":
        rows = scan_relations(
            edges, heterograph.num_nodes(), features, labels, train_mask, val_mask, test_mask, cfg, device
        )
        output = Path("results/amazon/relation_scan") / f"amazon_relation_scan_seed_{config.seed}.csv"
    else:
        rows = reuse_study(
            args.reuse_relation, edges, heterograph.num_nodes(), features, labels,
            train_mask, val_mask, test_mask, cfg, device
        )
        output = Path("results/amazon/reuse") / f"amazon_reuse_{args.reuse_relation}_seed_{config.seed}.csv"

    save_rows(rows, output)


if __name__ == "__main__":
    main()
