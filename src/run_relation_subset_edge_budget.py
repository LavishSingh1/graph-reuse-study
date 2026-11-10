from __future__ import annotations

import argparse
import gc
import itertools
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import dgl
import numpy as np
import pandas as pd
import torch

from dgl.data import FraudYelpDataset

from run_matched_ablation import (
    Configuration,
    EndToEndGraphSAGE,
    append_common_fields,
    evaluate_model,
    extract_embeddings,
    parameter_count,
    set_seed,
    train_feature_classifier,
    train_model,
)


RELATION_NAMES = (
    "net_rsr",
    "net_rtr",
    "net_rur",
)


@dataclass
class ExperimentConfiguration:
    seed: int = 42
    hidden_dimension: int = 64
    embedding_dimension: int = 64
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 300
    patience: int = 30


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_memory_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0

    return torch.cuda.max_memory_allocated(device) / (1024**2)


def make_training_configuration(
    configuration: ExperimentConfiguration,
) -> Configuration:
    return Configuration(
        seed=configuration.seed,
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
        learning_rate=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
        maximum_epochs=configuration.maximum_epochs,
        patience=configuration.patience,
        encoder_label_fraction=1.0,
        downstream_label_fraction=1.0,
    )


def load_yelp_data(
    data_directory: Path,
):
    dataset = FraudYelpDataset(
        raw_dir=str(data_directory),
        random_seed=717,
        train_size=0.70,
        val_size=0.10,
        force_reload=False,
        verbose=True,
    )

    heterograph = dataset[0]

    features = heterograph.ndata["feature"].float()
    labels = heterograph.ndata["label"].long().view(-1)
    train_mask = heterograph.ndata["train_mask"].bool().view(-1)
    validation_mask = heterograph.ndata["val_mask"].bool().view(-1)
    test_mask = heterograph.ndata["test_mask"].bool().view(-1)

    return (
        heterograph,
        features,
        labels,
        train_mask,
        validation_mask,
        test_mask,
    )


def extract_relation_edges(
    heterograph: dgl.DGLHeteroGraph,
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    relation_edges = {}

    print("\nAvailable YelpChi relations")

    for canonical_edge_type in heterograph.canonical_etypes:
        relation_name = canonical_edge_type[1]

        sources, destinations = heterograph.edges(
            etype=canonical_edge_type,
        )

        relation_edges[relation_name] = (
            sources.detach().cpu(),
            destinations.detach().cpu(),
        )

        print(
            f"{relation_name}: "
            f"{sources.numel():,} stored directed edges"
        )

    missing = set(RELATION_NAMES).difference(
        relation_edges.keys()
    )

    if missing:
        raise ValueError(
            f"Expected relations are missing: {sorted(missing)}"
        )

    return relation_edges


def relation_subset_configurations() -> list[tuple[str, ...]]:
    subsets = []

    for subset_size in range(1, len(RELATION_NAMES) + 1):
        subsets.extend(
            itertools.combinations(
                RELATION_NAMES,
                subset_size,
            )
        )

    return subsets


def subset_name(
    relations: Iterable[str],
) -> str:
    return "__".join(sorted(relations))


def make_graph_from_edges(
    sources: torch.Tensor,
    destinations: torch.Tensor,
    number_of_nodes: int,
) -> dgl.DGLGraph:
    graph = dgl.graph(
        (sources, destinations),
        num_nodes=number_of_nodes,
    )

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    return graph


def build_subset_graph(
    relation_edges: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ],
    relations: tuple[str, ...],
    number_of_nodes: int,
) -> dgl.DGLGraph:
    source_parts = []
    destination_parts = []

    for relation_name in relations:
        sources, destinations = relation_edges[relation_name]

        source_parts.append(sources)
        destination_parts.append(destinations)

    combined_sources = torch.cat(source_parts)
    combined_destinations = torch.cat(destination_parts)

    return make_graph_from_edges(
        sources=combined_sources,
        destinations=combined_destinations,
        number_of_nodes=number_of_nodes,
    )


def sample_directed_edges(
    sources: torch.Tensor,
    destinations: torch.Tensor,
    requested_budget: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Reproducibly sample stored directed edges.

    The YelpChi relations are already stored as directed/bidirectional edge
    lists by DGL. This function samples stored edges directly. The paper must
    therefore describe the edge budget as a stored directed-edge budget.
    """

    total_edges = sources.numel()

    if requested_budget >= total_edges:
        return sources.clone(), destinations.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    selected_indices = torch.randperm(
        total_edges,
        generator=generator,
    )[:requested_budget]

    return (
        sources[selected_indices],
        destinations[selected_indices],
    )


def representation_extraction(
    forward_function,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        representation = forward_function()

    synchronize(device)

    elapsed_seconds = time.perf_counter() - start_time

    return representation.detach(), {
        "representation_extraction_seconds":
            elapsed_seconds,
        "representation_extraction_peak_gpu_memory_mb":
            peak_gpu_memory_mb(device),
    }


def save_embedding(
    embeddings: torch.Tensor,
    output_path: Path,
    metadata: dict,
) -> float:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "embeddings":
                embeddings.detach().cpu().contiguous(),
            "metadata": metadata,
        },
        output_path,
    )

    return output_path.stat().st_size / (1024**2)


def train_end_to_end_graphsage(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> tuple[dict, EndToEndGraphSAGE]:
    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
        head_type="mlp",
    ).to(device)

    number_of_parameters = parameter_count(model)

    model, training_information = train_model(
        model=model,
        forward_function=lambda: model(
            graph,
            features,
        ),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    evaluation_information = evaluate_model(
        model=model,
        forward_function=lambda: model(
            graph,
            features,
        ),
        labels=labels,
        validation_mask=validation_mask,
        test_mask=test_mask,
        device=device,
    )

    result = {
        "pipeline": "end_to_end_graphsage_mlp",
        "head_type": "mlp",
        "trainable_parameters": number_of_parameters,
        **training_information,
        **evaluation_information,
    }

    return result, model


def train_reusable_linear_encoder(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> tuple[
    EndToEndGraphSAGE,
    dict[str, float],
]:
    """
    Train the reusable GraphSAGE encoder with a linear classifier head.

    This preserves consistency with the previously used frozen-embedding
    protocol.
    """

    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
        head_type="linear",
    ).to(device)

    model, training_information = train_model(
        model=model,
        forward_function=lambda: model(
            graph,
            features,
        ),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    evaluation_information = evaluate_model(
        model=model,
        forward_function=lambda: model(
            graph,
            features,
        ),
        labels=labels,
        validation_mask=validation_mask,
        test_mask=test_mask,
        device=device,
    )

    training_information[
        "linear_encoder_test_roc_auc"
    ] = evaluation_information["roc_auc"]

    training_information[
        "linear_encoder_test_pr_auc"
    ] = evaluation_information["pr_auc"]

    training_information[
        "linear_encoder_test_f1"
    ] = evaluation_information["f1"]

    return model, training_information


def add_common_fields(
    result: dict,
    *,
    experiment_name: str,
    graph_name: str,
    relations: str,
    seed: int,
    edge_budget: int,
    graph_edges_before_self_loops: int,
    graph_edges_after_self_loops: int,
    one_time_encoder_seconds: float = 0.0,
    extraction_seconds: float = 0.0,
    embedding_storage_mb: float = 0.0,
) -> dict:
    result["experiment_name"] = experiment_name
    result["graph_name"] = graph_name
    result["relations"] = relations
    result["seed"] = seed
    result["edge_budget"] = edge_budget

    result["graph_edges_before_self_loops"] = (
        graph_edges_before_self_loops
    )

    result["graph_edges_after_self_loops"] = (
        graph_edges_after_self_loops
    )

    result["one_time_encoder_seconds"] = (
        one_time_encoder_seconds
    )

    result["representation_extraction_seconds"] = (
        extraction_seconds
    )

    result["embedding_storage_mb"] = (
        embedding_storage_mb
    )

    result["first_run_seconds"] = (
        one_time_encoder_seconds
        + extraction_seconds
        + result["training_seconds"]
    )

    return result


def run_relation_subset(
    relation_edges: dict[
        str,
        tuple[torch.Tensor, torch.Tensor],
    ],
    relations: tuple[str, ...],
    number_of_nodes: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    experiment_configuration: ExperimentConfiguration,
    training_configuration: Configuration,
    device: torch.device,
) -> list[dict]:
    graph_configuration_name = subset_name(relations)

    graph_edges_before_self_loops = sum(
        relation_edges[relation_name][0].numel()
        for relation_name in relations
    )

    cpu_graph = build_subset_graph(
        relation_edges=relation_edges,
        relations=relations,
        number_of_nodes=number_of_nodes,
    )

    graph_edges_after_self_loops = cpu_graph.num_edges()
    graph = cpu_graph.to(device)

    print("\n" + "=" * 78)
    print("Relation subset:", graph_configuration_name)
    print(
        "Edges before self-loops:",
        f"{graph_edges_before_self_loops:,}",
    )
    print(
        "Edges after self-loops:",
        f"{graph_edges_after_self_loops:,}",
    )
    print("=" * 78)

    results = []

    # ---------------------------------------------------------
    # End-to-end GraphSAGE + MLP
    # ---------------------------------------------------------
    print("\nEnd-to-end GraphSAGE + MLP")

    set_seed(experiment_configuration.seed)

    end_to_end_result, end_to_end_model = (
        train_end_to_end_graphsage(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=training_configuration,
            device=device,
        )
    )

    results.append(
        add_common_fields(
            result=end_to_end_result,
            experiment_name="relation_subsets",
            graph_name=graph_configuration_name,
            relations="|".join(relations),
            seed=experiment_configuration.seed,
            edge_budget=graph_edges_before_self_loops,
            graph_edges_before_self_loops=(
                graph_edges_before_self_loops
            ),
            graph_edges_after_self_loops=(
                graph_edges_after_self_loops
            ),
        )
    )

    del end_to_end_model
    gc.collect()
    torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Reusable encoder trained with linear head
    # ---------------------------------------------------------
    print("\nTraining reusable GraphSAGE encoder")

    set_seed(experiment_configuration.seed)

    reusable_model, encoder_information = (
        train_reusable_linear_encoder(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=training_configuration,
            device=device,
        )
    )

    embeddings, extraction_information = (
        extract_embeddings(
            encoder=reusable_model.encoder,
            graph=graph,
            features=features,
            device=device,
        )
    )

    embedding_path = Path(
        "embeddings/relation_subsets"
    ) / (
        f"{graph_configuration_name}_"
        f"seed_{experiment_configuration.seed}.pt"
    )

    embedding_storage_mb = save_embedding(
        embeddings=embeddings,
        output_path=embedding_path,
        metadata={
            "seed": experiment_configuration.seed,
            "graph_name": graph_configuration_name,
            "relations": list(relations),
            "graph_edges_before_self_loops":
                graph_edges_before_self_loops,
            "encoder_training_seconds":
                encoder_information["training_seconds"],
        },
    )

    del reusable_model
    gc.collect()
    torch.cuda.empty_cache()

    cached_object = torch.load(
        embedding_path,
        map_location=device,
        weights_only=False,
    )

    cached_embeddings = cached_object[
        "embeddings"
    ].to(device)

    # ---------------------------------------------------------
    # Frozen embeddings + MLP
    # ---------------------------------------------------------
    print("\nFrozen GraphSAGE embedding + MLP")

    set_seed(experiment_configuration.seed)

    frozen_result = train_feature_classifier(
        feature_matrix=cached_embeddings,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="frozen_graphsage_mlp",
    )

    results.append(
        add_common_fields(
            result=frozen_result,
            experiment_name="relation_subsets",
            graph_name=graph_configuration_name,
            relations="|".join(relations),
            seed=experiment_configuration.seed,
            edge_budget=graph_edges_before_self_loops,
            graph_edges_before_self_loops=(
                graph_edges_before_self_loops
            ),
            graph_edges_after_self_loops=(
                graph_edges_after_self_loops
            ),
            one_time_encoder_seconds=(
                encoder_information["training_seconds"]
            ),
            extraction_seconds=(
                extraction_information[
                    "embedding_extraction_seconds"
                ]
            ),
            embedding_storage_mb=embedding_storage_mb,
        )
    )

    # ---------------------------------------------------------
    # Raw features + frozen embeddings + MLP
    # ---------------------------------------------------------
    print("\nRaw features + frozen embedding + MLP")

    combined_features = torch.cat(
        [features, cached_embeddings],
        dim=1,
    )

    set_seed(experiment_configuration.seed)

    combined_result = train_feature_classifier(
        feature_matrix=combined_features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="raw_plus_frozen_embedding_mlp",
    )

    results.append(
        add_common_fields(
            result=combined_result,
            experiment_name="relation_subsets",
            graph_name=graph_configuration_name,
            relations="|".join(relations),
            seed=experiment_configuration.seed,
            edge_budget=graph_edges_before_self_loops,
            graph_edges_before_self_loops=(
                graph_edges_before_self_loops
            ),
            graph_edges_after_self_loops=(
                graph_edges_after_self_loops
            ),
            one_time_encoder_seconds=(
                encoder_information["training_seconds"]
            ),
            extraction_seconds=(
                extraction_information[
                    "embedding_extraction_seconds"
                ]
            ),
            embedding_storage_mb=embedding_storage_mb,
        )
    )

    del embeddings
    del cached_embeddings
    del combined_features
    del graph
    del cpu_graph

    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_edge_budget_condition(
    relation_name: str,
    sources: torch.Tensor,
    destinations: torch.Tensor,
    requested_budget: int,
    number_of_nodes: int,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    experiment_configuration: ExperimentConfiguration,
    training_configuration: Configuration,
    device: torch.device,
) -> dict:
    sampling_seed = (
        experiment_configuration.seed
        + requested_budget
        + sum(ord(character) for character in relation_name)
    )

    sampled_sources, sampled_destinations = (
        sample_directed_edges(
            sources=sources,
            destinations=destinations,
            requested_budget=requested_budget,
            seed=sampling_seed,
        )
    )

    actual_budget = sampled_sources.numel()

    cpu_graph = make_graph_from_edges(
        sources=sampled_sources,
        destinations=sampled_destinations,
        number_of_nodes=number_of_nodes,
    )

    graph_edges_after_self_loops = cpu_graph.num_edges()
    graph = cpu_graph.to(device)

    graph_name = (
        f"{relation_name}__budget_{actual_budget}"
    )

    print("\n" + "-" * 78)
    print("Relation:", relation_name)
    print("Requested budget:", f"{requested_budget:,}")
    print("Actual stored edges:", f"{actual_budget:,}")
    print(
        "Edges after self-loops:",
        f"{graph_edges_after_self_loops:,}",
    )
    print("-" * 78)

    set_seed(experiment_configuration.seed)

    result, model = train_end_to_end_graphsage(
        graph=graph,
        features=features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
    )

    result = add_common_fields(
        result=result,
        experiment_name="equal_edge_budget",
        graph_name=graph_name,
        relations=relation_name,
        seed=experiment_configuration.seed,
        edge_budget=actual_budget,
        graph_edges_before_self_loops=actual_budget,
        graph_edges_after_self_loops=(
            graph_edges_after_self_loops
        ),
    )

    result["requested_edge_budget"] = requested_budget
    result["available_relation_edges"] = sources.numel()
    result["sampling_seed"] = sampling_seed

    del model
    del graph
    del cpu_graph

    gc.collect()
    torch.cuda.empty_cache()

    return result


def edge_budgets_for_relation(
    number_of_edges: int,
) -> list[int]:
    requested_budgets = [
        25_000,
        50_000,
        98_630,
        250_000,
        500_000,
    ]

    supported_budgets = [
        budget
        for budget in requested_budgets
        if budget < number_of_edges
    ]

    supported_budgets.append(number_of_edges)

    return sorted(set(supported_budgets))


def run_relation_subset_experiment(
    relation_edges,
    number_of_nodes,
    features,
    labels,
    train_mask,
    validation_mask,
    test_mask,
    experiment_configuration,
    training_configuration,
    device,
) -> list[dict]:
    results = []

    for relations in relation_subset_configurations():
        subset_results = run_relation_subset(
            relation_edges=relation_edges,
            relations=relations,
            number_of_nodes=number_of_nodes,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            experiment_configuration=(
                experiment_configuration
            ),
            training_configuration=(
                training_configuration
            ),
            device=device,
        )

        results.extend(subset_results)

    return results


def run_edge_budget_experiment(
    relation_edges,
    number_of_nodes,
    features,
    labels,
    train_mask,
    validation_mask,
    test_mask,
    experiment_configuration,
    training_configuration,
    device,
) -> list[dict]:
    results = []

    for relation_name in RELATION_NAMES:
        sources, destinations = relation_edges[
            relation_name
        ]

        budgets = edge_budgets_for_relation(
            sources.numel()
        )

        for budget in budgets:
            result = run_edge_budget_condition(
                relation_name=relation_name,
                sources=sources,
                destinations=destinations,
                requested_budget=budget,
                number_of_nodes=number_of_nodes,
                features=features,
                labels=labels,
                train_mask=train_mask,
                validation_mask=validation_mask,
                test_mask=test_mask,
                experiment_configuration=(
                    experiment_configuration
                ),
                training_configuration=(
                    training_configuration
                ),
                device=device,
            )

            results.append(result)

    return results


def save_results(
    results: list[dict],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(output_path, index=False)

    display_columns = [
        "experiment_name",
        "graph_name",
        "pipeline",
        "roc_auc",
        "pr_auc",
        "f1",
        "training_seconds",
        "first_run_seconds",
        "peak_gpu_memory_mb",
    ]

    existing_display_columns = [
        column
        for column in display_columns
        if column in result_frame.columns
    ]

    print("\nFinal results")

    print(
        result_frame[
            existing_display_columns
        ].to_string(index=False)
    )

    print("\nSaved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            "relation_subsets",
            "edge_budgets",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--maximum-epochs",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--hidden-dimension",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--embedding-dimension",
        type=int,
        default=64,
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the PyTorch installation."
        )

    experiment_configuration = ExperimentConfiguration(
        seed=arguments.seed,
        hidden_dimension=arguments.hidden_dimension,
        embedding_dimension=arguments.embedding_dimension,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
    )

    training_configuration = make_training_configuration(
        experiment_configuration
    )

    set_seed(experiment_configuration.seed)

    device = torch.device("cuda:0")

    (
        heterograph,
        features,
        labels,
        train_mask,
        validation_mask,
        test_mask,
    ) = load_yelp_data(Path("data"))

    relation_edges = extract_relation_edges(
        heterograph
    )

    number_of_nodes = heterograph.num_nodes()

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    print("\nExperiment configuration")

    print(
        json.dumps(
            asdict(experiment_configuration),
            indent=2,
        )
    )

    print("Nodes:", f"{number_of_nodes:,}")

    if arguments.experiment in (
        "relation_subsets",
        "all",
    ):
        subset_results = run_relation_subset_experiment(
            relation_edges=relation_edges,
            number_of_nodes=number_of_nodes,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            experiment_configuration=(
                experiment_configuration
            ),
            training_configuration=(
                training_configuration
            ),
            device=device,
        )

        subset_output_path = Path(
            "results/relation_subsets"
        ) / (
            f"relation_subsets_seed_"
            f"{experiment_configuration.seed}.csv"
        )

        save_results(
            subset_results,
            subset_output_path,
        )

    if arguments.experiment in (
        "edge_budgets",
        "all",
    ):
        budget_results = run_edge_budget_experiment(
            relation_edges=relation_edges,
            number_of_nodes=number_of_nodes,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            experiment_configuration=(
                experiment_configuration
            ),
            training_configuration=(
                training_configuration
            ),
            device=device,
        )

        budget_output_path = Path(
            "results/edge_budgets"
        ) / (
            f"edge_budgets_seed_"
            f"{experiment_configuration.seed}.csv"
        )

        save_results(
            budget_results,
            budget_output_path,
        )


if __name__ == "__main__":
    main()
