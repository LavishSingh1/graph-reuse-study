from __future__ import annotations

import argparse
import gc
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import dgl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from dgl.data import FraudYelpDataset

from run_matched_ablation import (
    Configuration,
    EndToEndGraphSAGE,
    GraphSAGEEncoder,
    append_common_fields,
    evaluate_model,
    extract_embeddings,
    parameter_count,
    reset_peak_memory,
    set_seed,
    synchronize,
    train_feature_classifier,
    train_model,
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


class RandomLinearEncoder(nn.Module):
    """
    Fixed random linear projection.

    The encoder is never trained. Only the downstream classifier is trained.
    """

    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        seed: int,
    ) -> None:
        super().__init__()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        weight = torch.randn(
            input_dimension,
            output_dimension,
            generator=generator,
        )

        weight = weight / np.sqrt(input_dimension)

        bias = torch.randn(
            output_dimension,
            generator=generator,
        ) * 0.01

        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return features @ self.weight + self.bias


class RandomMLPEncoder(nn.Module):
    """
    Fixed nonlinear, non-graph random encoder.

    This controls for random projection and nonlinear feature expansion
    without using graph connectivity.
    """

    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        output_dimension: int,
        seed: int,
    ) -> None:
        super().__init__()

        previous_state = torch.random.get_rng_state()
        torch.manual_seed(seed)

        self.layer_1 = nn.Linear(
            input_dimension,
            hidden_dimension,
        )

        self.layer_2 = nn.Linear(
            hidden_dimension,
            output_dimension,
        )

        torch.random.set_rng_state(previous_state)

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = torch.relu(self.layer_1(features))
        return self.layer_2(hidden)


def peak_memory_mb(
    device: torch.device,
) -> float:
    if device.type != "cuda":
        return 0.0

    return torch.cuda.max_memory_allocated(device) / (1024**2)


def load_yelp_heterograph(
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


def build_relation_graphs(
    heterograph: dgl.DGLHeteroGraph,
) -> dict[str, dgl.DGLGraph]:
    """
    Build one homogeneous graph per relation and a union graph.

    All graphs contain the complete YelpChi node set. A node can be isolated
    under a particular relation. One self-loop is added to every node.
    """

    number_of_nodes = heterograph.num_nodes()
    relation_graphs: dict[str, dgl.DGLGraph] = {}

    all_sources = []
    all_destinations = []

    print("\nCanonical relation types")

    for canonical_edge_type in heterograph.canonical_etypes:
        source_type, relation_name, destination_type = (
            canonical_edge_type
        )

        sources, destinations = heterograph.edges(
            etype=canonical_edge_type
        )

        relation_graph = dgl.graph(
            (sources, destinations),
            num_nodes=number_of_nodes,
        )

        relation_graph = dgl.remove_self_loop(
            relation_graph
        )
        relation_graph = dgl.add_self_loop(
            relation_graph
        )

        relation_graphs[relation_name] = relation_graph

        all_sources.append(sources)
        all_destinations.append(destinations)

        print(
            canonical_edge_type,
            "stored edges:",
            heterograph.num_edges(
                canonical_edge_type
            ),
        )

    union_sources = torch.cat(all_sources)
    union_destinations = torch.cat(all_destinations)

    union_graph = dgl.graph(
        (union_sources, union_destinations),
        num_nodes=number_of_nodes,
    )

    union_graph = dgl.remove_self_loop(
        union_graph
    )
    union_graph = dgl.add_self_loop(
        union_graph
    )

    relation_graphs["all_relations"] = union_graph

    return relation_graphs


def representation_extraction(
    encoder: nn.Module,
    forward_function,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    dict[str, float],
]:
    encoder.eval()

    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        representation = forward_function()

    synchronize(device)
    extraction_seconds = time.perf_counter() - start_time

    return representation.detach(), {
        "representation_extraction_seconds":
            extraction_seconds,
        "representation_extraction_peak_gpu_memory_mb":
            peak_memory_mb(device),
    }


def save_representation(
    representation: torch.Tensor,
    output_path: Path,
    metadata: dict,
) -> float:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "representation":
                representation.detach().cpu().contiguous(),
            "metadata": metadata,
        },
        output_path,
    )

    return output_path.stat().st_size / (1024**2)


def make_training_configuration(
    experiment_configuration:
        ExperimentConfiguration,
) -> Configuration:
    return Configuration(
        seed=experiment_configuration.seed,
        hidden_dimension=(
            experiment_configuration.hidden_dimension
        ),
        embedding_dimension=(
            experiment_configuration.embedding_dimension
        ),
        dropout=experiment_configuration.dropout,
        learning_rate=(
            experiment_configuration.learning_rate
        ),
        weight_decay=(
            experiment_configuration.weight_decay
        ),
        maximum_epochs=(
            experiment_configuration.maximum_epochs
        ),
        patience=experiment_configuration.patience,
        encoder_label_fraction=1.0,
        downstream_label_fraction=1.0,
    )


def add_experiment_fields(
    result: dict,
    seed: int,
    experiment_group: str,
    representation_name: str,
    relation_name: str,
    one_time_training_seconds: float = 0.0,
    representation_extraction_seconds: float = 0.0,
    representation_storage_mb: float = 0.0,
) -> dict:
    result["seed"] = seed
    result["experiment_group"] = experiment_group
    result["representation_name"] = representation_name
    result["relation_name"] = relation_name
    result["one_time_training_seconds"] = (
        one_time_training_seconds
    )
    result["representation_extraction_seconds"] = (
        representation_extraction_seconds
    )
    result["representation_storage_mb"] = (
        representation_storage_mb
    )

    result["first_run_seconds"] = (
        one_time_training_seconds
        + representation_extraction_seconds
        + result["training_seconds"]
    )

    return result


def train_supervised_graph_encoder(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    training_configuration: Configuration,
    device: torch.device,
) -> tuple[
    EndToEndGraphSAGE,
    dict,
]:
    """
    The reusable encoder is trained with a linear head.

    This matches the previously used frozen-representation protocol.
    """

    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=(
            training_configuration.hidden_dimension
        ),
        embedding_dimension=(
            training_configuration.embedding_dimension
        ),
        dropout=training_configuration.dropout,
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
        configuration=training_configuration,
        device=device,
    )

    test_information = evaluate_model(
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

    training_information.update(
        {
            "encoder_linear_roc_auc":
                test_information["roc_auc"],
            "encoder_linear_pr_auc":
                test_information["pr_auc"],
            "encoder_linear_f1":
                test_information["f1"],
        }
    )

    return model, training_information


def run_end_to_end_graphsage_mlp(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    training_configuration: Configuration,
    device: torch.device,
    relation_name: str,
) -> dict:
    set_seed(training_configuration.seed)

    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=(
            training_configuration.hidden_dimension
        ),
        embedding_dimension=(
            training_configuration.embedding_dimension
        ),
        dropout=training_configuration.dropout,
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
        configuration=training_configuration,
        device=device,
    )

    test_information = evaluate_model(
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
        "pipeline":
            f"{relation_name}_end_to_end_graphsage_mlp",
        "head_type": "mlp",
        "trainable_parameters":
            number_of_parameters,
        **training_information,
        **test_information,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def run_random_controls(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    graph: dgl.DGLGraph,
    experiment_configuration:
        ExperimentConfiguration,
    training_configuration: Configuration,
    device: torch.device,
) -> list[dict]:
    results = []

    print("\nRandom controls: raw features + MLP")

    set_seed(experiment_configuration.seed)

    raw_result = train_feature_classifier(
        feature_matrix=features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="raw_features_mlp",
    )

    results.append(
        add_experiment_fields(
            result=raw_result,
            seed=experiment_configuration.seed,
            experiment_group="random_controls",
            representation_name="raw_features",
            relation_name="no_graph",
        )
    )

    print("\nRandom controls: random linear projection")

    set_seed(experiment_configuration.seed)

    random_linear_encoder = RandomLinearEncoder(
        input_dimension=features.shape[1],
        output_dimension=(
            experiment_configuration.embedding_dimension
        ),
        seed=experiment_configuration.seed,
    ).to(device)

    random_linear_features, linear_information = (
        representation_extraction(
            encoder=random_linear_encoder,
            forward_function=lambda:
                random_linear_encoder(features),
            device=device,
        )
    )

    random_linear_file = Path(
        "embeddings/next_experiments/"
        f"random_linear_seed_"
        f"{experiment_configuration.seed}.pt"
    )

    random_linear_storage = save_representation(
        representation=random_linear_features,
        output_path=random_linear_file,
        metadata={
            "seed": experiment_configuration.seed,
            "type": "random_linear_projection",
        },
    )

    set_seed(experiment_configuration.seed)

    random_linear_result = train_feature_classifier(
        feature_matrix=random_linear_features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="random_linear_projection_mlp",
    )

    results.append(
        add_experiment_fields(
            result=random_linear_result,
            seed=experiment_configuration.seed,
            experiment_group="random_controls",
            representation_name=
                "random_linear_projection",
            relation_name="no_graph",
            representation_extraction_seconds=(
                linear_information[
                    "representation_extraction_seconds"
                ]
            ),
            representation_storage_mb=(
                random_linear_storage
            ),
        )
    )

    print("\nRandom controls: random nonlinear MLP encoder")

    set_seed(experiment_configuration.seed)

    random_mlp_encoder = RandomMLPEncoder(
        input_dimension=features.shape[1],
        hidden_dimension=(
            experiment_configuration.hidden_dimension
        ),
        output_dimension=(
            experiment_configuration.embedding_dimension
        ),
        seed=experiment_configuration.seed,
    ).to(device)

    random_mlp_features, mlp_information = (
        representation_extraction(
            encoder=random_mlp_encoder,
            forward_function=lambda:
                random_mlp_encoder(features),
            device=device,
        )
    )

    random_mlp_file = Path(
        "embeddings/next_experiments/"
        f"random_mlp_seed_"
        f"{experiment_configuration.seed}.pt"
    )

    random_mlp_storage = save_representation(
        representation=random_mlp_features,
        output_path=random_mlp_file,
        metadata={
            "seed": experiment_configuration.seed,
            "type": "random_nonlinear_mlp",
        },
    )

    set_seed(experiment_configuration.seed)

    random_mlp_result = train_feature_classifier(
        feature_matrix=random_mlp_features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="random_nonlinear_mlp_embedding",
    )

    results.append(
        add_experiment_fields(
            result=random_mlp_result,
            seed=experiment_configuration.seed,
            experiment_group="random_controls",
            representation_name="random_nonlinear_mlp",
            relation_name="no_graph",
            representation_extraction_seconds=(
                mlp_information[
                    "representation_extraction_seconds"
                ]
            ),
            representation_storage_mb=(
                random_mlp_storage
            ),
        )
    )

    print("\nRandom controls: random GraphSAGE encoder")

    set_seed(experiment_configuration.seed)

    random_graph_encoder = GraphSAGEEncoder(
        input_dimension=features.shape[1],
        hidden_dimension=(
            experiment_configuration.hidden_dimension
        ),
        embedding_dimension=(
            experiment_configuration.embedding_dimension
        ),
        dropout=experiment_configuration.dropout,
    ).to(device)

    random_graph_embeddings, graph_information = (
        representation_extraction(
            encoder=random_graph_encoder,
            forward_function=lambda:
                random_graph_encoder(
                    graph,
                    features,
                ),
            device=device,
        )
    )

    random_graph_file = Path(
        "embeddings/next_experiments/"
        f"random_graphsage_all_relations_seed_"
        f"{experiment_configuration.seed}.pt"
    )

    random_graph_storage = save_representation(
        representation=random_graph_embeddings,
        output_path=random_graph_file,
        metadata={
            "seed": experiment_configuration.seed,
            "type": "random_graphsage",
            "relation": "all_relations",
        },
    )

    set_seed(experiment_configuration.seed)

    random_graph_result = train_feature_classifier(
        feature_matrix=random_graph_embeddings,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name="random_graphsage_embedding_mlp",
    )

    results.append(
        add_experiment_fields(
            result=random_graph_result,
            seed=experiment_configuration.seed,
            experiment_group="random_controls",
            representation_name="random_graphsage",
            relation_name="all_relations",
            representation_extraction_seconds=(
                graph_information[
                    "representation_extraction_seconds"
                ]
            ),
            representation_storage_mb=(
                random_graph_storage
            ),
        )
    )

    del random_linear_encoder
    del random_mlp_encoder
    del random_graph_encoder
    del random_linear_features
    del random_mlp_features
    del random_graph_embeddings

    gc.collect()
    torch.cuda.empty_cache()

    return results


def run_relation_ablation(
    relation_name: str,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    experiment_configuration:
        ExperimentConfiguration,
    training_configuration: Configuration,
    device: torch.device,
) -> list[dict]:
    print("\n" + "=" * 72)
    print("Relation_name")
    print("Nodes:", graph.num_nodes())
    print("Edges including self-loops:", graph.num_edges())
    print("=" * 72)

    results = []

    print(
        f"\n{relation_name}: "
        "end-to-end GraphSAGE + MLP"
    )

    end_to_end_result = run_end_to_end_graphsage_mlp(
        graph=graph,
        features=features,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        training_configuration=training_configuration,
        device=device,
        relation_name=relation_name,
    )

    results.append(
        add_experiment_fields(
            result=end_to_end_result,
            seed=experiment_configuration.seed,
            experiment_group="relation_ablation",
            representation_name=
                "joint_supervised_graphsage",
            relation_name=relation_name,
        )
    )

    print(
        f"\n{relation_name}: "
        "training reusable supervised encoder"
    )

    set_seed(experiment_configuration.seed)

    supervised_model, encoder_information = (
        train_supervised_graph_encoder(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            training_configuration=
                training_configuration,
            device=device,
        )
    )

    embeddings, extraction_information = (
        extract_embeddings(
            encoder=supervised_model.encoder,
            graph=graph,
            features=features,
            device=device,
        )
    )

    embedding_file = Path(
        "embeddings/next_experiments/"
        f"supervised_{relation_name}_seed_"
        f"{experiment_configuration.seed}.pt"
    )

    embedding_storage_mb = save_representation(
        representation=embeddings,
        output_path=embedding_file,
        metadata={
            "seed": experiment_configuration.seed,
            "relation": relation_name,
            "encoder_training_seconds":
                encoder_information[
                    "training_seconds"
                ],
        },
    )

    del supervised_model
    gc.collect()
    torch.cuda.empty_cache()

    cached = torch.load(
        embedding_file,
        map_location=device,
        weights_only=False,
    )

    cached_embeddings = cached[
        "representation"
    ].to(device)

    print(
        f"\n{relation_name}: "
        "frozen supervised embedding + MLP"
    )

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
        pipeline_name=(
            f"{relation_name}_"
            "frozen_supervised_graphsage_mlp"
        ),
    )

    results.append(
        add_experiment_fields(
            result=frozen_result,
            seed=experiment_configuration.seed,
            experiment_group="relation_ablation",
            representation_name=
                "frozen_supervised_graphsage",
            relation_name=relation_name,
            one_time_training_seconds=(
                encoder_information[
                    "training_seconds"
                ]
            ),
            representation_extraction_seconds=(
                extraction_information[
                    "embedding_extraction_seconds"
                ]
            ),
            representation_storage_mb=(
                embedding_storage_mb
            ),
        )
    )

    print(
        f"\n{relation_name}: "
        "raw features + frozen embedding + MLP"
    )

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
        pipeline_name=(
            f"{relation_name}_"
            "raw_plus_embedding_mlp"
        ),
    )

    results.append(
        add_experiment_fields(
            result=combined_result,
            seed=experiment_configuration.seed,
            experiment_group="relation_ablation",
            representation_name=
                "raw_plus_supervised_graphsage",
            relation_name=relation_name,
            one_time_training_seconds=(
                encoder_information[
                    "training_seconds"
                ]
            ),
            representation_extraction_seconds=(
                extraction_information[
                    "embedding_extraction_seconds"
                ]
            ),
            representation_storage_mb=(
                embedding_storage_mb
            ),
        )
    )

    print(
        f"\n{relation_name}: "
        "random GraphSAGE embedding + MLP"
    )

    set_seed(experiment_configuration.seed)

    random_encoder = GraphSAGEEncoder(
        input_dimension=features.shape[1],
        hidden_dimension=(
            experiment_configuration.hidden_dimension
        ),
        embedding_dimension=(
            experiment_configuration.embedding_dimension
        ),
        dropout=experiment_configuration.dropout,
    ).to(device)

    random_embeddings, random_extraction = (
        representation_extraction(
            encoder=random_encoder,
            forward_function=lambda:
                random_encoder(
                    graph,
                    features,
                ),
            device=device,
        )
    )

    set_seed(experiment_configuration.seed)

    random_result = train_feature_classifier(
        feature_matrix=random_embeddings,
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=training_configuration,
        device=device,
        head_type="mlp",
        pipeline_name=(
            f"{relation_name}_"
            "random_graphsage_embedding_mlp"
        ),
    )

    results.append(
        add_experiment_fields(
            result=random_result,
            seed=experiment_configuration.seed,
            experiment_group="relation_ablation",
            representation_name="random_graphsage",
            relation_name=relation_name,
            representation_extraction_seconds=(
                random_extraction[
                    "representation_extraction_seconds"
                ]
            ),
        )
    )

    del embeddings
    del cached_embeddings
    del combined_features
    del random_embeddings
    del random_encoder

    gc.collect()
    torch.cuda.empty_cache()

    return results


def main() -> None:
    parser = argparse.ArgumentParser()

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

    parser.add_argument(
        "--experiment",
        choices=[
            "all",
            "random_controls",
            "relation_ablation",
        ],
        default="all",
    )

    arguments = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable."
        )

    experiment_configuration = (
        ExperimentConfiguration(
            seed=arguments.seed,
            hidden_dimension=(
                arguments.hidden_dimension
            ),
            embedding_dimension=(
                arguments.embedding_dimension
            ),
            maximum_epochs=(
                arguments.maximum_epochs
            ),
            patience=arguments.patience,
        )
    )

    training_configuration = (
        make_training_configuration(
            experiment_configuration
        )
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
    ) = load_yelp_heterograph(
        Path("data")
    )

    relation_graphs = build_relation_graphs(
        heterograph
    )

    features = features.to(device)
    labels = labels.to(device)
    train_mask = train_mask.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    results = []

    print("\nExperiment configuration")
    print(
        json.dumps(
            asdict(experiment_configuration),
            indent=2,
        )
    )

    if arguments.experiment in [
        "all",
        "random_controls",
    ]:
        all_relations_graph = (
            relation_graphs[
                "all_relations"
            ].to(device)
        )

        random_results = run_random_controls(
            features=features,
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            graph=all_relations_graph,
            experiment_configuration=(
                experiment_configuration
            ),
            training_configuration=(
                training_configuration
            ),
            device=device,
        )

        results.extend(random_results)

        del all_relations_graph
        torch.cuda.empty_cache()

    if arguments.experiment in [
        "all",
        "relation_ablation",
    ]:
        for relation_name, cpu_graph in (
            relation_graphs.items()
        ):
            graph = cpu_graph.to(device)

            relation_results = (
                run_relation_ablation(
                    relation_name=relation_name,
                    graph=graph,
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
            )

            results.extend(relation_results)

            del graph
            gc.collect()
            torch.cuda.empty_cache()

    result_frame = pd.DataFrame(results)

    output_directory = Path(
        "results/next_experiments"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_directory / (
        f"random_relation_seed_"
        f"{experiment_configuration.seed}.csv"
    )

    result_frame.to_csv(
        output_file,
        index=False,
    )

    print("\nFinal results")

    print(
        result_frame[
            [
                "experiment_group",
                "relation_name",
                "representation_name",
                "pipeline",
                "roc_auc",
                "pr_auc",
                "f1",
                "training_seconds",
                "first_run_seconds",
                "peak_gpu_memory_mb",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
