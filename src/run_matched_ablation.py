from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import dgl
import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.data import FraudYelpDataset
from dgl.nn import SAGEConv
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


@dataclass
class Configuration:
    seed: int = 42
    hidden_dimension: int = 64
    embedding_dimension: int = 64
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 300
    patience: int = 30
    encoder_label_fraction: float = 1.0
    downstream_label_fraction: float = 1.0


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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


def process_memory_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / (1024**2)


def parameter_count(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def load_yelp_graph(data_directory: Path):
    dataset = FraudYelpDataset(
        raw_dir=str(data_directory),
        random_seed=717,
        train_size=0.70,
        val_size=0.10,
        force_reload=False,
        verbose=True,
    )

    heterograph = dataset[0]

    graph = dgl.to_homogeneous(
        heterograph,
        ndata=[
            "feature",
            "label",
            "train_mask",
            "val_mask",
            "test_mask",
        ],
        store_type=True,
    )

    graph = dgl.remove_self_loop(graph)
    graph = dgl.add_self_loop(graph)

    features = graph.ndata["feature"].float()
    labels = graph.ndata["label"].long().view(-1)
    train_mask = graph.ndata["train_mask"].bool().view(-1)
    validation_mask = graph.ndata["val_mask"].bool().view(-1)
    test_mask = graph.ndata["test_mask"].bool().view(-1)

    return (
        graph,
        features,
        labels,
        train_mask,
        validation_mask,
        test_mask,
    )


def stratified_fraction_mask(
    labels: torch.Tensor,
    original_mask: torch.Tensor,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    if fraction >= 1.0:
        return original_mask.clone()

    labels_cpu = labels.detach().cpu()
    mask_cpu = original_mask.detach().cpu()

    selected_mask = torch.zeros_like(
        mask_cpu,
        dtype=torch.bool,
    )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for class_value in torch.unique(labels_cpu[mask_cpu]):
        candidate_indices = torch.where(
            mask_cpu & (labels_cpu == class_value)
        )[0]

        number_to_select = max(
            1,
            int(round(candidate_indices.numel() * fraction)),
        )

        permutation = torch.randperm(
            candidate_indices.numel(),
            generator=generator,
        )

        selected_indices = candidate_indices[
            permutation[:number_to_select]
        ]

        selected_mask[selected_indices] = True

    return selected_mask.to(original_mask.device)


def positive_class_weight(
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    selected_labels = labels[mask]

    positive_count = (selected_labels == 1).sum().float()
    negative_count = (selected_labels == 0).sum().float()

    return negative_count / positive_count.clamp_min(1.0)


def select_validation_threshold(
    labels: torch.Tensor,
    logits: torch.Tensor,
    validation_mask: torch.Tensor,
) -> float:
    validation_labels = (
        labels[validation_mask]
        .detach()
        .cpu()
        .numpy()
    )

    validation_probabilities = (
        torch.sigmoid(logits[validation_mask])
        .detach()
        .cpu()
        .numpy()
    )

    precision, recall, thresholds = precision_recall_curve(
        validation_labels,
        validation_probabilities,
    )

    if thresholds.size == 0:
        return 0.5

    f1_values = (
        2.0
        * precision[:-1]
        * recall[:-1]
        / np.maximum(
            precision[:-1] + recall[:-1],
            1e-12,
        )
    )

    best_index = int(np.nanargmax(f1_values))
    return float(thresholds[best_index])


def calculate_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
    threshold: float,
) -> dict[str, float]:
    selected_labels = (
        labels[mask]
        .detach()
        .cpu()
        .numpy()
    )

    probabilities = (
        torch.sigmoid(logits[mask])
        .detach()
        .cpu()
        .numpy()
    )

    predictions = (probabilities >= threshold).astype(np.int64)

    return {
        "roc_auc": float(
            roc_auc_score(
                selected_labels,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                selected_labels,
                probabilities,
            )
        ),
        "f1": float(
            f1_score(
                selected_labels,
                predictions,
                zero_division=0,
            )
        ),
        "decision_threshold": threshold,
    }


class LinearHead(nn.Module):
    def __init__(self, input_dimension: int) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dimension, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layer(features).view(-1)


class MLPHead(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).view(-1)


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        embedding_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.layer_1 = SAGEConv(
            input_dimension,
            hidden_dimension,
            aggregator_type="mean",
        )

        self.layer_2 = SAGEConv(
            hidden_dimension,
            embedding_dimension,
            aggregator_type="mean",
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.layer_1(graph, features)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        embeddings = self.layer_2(graph, hidden)
        return embeddings


class EndToEndGraphSAGE(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        embedding_dimension: int,
        dropout: float,
        head_type: str,
    ) -> None:
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            embedding_dimension=embedding_dimension,
            dropout=dropout,
        )

        if head_type == "linear":
            self.head = LinearHead(embedding_dimension)
        elif head_type == "mlp":
            self.head = MLPHead(
                input_dimension=embedding_dimension,
                hidden_dimension=hidden_dimension,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown head type: {head_type}")

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encoder(graph, features)
        return self.head(embeddings)


def train_model(
    model: nn.Module,
    forward_function,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> tuple[
    nn.Module,
    dict[str, float],
]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=positive_class_weight(
            labels,
            train_mask,
        ).to(device)
    )

    best_state = None
    best_epoch = 0
    best_validation_pr_auc = float("-inf")
    epochs_without_improvement = 0

    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    for epoch in range(
        1,
        configuration.maximum_epochs + 1,
    ):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = forward_function()

        loss = loss_function(
            logits[train_mask],
            labels[train_mask].float(),
        )

        loss.backward()
        optimizer.step()

        model.eval()

        with torch.no_grad():
            validation_logits = forward_function()

        validation_probabilities = (
            torch.sigmoid(
                validation_logits[validation_mask]
            )
            .detach()
            .cpu()
            .numpy()
        )

        validation_labels = (
            labels[validation_mask]
            .detach()
            .cpu()
            .numpy()
        )

        validation_pr_auc = average_precision_score(
            validation_labels,
            validation_probabilities,
        )

        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = float(
                validation_pr_auc
            )
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter
                in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 25 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.5f} | "
                f"validation PR-AUC="
                f"{validation_pr_auc:.5f}"
            )

        if epochs_without_improvement >= configuration.patience:
            break

    synchronize(device)
    training_seconds = time.perf_counter() - start_time

    if best_state is None:
        raise RuntimeError("No valid checkpoint was created.")

    model.load_state_dict(best_state)

    return model, {
        "training_seconds": training_seconds,
        "peak_gpu_memory_mb": peak_gpu_memory_mb(device),
        "best_epoch": best_epoch,
        "best_validation_pr_auc":
            best_validation_pr_auc,
    }


def evaluate_model(
    model: nn.Module,
    forward_function,
    labels: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        logits = forward_function()

    synchronize(device)
    inference_seconds = time.perf_counter() - start_time

    threshold = select_validation_threshold(
        labels=labels,
        logits=logits,
        validation_mask=validation_mask,
    )

    metrics = calculate_metrics(
        labels=labels,
        logits=logits,
        mask=test_mask,
        threshold=threshold,
    )

    metrics["inference_seconds"] = inference_seconds
    metrics["inference_peak_gpu_memory_mb"] = (
        peak_gpu_memory_mb(device)
    )

    return metrics


def train_feature_classifier(
    feature_matrix: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
    head_type: str,
    pipeline_name: str,
) -> dict[str, float]:
    if head_type == "linear":
        model = LinearHead(feature_matrix.shape[1])
    elif head_type == "mlp":
        model = MLPHead(
            input_dimension=feature_matrix.shape[1],
            hidden_dimension=configuration.hidden_dimension,
            dropout=configuration.dropout,
        )
    else:
        raise ValueError(f"Unknown head: {head_type}")

    model = model.to(device)

    number_of_parameters = parameter_count(model)

    model, training_information = train_model(
        model=model,
        forward_function=lambda: model(feature_matrix),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        forward_function=lambda: model(feature_matrix),
        labels=labels,
        validation_mask=validation_mask,
        test_mask=test_mask,
        device=device,
    )

    result = {
        "pipeline": pipeline_name,
        "head_type": head_type,
        "trainable_parameters": number_of_parameters,
        **training_information,
        **test_metrics,
    }

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return result


def train_end_to_end_model(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
    head_type: str,
) -> tuple[
    dict[str, float],
    EndToEndGraphSAGE,
]:
    model = EndToEndGraphSAGE(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
        head_type=head_type,
    ).to(device)

    number_of_parameters = parameter_count(model)

    model, training_information = train_model(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    test_metrics = evaluate_model(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        validation_mask=validation_mask,
        test_mask=test_mask,
        device=device,
    )

    result = {
        "pipeline":
            f"end_to_end_graphsage_{head_type}",
        "head_type": head_type,
        "trainable_parameters": number_of_parameters,
        **training_information,
        **test_metrics,
    }

    return result, model


def extract_embeddings(
    encoder: GraphSAGEEncoder,
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoder.eval()

    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    with torch.no_grad():
        embeddings = encoder(graph, features)

    synchronize(device)
    extraction_seconds = time.perf_counter() - start_time

    information = {
        "embedding_extraction_seconds":
            extraction_seconds,
        "embedding_extraction_peak_gpu_memory_mb":
            peak_gpu_memory_mb(device),
    }

    return embeddings.detach(), information


def random_graphsage_embeddings(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    set_seed(configuration.seed)

    encoder = GraphSAGEEncoder(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
    ).to(device)

    embeddings, information = extract_embeddings(
        encoder=encoder,
        graph=graph,
        features=features,
        device=device,
    )

    del encoder
    gc.collect()
    torch.cuda.empty_cache()

    return embeddings, information


def neighbor_averaged_features(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    reset_peak_memory(device)
    synchronize(device)

    start_time = time.perf_counter()

    with graph.local_scope():
        graph.ndata["node_features"] = features

        graph.update_all(
            dgl.function.copy_u(
                "node_features",
                "message",
            ),
            dgl.function.mean(
                "message",
                "neighbor_mean",
            ),
        )

        neighbor_features = graph.ndata[
            "neighbor_mean"
        ]

    synchronize(device)
    extraction_seconds = time.perf_counter() - start_time

    information = {
        "embedding_extraction_seconds":
            extraction_seconds,
        "embedding_extraction_peak_gpu_memory_mb":
            peak_gpu_memory_mb(device),
    }

    return neighbor_features.detach(), information


def save_embedding_cache(
    embeddings: torch.Tensor,
    path: Path,
    configuration: Configuration,
    encoder_training_information:
        dict[str, float],
    extraction_information:
        dict[str, float],
) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)

    object_to_save = {
        "embeddings":
            embeddings.detach().cpu().contiguous(),
        "configuration":
            asdict(configuration),
        "encoder_training_information":
            encoder_training_information,
        "extraction_information":
            extraction_information,
    }

    torch.save(object_to_save, path)

    return path.stat().st_size / (1024**2)


def append_common_fields(
    result: dict[str, float],
    configuration: Configuration,
    representation_type: str,
    encoder_training_seconds: float = 0.0,
    embedding_extraction_seconds: float = 0.0,
    embedding_storage_mb: float = 0.0,
) -> dict[str, float]:
    result["seed"] = configuration.seed
    result["encoder_label_fraction"] = (
        configuration.encoder_label_fraction
    )
    result["downstream_label_fraction"] = (
        configuration.downstream_label_fraction
    )
    result["representation_type"] = representation_type
    result["encoder_training_seconds"] = (
        encoder_training_seconds
    )
    result["embedding_extraction_seconds"] = (
        embedding_extraction_seconds
    )
    result["embedding_storage_mb"] = (
        embedding_storage_mb
    )
    result["process_memory_mb_end"] = process_memory_mb()

    return result


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--encoder-label-fraction",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--downstream-label-fraction",
        type=float,
        default=1.0,
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

    configuration = Configuration(
        seed=arguments.seed,
        hidden_dimension=arguments.hidden_dimension,
        embedding_dimension=arguments.embedding_dimension,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
        encoder_label_fraction=
            arguments.encoder_label_fraction,
        downstream_label_fraction=
            arguments.downstream_label_fraction,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Check the environment."
        )

    for fraction_name, fraction in [
        (
            "encoder-label-fraction",
            configuration.encoder_label_fraction,
        ),
        (
            "downstream-label-fraction",
            configuration.downstream_label_fraction,
        ),
    ]:
        if not 0.0 < fraction <= 1.0:
            raise ValueError(
                f"{fraction_name} must be in (0, 1]."
            )

    set_seed(configuration.seed)
    device = torch.device("cuda:0")

    (
        graph,
        features,
        labels,
        full_train_mask,
        validation_mask,
        test_mask,
    ) = load_yelp_graph(Path("data"))

    encoder_train_mask = stratified_fraction_mask(
        labels=labels,
        original_mask=full_train_mask,
        fraction=configuration.encoder_label_fraction,
        seed=configuration.seed,
    )

    downstream_train_mask = stratified_fraction_mask(
        labels=labels,
        original_mask=full_train_mask,
        fraction=configuration.downstream_label_fraction,
        seed=configuration.seed + 10000,
    )

    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    encoder_train_mask = encoder_train_mask.to(device)
    downstream_train_mask = downstream_train_mask.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    print("\nConfiguration")
    print(json.dumps(asdict(configuration), indent=2))
    print("Nodes:", graph.num_nodes())
    print("Edges:", graph.num_edges())
    print(
        "Encoder training labels:",
        int(encoder_train_mask.sum()),
    )
    print(
        "Downstream training labels:",
        int(downstream_train_mask.sum()),
    )

    results = []

    # ---------------------------------------------------------
    # Raw-feature baselines
    # ---------------------------------------------------------
    for head_type in ["linear", "mlp"]:
        print(f"\nRaw features + {head_type}")

        set_seed(configuration.seed)

        raw_result = train_feature_classifier(
            feature_matrix=features,
            labels=labels,
            train_mask=downstream_train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=configuration,
            device=device,
            head_type=head_type,
            pipeline_name=f"raw_features_{head_type}",
        )

        results.append(
            append_common_fields(
                result=raw_result,
                configuration=configuration,
                representation_type="raw_features",
            )
        )

    # ---------------------------------------------------------
    # End-to-end GraphSAGE with matched heads
    # ---------------------------------------------------------
    trained_linear_graphsage = None
    linear_graphsage_training_information = None

    for head_type in ["linear", "mlp"]:
        print(
            f"\nEnd-to-end GraphSAGE + {head_type}"
        )

        set_seed(configuration.seed)

        graph_result, graph_model = (
            train_end_to_end_model(
                graph=graph,
                features=features,
                labels=labels,
                train_mask=encoder_train_mask,
                validation_mask=validation_mask,
                test_mask=test_mask,
                configuration=configuration,
                device=device,
                head_type=head_type,
            )
        )

        results.append(
            append_common_fields(
                result=graph_result,
                configuration=configuration,
                representation_type="joint_graphsage",
            )
        )

        if head_type == "linear":
            trained_linear_graphsage = graph_model
            linear_graphsage_training_information = {
                "training_seconds":
                    graph_result["training_seconds"],
                "peak_gpu_memory_mb":
                    graph_result["peak_gpu_memory_mb"],
                "best_epoch":
                    graph_result["best_epoch"],
            }
        else:
            del graph_model
            gc.collect()
            torch.cuda.empty_cache()

    if trained_linear_graphsage is None:
        raise RuntimeError(
            "The supervised encoder was not created."
        )

    # ---------------------------------------------------------
    # Extract and save one supervised embedding file
    # ---------------------------------------------------------
    print("\nExtracting supervised embeddings")

    supervised_embeddings, extraction_information = (
        extract_embeddings(
            encoder=trained_linear_graphsage.encoder,
            graph=graph,
            features=features,
            device=device,
        )
    )

    embedding_path = Path(
        "embeddings/ablation/"
        f"supervised_graphsage_seed_"
        f"{configuration.seed}_"
        f"encoder_fraction_"
        f"{configuration.encoder_label_fraction:.2f}.pt"
    )

    embedding_storage_mb = save_embedding_cache(
        embeddings=supervised_embeddings,
        path=embedding_path,
        configuration=configuration,
        encoder_training_information=
            linear_graphsage_training_information,
        extraction_information=
            extraction_information,
    )

    print("Saved:", embedding_path)
    print(
        "Embedding storage:",
        f"{embedding_storage_mb:.3f} MB",
    )

    del trained_linear_graphsage
    gc.collect()
    torch.cuda.empty_cache()

    # Reload the exact embedding cache from disk.
    cached_object = torch.load(
        embedding_path,
        map_location=device,
        weights_only=False,
    )

    supervised_embeddings = cached_object[
        "embeddings"
    ].to(device)

    # ---------------------------------------------------------
    # Frozen supervised embeddings with matched heads
    # ---------------------------------------------------------
    for head_type in ["linear", "mlp"]:
        print(
            f"\nFrozen supervised embeddings + "
            f"{head_type}"
        )

        set_seed(configuration.seed)

        frozen_result = train_feature_classifier(
            feature_matrix=supervised_embeddings,
            labels=labels,
            train_mask=downstream_train_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=configuration,
            device=device,
            head_type=head_type,
            pipeline_name=(
                f"frozen_supervised_graphsage_"
                f"{head_type}"
            ),
        )

        results.append(
            append_common_fields(
                result=frozen_result,
                configuration=configuration,
                representation_type=
                    "supervised_graphsage_embedding",
                encoder_training_seconds=
                    linear_graphsage_training_information[
                        "training_seconds"
                    ],
                embedding_extraction_seconds=
                    extraction_information[
                        "embedding_extraction_seconds"
                    ],
                embedding_storage_mb=
                    embedding_storage_mb,
            )
        )

    # ---------------------------------------------------------
    # Raw features concatenated with supervised embeddings
    # ---------------------------------------------------------
    print("\nRaw features + supervised embeddings + MLP")

    concatenated_features = torch.cat(
        [features, supervised_embeddings],
        dim=1,
    )

    set_seed(configuration.seed)

    concatenated_result = train_feature_classifier(
        feature_matrix=concatenated_features,
        labels=labels,
        train_mask=downstream_train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=configuration,
        device=device,
        head_type="mlp",
        pipeline_name="raw_plus_embedding_mlp",
    )

    results.append(
        append_common_fields(
            result=concatenated_result,
            configuration=configuration,
            representation_type=
                "raw_plus_supervised_embedding",
            encoder_training_seconds=
                linear_graphsage_training_information[
                    "training_seconds"
                ],
            embedding_extraction_seconds=
                extraction_information[
                    "embedding_extraction_seconds"
                ],
            embedding_storage_mb=embedding_storage_mb,
        )
    )

    del concatenated_features

    # ---------------------------------------------------------
    # Random untrained GraphSAGE baseline
    # ---------------------------------------------------------
    print("\nRandom GraphSAGE embeddings + MLP")

    random_embeddings, random_information = (
        random_graphsage_embeddings(
            graph=graph,
            features=features,
            configuration=configuration,
            device=device,
        )
    )

    set_seed(configuration.seed)

    random_result = train_feature_classifier(
        feature_matrix=random_embeddings,
        labels=labels,
        train_mask=downstream_train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=configuration,
        device=device,
        head_type="mlp",
        pipeline_name="random_graphsage_embedding_mlp",
    )

    results.append(
        append_common_fields(
            result=random_result,
            configuration=configuration,
            representation_type=
                "random_graphsage_embedding",
            embedding_extraction_seconds=
                random_information[
                    "embedding_extraction_seconds"
                ],
        )
    )

    del random_embeddings

    # ---------------------------------------------------------
    # Neighbor-mean baseline
    # ---------------------------------------------------------
    print("\nNeighbor-averaged features + MLP")

    neighbor_features, neighbor_information = (
        neighbor_averaged_features(
            graph=graph,
            features=features,
            device=device,
        )
    )

    set_seed(configuration.seed)

    neighbor_result = train_feature_classifier(
        feature_matrix=neighbor_features,
        labels=labels,
        train_mask=downstream_train_mask,
        validation_mask=validation_mask,
        test_mask=test_mask,
        configuration=configuration,
        device=device,
        head_type="mlp",
        pipeline_name="neighbor_mean_features_mlp",
    )

    results.append(
        append_common_fields(
            result=neighbor_result,
            configuration=configuration,
            representation_type=
                "neighbor_mean_features",
            embedding_extraction_seconds=
                neighbor_information[
                    "embedding_extraction_seconds"
                ],
        )
    )

    result_frame = pd.DataFrame(results)

    output_directory = Path("results/ablation")
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = output_directory / (
        f"matched_ablation_seed_"
        f"{configuration.seed}_"
        f"encoder_{configuration.encoder_label_fraction:.2f}_"
        f"downstream_"
        f"{configuration.downstream_label_fraction:.2f}.csv"
    )

    result_frame.to_csv(
        output_file,
        index=False,
    )

    print("\nFinal results")
    print(
        result_frame[
            [
                "pipeline",
                "roc_auc",
                "pr_auc",
                "f1",
                "training_seconds",
                "peak_gpu_memory_mb",
                "best_epoch",
            ]
        ].to_string(index=False)
    )

    print("\nSaved:", output_file)


if __name__ == "__main__":
    main()
