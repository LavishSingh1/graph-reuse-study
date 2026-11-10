from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

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
    roc_auc_score,
)


@dataclass
class Configuration:
    hidden_dimension: int = 64
    embedding_dimension: int = 64
    dropout: float = 0.30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    maximum_epochs: int = 300
    patience: int = 30
    label_fraction: float = 1.0
    seed: int = 42
    graph_device: str = "cuda"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic execution is useful for controlled experiments.
    # Some sparse operations can still vary slightly between drivers.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def synchronize_if_needed(device: torch.device) -> None:
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
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024**2)


def binary_metrics(
    labels: torch.Tensor,
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    selected_labels = labels[mask].detach().cpu().numpy()
    selected_logits = logits[mask].detach().float().cpu()
    probabilities = torch.sigmoid(selected_logits).numpy()
    predictions = (probabilities >= 0.5).astype(np.int64)

    return {
        "roc_auc": float(roc_auc_score(selected_labels, probabilities)),
        "pr_auc": float(
            average_precision_score(selected_labels, probabilities)
        ),
        "f1": float(f1_score(selected_labels, predictions, zero_division=0)),
    }


def stratified_label_fraction(
    labels: torch.Tensor,
    original_mask: torch.Tensor,
    fraction: float,
    seed: int,
) -> torch.Tensor:
    if fraction >= 1.0:
        return original_mask.clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    labels_cpu = labels.detach().cpu()
    mask_cpu = original_mask.detach().cpu()
    selected_mask = torch.zeros_like(mask_cpu, dtype=torch.bool)

    for class_value in torch.unique(labels_cpu[mask_cpu]):
        candidates = torch.where(
            mask_cpu & (labels_cpu == class_value)
        )[0]

        number_to_select = max(
            1,
            int(round(len(candidates) * fraction)),
        )

        ordering = torch.randperm(
            len(candidates),
            generator=generator,
        )

        selected = candidates[ordering[:number_to_select]]
        selected_mask[selected] = True

    return selected_mask.to(original_mask.device)


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

    # YelpChi has one node type and three edge relations.
    # We explicitly convert the relation-preserving heterograph into a
    # homogeneous graph by taking the union of all relation-specific edges.
    # The relation identity remains available in dgl.ETYPE, although the
    # baseline GraphSAGE model below does not use relation-specific weights.
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

    # Ensure every node has a self-loop exactly once.
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
        heterograph.canonical_etypes,
    )


class MLP(nn.Module):
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).view(-1)


class GraphSAGEEncoder(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        embedding_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.convolution_1 = SAGEConv(
            input_dimension,
            hidden_dimension,
            aggregator_type="mean",
        )
        self.convolution_2 = SAGEConv(
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
        hidden = self.convolution_1(graph, features)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)

        embeddings = self.convolution_2(graph, hidden)
        return embeddings


class GraphSAGEClassifier(nn.Module):
    def __init__(
        self,
        input_dimension: int,
        hidden_dimension: int,
        embedding_dimension: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.encoder = GraphSAGEEncoder(
            input_dimension=input_dimension,
            hidden_dimension=hidden_dimension,
            embedding_dimension=embedding_dimension,
            dropout=dropout,
        )

        self.classifier = nn.Linear(embedding_dimension, 1)

    def forward(
        self,
        graph: dgl.DGLGraph,
        features: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.encoder(graph, features)
        return self.classifier(embeddings).view(-1)


def positive_class_weight(
    labels: torch.Tensor,
    train_mask: torch.Tensor,
) -> torch.Tensor:
    selected_labels = labels[train_mask]

    positive_count = (selected_labels == 1).sum().float()
    negative_count = (selected_labels == 0).sum().float()

    return negative_count / positive_count.clamp_min(1.0)


def train_with_early_stopping(
    model: nn.Module,
    forward_function: Callable[[], torch.Tensor],
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> tuple[nn.Module, dict[str, float]]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=configuration.learning_rate,
        weight_decay=configuration.weight_decay,
    )

    class_weight = positive_class_weight(
        labels,
        train_mask,
    ).to(device)

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=class_weight,
    )

    best_validation_pr_auc = float("-inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    reset_peak_memory(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()

    for epoch in range(1, configuration.maximum_epochs + 1):
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
            validation_metrics = binary_metrics(
                labels,
                validation_logits,
                validation_mask,
            )

        validation_pr_auc = validation_metrics["pr_auc"]

        if validation_pr_auc > best_validation_pr_auc:
            best_validation_pr_auc = validation_pr_auc
            best_epoch = epoch
            epochs_without_improvement = 0

            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"loss={loss.item():.5f} | "
                f"val PR-AUC={validation_pr_auc:.5f}"
            )

        if epochs_without_improvement >= configuration.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    synchronize_if_needed(device)
    training_time = time.perf_counter() - start_time
    maximum_gpu_memory = peak_gpu_memory_mb(device)

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)

    return model, {
        "training_time_seconds": training_time,
        "peak_gpu_memory_mb": maximum_gpu_memory,
        "best_epoch": best_epoch,
        "best_validation_pr_auc": best_validation_pr_auc,
    }


def evaluate_model(
    model: nn.Module,
    forward_function: Callable[[], torch.Tensor],
    labels: torch.Tensor,
    test_mask: torch.Tensor,
    device: torch.device,
) -> tuple[dict[str, float], torch.Tensor]:
    model.eval()

    reset_peak_memory(device)
    synchronize_if_needed(device)
    start_time = time.perf_counter()

    with torch.no_grad():
        logits = forward_function()

    synchronize_if_needed(device)
    inference_time = time.perf_counter() - start_time

    metrics = binary_metrics(
        labels,
        logits,
        test_mask,
    )

    metrics["inference_time_seconds"] = inference_time
    metrics["inference_peak_gpu_memory_mb"] = peak_gpu_memory_mb(device)

    return metrics, logits


def run_raw_mlp(
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> dict[str, float]:
    model = MLP(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        dropout=configuration.dropout,
    ).to(device)

    model, training_information = train_with_early_stopping(
        model=model,
        forward_function=lambda: model(features),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    test_metrics, _ = evaluate_model(
        model=model,
        forward_function=lambda: model(features),
        labels=labels,
        test_mask=test_mask,
        device=device,
    )

    return {
        "pipeline": "raw_mlp",
        **training_information,
        **test_metrics,
        "embedding_training_seconds": 0.0,
        "embedding_extraction_seconds": 0.0,
        "embedding_storage_mb": 0.0,
    }


def run_end_to_end_graphsage(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
) -> dict[str, float]:
    model = GraphSAGEClassifier(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
    ).to(device)

    model, training_information = train_with_early_stopping(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        train_mask=train_mask,
        validation_mask=validation_mask,
        configuration=configuration,
        device=device,
    )

    test_metrics, _ = evaluate_model(
        model=model,
        forward_function=lambda: model(graph, features),
        labels=labels,
        test_mask=test_mask,
        device=device,
    )

    return {
        "pipeline": "end_to_end_graphsage",
        **training_information,
        **test_metrics,
        "embedding_training_seconds": 0.0,
        "embedding_extraction_seconds": 0.0,
        "embedding_storage_mb": 0.0,
    }


def run_precomputed_pipeline(
    graph: dgl.DGLGraph,
    features: torch.Tensor,
    labels: torch.Tensor,
    train_mask: torch.Tensor,
    validation_mask: torch.Tensor,
    test_mask: torch.Tensor,
    configuration: Configuration,
    device: torch.device,
    embedding_directory: Path,
) -> dict[str, float]:
    encoder_classifier = GraphSAGEClassifier(
        input_dimension=features.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        embedding_dimension=configuration.embedding_dimension,
        dropout=configuration.dropout,
    ).to(device)

    encoder_classifier, encoder_training_information = (
        train_with_early_stopping(
            model=encoder_classifier,
            forward_function=lambda: encoder_classifier(graph, features),
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            configuration=configuration,
            device=device,
        )
    )

    encoder_classifier.eval()

    reset_peak_memory(device)
    synchronize_if_needed(device)
    extraction_start = time.perf_counter()

    with torch.no_grad():
        embeddings = encoder_classifier.encoder(
            graph,
            features,
        )

    synchronize_if_needed(device)
    embedding_extraction_seconds = (
        time.perf_counter() - extraction_start
    )

    embeddings_cpu = embeddings.detach().cpu().contiguous()

    embedding_directory.mkdir(parents=True, exist_ok=True)
    embedding_file = embedding_directory / (
        f"yelp_graphsage_seed_{configuration.seed}"
        f"_fraction_{configuration.label_fraction:.2f}.pt"
    )

    torch.save(
        {
            "embeddings": embeddings_cpu,
            "seed": configuration.seed,
            "label_fraction": configuration.label_fraction,
            "configuration": asdict(configuration),
        },
        embedding_file,
    )

    embedding_storage_mb = embedding_file.stat().st_size / (1024**2)

    # Explicitly reload the cache to demonstrate that downstream training
    # does not require rerunning the graph encoder.
    cached_object = torch.load(
        embedding_file,
        map_location=device,
        weights_only=False,
    )
    cached_embeddings = cached_object["embeddings"].to(device)

    del embeddings
    del embeddings_cpu
    del encoder_classifier

    gc.collect()
    torch.cuda.empty_cache()

    downstream_mlp = MLP(
        input_dimension=cached_embeddings.shape[1],
        hidden_dimension=configuration.hidden_dimension,
        dropout=configuration.dropout,
    ).to(device)

    downstream_mlp, downstream_training_information = (
        train_with_early_stopping(
            model=downstream_mlp,
            forward_function=lambda: downstream_mlp(cached_embeddings),
            labels=labels,
            train_mask=train_mask,
            validation_mask=validation_mask,
            configuration=configuration,
            device=device,
        )
    )

    test_metrics, _ = evaluate_model(
        model=downstream_mlp,
        forward_function=lambda: downstream_mlp(cached_embeddings),
        labels=labels,
        test_mask=test_mask,
        device=device,
    )

    return {
        "pipeline": "precomputed_graphsage_mlp",
        "training_time_seconds": downstream_training_information[
            "training_time_seconds"
        ],
        "peak_gpu_memory_mb": downstream_training_information[
            "peak_gpu_memory_mb"
        ],
        "best_epoch": downstream_training_information["best_epoch"],
        "best_validation_pr_auc": downstream_training_information[
            "best_validation_pr_auc"
        ],
        **test_metrics,
        "embedding_training_seconds": encoder_training_information[
            "training_time_seconds"
        ],
        "embedding_training_peak_gpu_memory_mb":
            encoder_training_information["peak_gpu_memory_mb"],
        "embedding_extraction_seconds": embedding_extraction_seconds,
        "embedding_storage_mb": embedding_storage_mb,
        "embedding_file": str(embedding_file),
    }


def save_environment_information(
    output_directory: Path,
    graph: dgl.DGLGraph,
    canonical_edge_types,
    features: torch.Tensor,
    device: torch.device,
) -> None:
    information = {
        "python": os.sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "dgl": dgl.__version__,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        "gpu_count": torch.cuda.device_count(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "system_ram_gb": psutil.virtual_memory().total / (1024**3),
        "node_count": graph.num_nodes(),
        "edge_count_including_self_loops": graph.num_edges(),
        "feature_dimension": features.shape[1],
        "canonical_edge_types": [str(item) for item in canonical_edge_types],
    }

    output_directory.mkdir(parents=True, exist_ok=True)

    with open(
        output_directory / "environment.json",
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(information, output_file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--label-fraction",
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
    parser.add_argument(
        "--pipeline",
        choices=[
            "all",
            "raw_mlp",
            "end_to_end_graphsage",
            "precomputed",
        ],
        default="all",
    )

    arguments = parser.parse_args()

    if not 0.0 < arguments.label_fraction <= 1.0:
        raise ValueError("--label-fraction must be in (0, 1].")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run the PyTorch CUDA verification first."
        )

    configuration = Configuration(
        hidden_dimension=arguments.hidden_dimension,
        embedding_dimension=arguments.embedding_dimension,
        maximum_epochs=arguments.maximum_epochs,
        patience=arguments.patience,
        label_fraction=arguments.label_fraction,
        seed=arguments.seed,
    )

    set_seed(configuration.seed)

    data_directory = Path("data")
    embedding_directory = Path("embeddings")
    result_directory = Path("results")
    result_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0")

    (
        graph,
        features,
        labels,
        full_train_mask,
        validation_mask,
        test_mask,
        canonical_edge_types,
    ) = load_yelp_graph(data_directory)

    training_mask = stratified_label_fraction(
        labels=labels,
        original_mask=full_train_mask,
        fraction=configuration.label_fraction,
        seed=configuration.seed,
    )

    print("\nExperiment configuration")
    print(json.dumps(asdict(configuration), indent=2))
    print("Selected training labels:", int(training_mask.sum()))
    print("Graph nodes:", graph.num_nodes())
    print("Graph edges:", graph.num_edges())

    # Full-graph baseline. YelpChi should be tested first on one GPU.
    graph = graph.to(device)
    features = features.to(device)
    labels = labels.to(device)
    training_mask = training_mask.to(device)
    validation_mask = validation_mask.to(device)
    test_mask = test_mask.to(device)

    save_environment_information(
        output_directory=result_directory,
        graph=graph,
        canonical_edge_types=canonical_edge_types,
        features=features,
        device=device,
    )

    results = []

    if arguments.pipeline in ("all", "raw_mlp"):
        print("\nRunning raw-feature MLP")
        set_seed(configuration.seed)

        raw_result = run_raw_mlp(
            features=features,
            labels=labels,
            train_mask=training_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=configuration,
            device=device,
        )
        results.append(raw_result)

    if arguments.pipeline in ("all", "end_to_end_graphsage"):
        print("\nRunning end-to-end GraphSAGE")
        set_seed(configuration.seed)

        graph_result = run_end_to_end_graphsage(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=training_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=configuration,
            device=device,
        )
        results.append(graph_result)

    if arguments.pipeline in ("all", "precomputed"):
        print("\nRunning precomputed GraphSAGE plus MLP")
        set_seed(configuration.seed)

        precomputed_result = run_precomputed_pipeline(
            graph=graph,
            features=features,
            labels=labels,
            train_mask=training_mask,
            validation_mask=validation_mask,
            test_mask=test_mask,
            configuration=configuration,
            device=device,
            embedding_directory=embedding_directory,
        )
        results.append(precomputed_result)

    for result in results:
        result["seed"] = configuration.seed
        result["label_fraction"] = configuration.label_fraction
        result["hidden_dimension"] = configuration.hidden_dimension
        result["embedding_dimension"] = configuration.embedding_dimension
        result["cpu_process_memory_mb_end"] = process_memory_mb()

    result_frame = pd.DataFrame(results)

    result_file = result_directory / (
        f"run_seed_{configuration.seed}"
        f"_fraction_{configuration.label_fraction:.2f}.csv"
    )

    result_frame.to_csv(result_file, index=False)

    print("\nFinal results")
    print(result_frame.to_string(index=False))
    print(f"\nSaved results to {result_file}")


if __name__ == "__main__":
    main()
