from pathlib import Path

import dgl
import torch
from dgl.data import FraudYelpDataset


def main() -> None:
    data_directory = Path("data")
    data_directory.mkdir(parents=True, exist_ok=True)

    print("Downloading or loading FraudYelpDataset...")

    dataset = FraudYelpDataset(
        raw_dir=str(data_directory),
        random_seed=717,
        train_size=0.70,
        val_size=0.10,
        force_reload=False,
        verbose=True,
    )

    graph = dataset[0]

    print("\nDataset loaded successfully")
    print("=" * 70)
    print("Dataset classes:", dataset.num_classes)
    print("Node types:", graph.ntypes)
    print("Canonical edge types:")

    for edge_type in graph.canonical_etypes:
        print(f"  {edge_type}: {graph.num_edges(edge_type):,} edges")

    print("\nTotal nodes:", f"{graph.num_nodes():,}")
    print("Total edges:", f"{graph.num_edges():,}")

    print("\nNode-data fields:", list(graph.ndata.keys()))

    features = graph.ndata["feature"]
    labels = graph.ndata["label"].long().view(-1)
    train_mask = graph.ndata["train_mask"].bool().view(-1)
    validation_mask = graph.ndata["val_mask"].bool().view(-1)
    test_mask = graph.ndata["test_mask"].bool().view(-1)

    print("Feature shape:", tuple(features.shape))
    print("Label shape:", tuple(labels.shape))
    print("Feature dtype:", features.dtype)
    print("Label dtype:", labels.dtype)

    print("\nSplit sizes")
    print("Train:", f"{train_mask.sum().item():,}")
    print("Validation:", f"{validation_mask.sum().item():,}")
    print("Test:", f"{test_mask.sum().item():,}")

    print("\nClass counts")
    unique_labels, counts = torch.unique(labels, return_counts=True)

    for label, count in zip(unique_labels.tolist(), counts.tolist()):
        print(f"Class {label}: {count:,}")

    positive_count = int((labels == 1).sum())
    negative_count = int((labels == 0).sum())

    print("Positive fraction:", positive_count / labels.numel())
    print("Negative-to-positive ratio:", negative_count / positive_count)

    print("\nCUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print("\nDataset check completed successfully.")


if __name__ == "__main__":
    main()
