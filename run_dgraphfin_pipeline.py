from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.datasets import DGraphFin
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm


@dataclass
class Config:
    seed: int
    hidden_dim: int = 64
    embedding_dim: int = 64
    dropout: float = 0.3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    patience: int = 8
    graph_batch_size: int = 4096
    inference_batch_size: int = 65536
    mlp_batch_size: int = 65536
    num_neighbors_1: int = 15
    num_neighbors_2: int = 10
    workers: int = 8


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / 1024**2


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


def indices_from_mask(mask: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if mask.dtype == torch.bool:
        idx = mask.nonzero(as_tuple=False).view(-1)
    else:
        idx = mask.view(-1).long()
    valid = (labels[idx] == 0) | (labels[idx] == 1)
    return idx[valid]


def metrics(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities >= threshold).astype(np.int64)
    return {
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
    }


def best_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if thresholds.size == 0:
        return 0.5
    values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(values))])


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


class SampledSAGE(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList([
            SAGEConv(input_dim, hidden_dim),
            SAGEConv(hidden_dim, embedding_dim),
        ])
        self.dropout = dropout
        self.head = MLP(embedding_dim, hidden_dim, dropout)

    def encode_sampled(self, x, edge_index):
        x = self.convs[0](x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[1](x, edge_index)

    def forward(self, x, edge_index):
        return self.head(self.encode_sampled(x, edge_index))


def class_weight(labels: torch.Tensor, idx: torch.Tensor, device: torch.device) -> torch.Tensor:
    selected = labels[idx]
    pos = (selected == 1).sum().float()
    neg = (selected == 0).sum().float()
    return (neg / pos.clamp_min(1)).to(device)


def train_mlp(features, labels, train_idx, val_idx, test_idx, cfg, device, name):
    model = MLP(features.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(labels, train_idx, device))
    train_ds = TensorDataset(features[train_idx], labels[train_idx].float())
    train_loader = DataLoader(train_ds, batch_size=cfg.mlp_batch_size, shuffle=True, num_workers=cfg.workers, pin_memory=True)

    best_pr = -1.0
    best_state = None
    best_epoch = 0
    stale = 0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync()
    start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); optimizer.step()
        val_prob = predict_mlp(model, features, val_idx, cfg, device)
        score = average_precision_score(labels[val_idx].numpy(), val_prob)
        if score > best_pr:
            best_pr, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= cfg.patience:
            break

    sync(); training_seconds = time.perf_counter() - start
    peak = gpu_peak_mb()
    model.load_state_dict(best_state)
    val_prob = predict_mlp(model, features, val_idx, cfg, device)
    threshold = best_threshold(labels[val_idx].numpy(), val_prob)
    sync(); inference_start = time.perf_counter()
    test_prob = predict_mlp(model, features, test_idx, cfg, device)
    sync(); inference_seconds = time.perf_counter() - inference_start
    result = {
        "pipeline": name, **metrics(labels[test_idx].numpy(), test_prob, threshold),
        "threshold": threshold, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "peak_gpu_memory_mb": peak,
        "best_epoch": best_epoch, "best_validation_pr_auc": best_pr,
    }
    del model, train_loader, train_ds
    gc.collect(); torch.cuda.empty_cache()
    return result


@torch.no_grad()
def predict_mlp(model, features, idx, cfg, device):
    model.eval()
    output = []
    for start in range(0, idx.numel(), cfg.mlp_batch_size):
        batch_idx = idx[start:start + cfg.mlp_batch_size]
        output.append(torch.sigmoid(model(features[batch_idx].to(device))).cpu())
    return torch.cat(output).numpy()


def make_neighbor_loader(data, input_nodes, cfg, shuffle):
    return NeighborLoader(
        data,
        input_nodes=input_nodes,
        num_neighbors=[cfg.num_neighbors_1, cfg.num_neighbors_2],
        batch_size=cfg.graph_batch_size,
        shuffle=shuffle,
        num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0,
        pin_memory=True,
    )


def train_graphsage(data, labels, train_idx, val_idx, test_idx, cfg, device):
    model = SampledSAGE(data.x.shape[1], cfg.hidden_dim, cfg.embedding_dim, cfg.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(labels, train_idx, device))
    train_loader = make_neighbor_loader(data, train_idx, cfg, True)
    val_loader = make_neighbor_loader(data, val_idx, cfg, False)
    test_loader = make_neighbor_loader(data, test_idx, cfg, False)

    best_pr = -1.0; best_state = None; best_epoch = 0; stale = 0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync(); start = time.perf_counter()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"seed {cfg.seed} graph epoch {epoch}", leave=False):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.x, batch.edge_index)[:batch.batch_size]
            y = batch.y[:batch.batch_size].float()
            loss = loss_fn(logits, y)
            loss.backward(); optimizer.step()
        y_val, p_val = predict_graph(model, val_loader, device)
        score = average_precision_score(y_val, p_val)
        if score > best_pr:
            best_pr, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        print(f"seed={cfg.seed} epoch={epoch} val_pr_auc={score:.6f} best={best_pr:.6f}")
        if stale >= cfg.patience:
            break

    sync(); training_seconds = time.perf_counter() - start; peak = gpu_peak_mb()
    model.load_state_dict(best_state)
    y_val, p_val = predict_graph(model, val_loader, device)
    threshold = best_threshold(y_val, p_val)
    sync(); inference_start = time.perf_counter()
    y_test, p_test = predict_graph(model, test_loader, device)
    sync(); inference_seconds = time.perf_counter() - inference_start
    result = {
        "pipeline": "end_to_end_graphsage_mlp", **metrics(y_test, p_test, threshold),
        "threshold": threshold, "training_seconds": training_seconds,
        "inference_seconds": inference_seconds, "peak_gpu_memory_mb": peak,
        "best_epoch": best_epoch, "best_validation_pr_auc": best_pr,
    }
    del train_loader, val_loader, test_loader
    gc.collect(); torch.cuda.empty_cache()
    return result, model


@torch.no_grad()
def predict_graph(model, loader, device):
    model.eval(); ys = []; probs = []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x, batch.edge_index)[:batch.batch_size]
        ys.append(batch.y[:batch.batch_size].cpu())
        probs.append(torch.sigmoid(logits).cpu())
    return torch.cat(ys).numpy(), torch.cat(probs).numpy()


@torch.no_grad()
def exact_layerwise_embeddings(model, data, cfg, device):
    model.eval()
    x_all = data.x.cpu()
    total_start = time.perf_counter()
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

    for layer_index, conv in enumerate(model.convs):
        loader = NeighborLoader(
            data,
            input_nodes=torch.arange(data.num_nodes),
            num_neighbors=[-1],
            batch_size=cfg.inference_batch_size,
            shuffle=False,
            num_workers=cfg.workers,
            persistent_workers=cfg.workers > 0,
            pin_memory=True,
        )
        out_dim = conv.out_channels
        output = torch.empty((data.num_nodes, out_dim), dtype=torch.float32)
        for batch in tqdm(loader, desc=f"exact embedding layer {layer_index + 1}"):
            n_id = batch.n_id
            local_x = x_all[n_id].to(device, non_blocking=True)
            local_edge = batch.edge_index.to(device, non_blocking=True)
            h = conv(local_x, local_edge)[:batch.batch_size]
            if layer_index == 0:
                h = F.relu(h)
            output[n_id[:batch.batch_size]] = h.cpu()
        x_all = output
        del loader, output
        gc.collect(); torch.cuda.empty_cache()

    sync()
    return x_all, time.perf_counter() - total_start, gpu_peak_mb()


def load_data(root: Path):
    zip_path = root / "raw" / "DGraphFin.zip"
    if not zip_path.exists():
        raise FileNotFoundError(
            f"Missing {zip_path}. Download DGraphFin.zip from the official DGraph site and place it there."
        )
    dataset = DGraphFin(root=str(root))
    data = dataset[0]
    data.x = data.x.float()
    data.y = data.y.view(-1).long()
    return data


def run_seed(data, cfg, output_dir, embedding_dir):
    seed_everything(cfg.seed)
    device = torch.device("cuda:0")
    labels = data.y.cpu()
    train_idx = indices_from_mask(data.train_mask, labels)
    val_idx = indices_from_mask(data.val_mask, labels)
    test_idx = indices_from_mask(data.test_mask, labels)

    print(json.dumps(asdict(cfg), indent=2))
    print(f"nodes={data.num_nodes:,} edges={data.num_edges:,} features={data.x.shape[1]}")
    print(f"train={train_idx.numel():,} val={val_idx.numel():,} test={test_idx.numel():,}")

    results = []
    raw = train_mlp(data.x.cpu(), labels, train_idx, val_idx, test_idx, cfg, device, "raw_features_mlp")
    raw.update({"seed": cfg.seed, "one_time_encoder_seconds": 0.0, "embedding_extraction_seconds": 0.0, "embedding_storage_mb": 0.0, "embedding_load_seconds": 0.0})
    results.append(raw)

    graph_result, graph_model = train_graphsage(data, labels, train_idx, val_idx, test_idx, cfg, device)
    graph_result.update({"seed": cfg.seed, "one_time_encoder_seconds": 0.0, "embedding_extraction_seconds": 0.0, "embedding_storage_mb": 0.0, "embedding_load_seconds": 0.0})
    results.append(graph_result)

    embeddings, extraction_seconds, extraction_peak = exact_layerwise_embeddings(graph_model, data, cfg, device)
    embedding_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = embedding_dir / f"dgraphfin_graphsage_seed_{cfg.seed}.pt"
    save_start = time.perf_counter()
    torch.save({"embeddings": embeddings, "config": asdict(cfg)}, embedding_path)
    save_seconds = time.perf_counter() - save_start
    storage_mb = embedding_path.stat().st_size / 1024**2
    del embeddings, graph_model
    gc.collect(); torch.cuda.empty_cache()

    load_start = time.perf_counter()
    cached = torch.load(embedding_path, map_location="cpu", weights_only=False)["embeddings"]
    load_seconds = time.perf_counter() - load_start
    frozen = train_mlp(cached, labels, train_idx, val_idx, test_idx, cfg, device, "frozen_graphsage_mlp")
    frozen.update({
        "seed": cfg.seed,
        "one_time_encoder_seconds": graph_result["training_seconds"],
        "embedding_extraction_seconds": extraction_seconds,
        "embedding_extraction_peak_gpu_memory_mb": extraction_peak,
        "embedding_save_seconds": save_seconds,
        "embedding_storage_mb": storage_mb,
        "embedding_load_seconds": load_seconds,
    })
    results.append(frozen)

    for row in results:
        one_time = row.get("one_time_encoder_seconds", 0.0) + row.get("embedding_extraction_seconds", 0.0) + row.get("embedding_save_seconds", 0.0)
        row["first_run_seconds"] = one_time + row["training_seconds"] + row.get("embedding_load_seconds", 0.0)
        row["cpu_rss_mb_end"] = rss_mb()
        row["max_process_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        row["num_nodes"] = data.num_nodes
        row["num_edges"] = data.num_edges
        row["embedding_dimension"] = cfg.embedding_dim

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"dgraphfin_seed_{cfg.seed}.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(pd.DataFrame(results)[["pipeline", "pr_auc", "roc_auc", "f1", "training_seconds", "first_run_seconds", "peak_gpu_memory_mb"]].to_string(index=False))
    print("saved", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--graph-batch-size", type=int, default=4096)
    parser.add_argument("--inference-batch-size", type=int, default=65536)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    cfg = Config(seed=args.seed, epochs=args.epochs, patience=args.patience, graph_batch_size=args.graph_batch_size, inference_batch_size=args.inference_batch_size, workers=args.workers)
    data = load_data(Path("data/dgraphfin"))
    run_seed(data, cfg, Path("results/dgraphfin"), Path("embeddings/dgraphfin"))


if __name__ == "__main__":
    main()
