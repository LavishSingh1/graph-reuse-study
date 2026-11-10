"""DGraph-Fin embedding-quantization experiment (reviewer R1/R2 request).

Reuses the exact data loading, encoder, exact layer-wise materialization, MLP and timing
code from run_dgraphfin_pipeline.py. For each seed it:
  1. trains the sampled end-to-end GraphSAGE once (this is both the reference C_b and the encoder C_e),
  2. materializes exact FP32 embeddings once (C_m),
  3. for each precision in {fp32, fp16, int8}: quantizes, saves, loads, and fits the same MLP head,
  4. does step 3 under two consumption modes:
       dataloader : identical to the paper's frozen pipeline (CPU tensor + torch DataLoader)
       gpu        : embeddings resident on the GPU in their stored dtype, decoded per batch
                    (diagnostic: separates "dense feature consumption" cost from DataLoader overhead)
Run:  python run_dgraphfin_quantization.py --seed 42   (see run_dgraphfin_quant.sh)
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from run_dgraphfin_pipeline import (
    MLP, Config, best_threshold, class_weight, exact_layerwise_embeddings, gpu_peak_mb,
    indices_from_mask, load_data, metrics, seed_everything, sync, train_graphsage, train_mlp,
)

PRECISIONS = ("fp32", "fp16", "int8")
FIT_MODES = ("dataloader", "gpu")


# ---------------------------------------------------------------- quantization
def quantize(emb: torch.Tensor, precision: str) -> dict:
    """Return a dict that is saved to disk. int8 is symmetric, per-column scale."""
    if precision == "fp32":
        return {"precision": precision, "q": emb.contiguous()}
    if precision == "fp16":
        return {"precision": precision, "q": emb.half().contiguous()}
    if precision == "int8":
        scale = (emb.abs().amax(dim=0) / 127.0).clamp_min(1e-12)
        q = torch.round(emb / scale).clamp_(-127, 127).to(torch.int8).contiguous()
        return {"precision": precision, "q": q, "scale": scale.float()}
    raise ValueError(precision)


def dequantize_cpu(blob: dict) -> torch.Tensor:
    if blob["precision"] == "int8":
        return blob["q"].float() * blob["scale"]
    return blob["q"].float()


def make_decoder(blob: dict, device):
    """Per-batch GPU decode back to FP32 (what the MLP consumes)."""
    if blob["precision"] == "int8":
        scale = blob["scale"].to(device)
        return lambda x: x.float() * scale
    return lambda x: x.float()


# ---------------------------------------------------------------- GPU-resident fit
@torch.no_grad()
def _predict_gpu(model, store, decode, idx, cfg, device):
    model.eval()
    out = []
    for s in range(0, idx.numel(), cfg.mlp_batch_size):
        b = idx[s:s + cfg.mlp_batch_size].to(device)
        out.append(torch.sigmoid(model(decode(store[b]))).cpu())
    return torch.cat(out).numpy()


def train_mlp_gpu(store, decode, labels, train_idx, val_idx, test_idx, cfg, device, name):
    """Same model/optimizer/early-stopping as train_mlp; batches built on the GPU (no DataLoader)."""
    model = MLP(store.shape[1], cfg.hidden_dim, cfg.dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=class_weight(labels, train_idx, device))
    labels_gpu = labels.to(device)
    train_gpu = train_idx.to(device)
    n = train_gpu.numel()
    best_pr, best_state, best_epoch, stale = -1.0, None, 0, 0
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); sync()
    start = time.perf_counter()
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        perm = train_gpu[torch.randperm(n, device=device)]
        for s in range(0, n, cfg.mlp_batch_size):
            b = perm[s:s + cfg.mlp_batch_size]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(decode(store[b])), labels_gpu[b].float())
            loss.backward(); opt.step()
        val_prob = _predict_gpu(model, store, decode, val_idx, cfg, device)
        score = average_precision_score(labels[val_idx].numpy(), val_prob)
        if score > best_pr:
            best_pr, best_epoch, stale = score, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    sync(); train_seconds = time.perf_counter() - start; peak = gpu_peak_mb()
    model.load_state_dict(best_state)
    val_prob = _predict_gpu(model, store, decode, val_idx, cfg, device)
    thr = best_threshold(labels[val_idx].numpy(), val_prob)
    sync(); t0 = time.perf_counter()
    test_prob = _predict_gpu(model, store, decode, test_idx, cfg, device)
    sync(); inf = time.perf_counter() - t0
    res = {"pipeline": name, **metrics(labels[test_idx].numpy(), test_prob, thr), "threshold": thr,
           "training_seconds": train_seconds, "inference_seconds": inf, "peak_gpu_memory_mb": peak,
           "best_epoch": best_epoch, "best_validation_pr_auc": best_pr}
    del model, labels_gpu, train_gpu
    gc.collect(); torch.cuda.empty_cache()
    return res


# ---------------------------------------------------------------- main experiment
def run_seed(data, cfg, precisions, fit_modes, out_dir: Path, emb_dir: Path):
    seed_everything(cfg.seed)
    device = torch.device("cuda:0")
    labels = data.y.cpu()
    train_idx = indices_from_mask(data.train_mask, labels)
    val_idx = indices_from_mask(data.val_mask, labels)
    test_idx = indices_from_mask(data.test_mask, labels)
    print(json.dumps(asdict(cfg), indent=2))

    graph_res, model = train_graphsage(data, labels, train_idx, val_idx, test_idx, cfg, device)
    ref_seconds = graph_res["training_seconds"]              # C_b == C_e
    emb, extract_seconds, extract_peak = exact_layerwise_embeddings(model, data, cfg, device)
    del model; gc.collect(); torch.cuda.empty_cache()
    print(f"reference e2e: {ref_seconds:.2f}s PR-AUC={graph_res['pr_auc']:.4f}; extraction {extract_seconds:.2f}s")

    emb_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for prec in precisions:
        # --- one-time: quantize + save (counted in C_m)
        t0 = time.perf_counter()
        blob = quantize(emb, prec)
        quant_s = time.perf_counter() - t0
        path = emb_dir / f"dgraphfin_quant_{prec}_seed_{cfg.seed}.pt"
        t0 = time.perf_counter(); torch.save(blob, path); save_s = time.perf_counter() - t0
        storage_mb = path.stat().st_size / 1024**2
        # embedding error vs FP32 (diagnostic)
        err = (dequantize_cpu(blob) - emb).abs()
        rel_rmse = float((err.pow(2).mean().sqrt() / emb.pow(2).mean().sqrt()).item())
        del blob

        for mode in fit_modes:
            # --- per-workload: load (+ decode/transfer) then downstream fit
            sync(); t0 = time.perf_counter()
            loaded = torch.load(path, map_location="cpu", weights_only=False)
            if mode == "dataloader":
                feats = dequantize_cpu(loaded)                       # dequantize on CPU, counted in C_l
                load_s = time.perf_counter() - t0
                seed_everything(cfg.seed)
                res = train_mlp(feats, labels, train_idx, val_idx, test_idx, cfg, device, f"frozen_{prec}")
                del feats
            else:
                store = loaded["q"].to(device)                       # H2D of the *stored* dtype, counted in C_l
                sync(); load_s = time.perf_counter() - t0
                decode = make_decoder(loaded, device)
                seed_everything(cfg.seed)
                res = train_mlp_gpu(store, decode, labels, train_idx, val_idx, test_idx, cfg, device, f"frozen_{prec}")
                del store
            del loaded; gc.collect(); torch.cuda.empty_cache()
            res.update({
                "seed": cfg.seed, "precision": prec, "fit_mode": mode,
                "encoder_seconds": ref_seconds, "extraction_seconds": extract_seconds,
                "quantize_seconds": quant_s, "save_seconds": save_s, "load_seconds": load_s,
                "storage_mb": storage_mb, "embedding_rel_rmse": rel_rmse,
                "reference_e2e_seconds": ref_seconds, "reference_e2e_pr_auc": graph_res["pr_auc"],
                "upfront_seconds": ref_seconds + extract_seconds + quant_s + save_s,
                "per_workload_seconds": load_s + res["training_seconds"],
            })
            rows.append(res)
            print(f"[seed {cfg.seed}] {prec:5s} {mode:10s} PR-AUC={res['pr_auc']:.4f} fit={res['training_seconds']:.2f}s "
                  f"load={load_s:.2f}s store={storage_mb:.0f}MB relRMSE={rel_rmse:.2e}")
        # incremental save so a crash keeps finished precisions
        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / f"dgraphfin_quant_seed_{cfg.seed}.csv", index=False)
    print("saved", out_dir / f"dgraphfin_quant_seed_{cfg.seed}.csv")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--graph-batch-size", type=int, default=4096)
    p.add_argument("--inference-batch-size", type=int, default=65536)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--fit-modes", nargs="+", default=list(FIT_MODES), choices=FIT_MODES)
    a = p.parse_args()
    cfg = Config(seed=a.seed, epochs=a.epochs, patience=a.patience, graph_batch_size=a.graph_batch_size,
                 inference_batch_size=a.inference_batch_size, workers=a.workers)
    data = load_data(Path("data/dgraphfin"))
    run_seed(data, cfg, a.precisions, a.fit_modes, Path("results/dgraphfin_quant"), Path("embeddings/dgraphfin_quant"))


if __name__ == "__main__":
    main()
