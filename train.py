"""
ChromaFormer training script.

Trains the transformer on chromatin accessibility prediction with:
  - Mini-batch gradient descent (Adam optimizer)
  - Focal loss to handle class imbalance
  - Per-epoch AUROC tracking on validation set
  - Checkpoint saving of the best validation AUROC model
"""

import argparse
import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from models.tokenizer import KmerTokenizer
from models.transformer import ChromaFormer
from models.losses import focal_loss
from utils.metrics import compute_metrics
from utils.viz import plot_training_curves


class AdamOptimizer:
    """Adam optimizer with gradient clipping."""

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8,
                 weight_decay: float = 1e-4, clip_norm: float = 1.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.clip_norm = clip_norm
        self.t = 0
        self.m = {}
        self.v = {}

    def step(self, params: dict, grads: dict):
        self.t += 1
        total_norm = 0.0
        for key, g in grads.items():
            total_norm += (g ** 2).sum()
        total_norm = np.sqrt(total_norm)
        clip_coef = min(1.0, self.clip_norm / (total_norm + 1e-6))

        for key, param in params.items():
            if key not in grads:
                continue
            g = grads[key] * clip_coef
            if self.weight_decay > 0 and param.ndim >= 2:
                g = g + self.weight_decay * param
            if key not in self.m:
                self.m[key] = np.zeros_like(param)
                self.v[key] = np.zeros_like(param)
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * g ** 2
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)
            param -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


def flatten_grads(model_grads: dict) -> dict:
    flat = {
        "embedding": model_grads["embedding"],
        "W_cls": model_grads["W_cls"],
        "b_cls": model_grads["b_cls"],
    }
    for i, block_grads in enumerate(model_grads["blocks"]):
        for k, v in block_grads.items():
            flat[f"block{i}_{k}"] = v
    return flat


def flatten_params(model) -> dict:
    params = {
        "embedding": model.embedding,
        "W_cls": model.W_cls,
        "b_cls": model.b_cls,
    }
    for i, block in enumerate(model.blocks):
        for k, p in block.attn.parameters().items():
            params[f"block{i}_attn_{k}"] = p
        for k, p in block.ln1.parameters().items():
            params[f"block{i}_ln1_{k}"] = p
        for k, p in block.ffn.parameters().items():
            params[f"block{i}_ffn_{k}"] = p
        for k, p in block.ln2.parameters().items():
            params[f"block{i}_ln2_{k}"] = p
    return params


def run_epoch(model, tokenizer, df, batch_size, is_train, optimizer=None,
              gamma=2.0, alpha=0.25):
    indices = np.arange(len(df))
    if is_train:
        np.random.shuffle(indices)

    total_loss = 0.0
    all_probs = []
    all_labels = []

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start : start + batch_size]
        batch = df.iloc[batch_idx]

        token_ids = tokenizer.batch_tokenize(batch["sequence"].tolist(), pad=True)
        labels = batch["label"].values.astype(np.float32)

        logits, _ = model.forward(token_ids)
        loss, d_logits = focal_loss(logits, labels, gamma=gamma, alpha=alpha)

        if is_train and optimizer is not None:
            grads = model.backward(d_logits)
            flat_grads = flatten_grads(grads)
            flat_params = flatten_params(model)
            optimizer.step(flat_params, flat_grads)

        total_loss += loss * len(batch_idx)
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.tolist())

    avg_loss = total_loss / len(df)
    metrics = compute_metrics(np.array(all_labels), np.array(all_probs))
    return avg_loss, metrics


def train(args):
    os.makedirs(args.results_dir, exist_ok=True)

    train_df = pd.read_csv(os.path.join(args.data_dir, "train.csv"))
    val_df = pd.read_csv(os.path.join(args.data_dir, "val.csv"))

    print(f"Train: {len(train_df)} seqs  ({int(train_df['label'].sum())} open)")
    print(f"Val:   {len(val_df)} seqs  ({int(val_df['label'].sum())} open)")

    tokenizer = KmerTokenizer(k=args.kmer, stride=1)
    seq_len = len(train_df["sequence"].iloc[0])
    max_seq_len = seq_len - args.kmer + 2 + 5

    model = ChromaFormer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=max_seq_len,
        seed=args.seed,
    )
    print(f"\nModel parameters: {model.count_parameters():,}")
    print(f"Tokenizer: {tokenizer}\n")

    optimizer = AdamOptimizer(lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [], "val_loss": [],
        "train_auroc": [], "val_auroc": [],
    }
    best_val_auroc = 0.0
    checkpoint_path = os.path.join(args.results_dir, "best_model")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(
            model, tokenizer, train_df, args.batch_size,
            is_train=True, optimizer=optimizer,
            gamma=args.focal_gamma, alpha=args.focal_alpha,
        )
        val_loss, val_metrics = run_epoch(
            model, tokenizer, val_df, args.batch_size,
            is_train=False,
            gamma=args.focal_gamma, alpha=args.focal_alpha,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_auroc"].append(train_metrics["auroc"])
        history["val_auroc"].append(val_metrics["auroc"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss {train_loss:.4f}/{val_loss:.4f}  "
            f"auroc {train_metrics['auroc']:.4f}/{val_metrics['auroc']:.4f}  "
            f"({elapsed:.1f}s)"
        )

        if val_metrics["auroc"] > best_val_auroc:
            best_val_auroc = val_metrics["auroc"]
            model.save(checkpoint_path)
            print(f"  ✓ New best val AUROC: {best_val_auroc:.4f}")

    plot_training_curves(
        history,
        os.path.join(args.results_dir, "training_curves.png"),
    )
    print(f"\nBest val AUROC: {best_val_auroc:.4f}")
    print(f"Checkpoint saved to {checkpoint_path}.npz")


def main():
    parser = argparse.ArgumentParser(description="Train ChromaFormer")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--kmer", type=int, default=6)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    train(args)


if __name__ == "__main__":
    main()
