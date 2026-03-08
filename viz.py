"""
  - Training/validation loss and AUROC curves
  - Attention heatmaps showing which k-mer positions the model attends to
  - ROC and Precision-Recall curves
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import os


plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})


def plot_training_curves(history: dict, output_path: str):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train", color="#4C72B0", lw=2)
    axes[0].plot(epochs, history["val_loss"], label="Val", color="#DD8452",
                 lw=2, linestyle="--")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Focal Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_auroc"], label="Train", color="#4C72B0", lw=2)
    axes[1].plot(epochs, history["val_auroc"], label="Val", color="#DD8452",
                 lw=2, linestyle="--")
    axes[1].axhline(0.5, color="gray", lw=1, linestyle=":")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("AUROC")
    axes[1].set_ylim(0.4, 1.0)
    axes[1].legend()

    plt.suptitle("ChromaFormer Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves → {output_path}")


def plot_roc_prc(labels: np.ndarray, probs: np.ndarray, output_path: str):
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fpr, tpr, _ = roc_curve(labels, probs)
    prec, rec, _ = precision_recall_curve(labels, probs)
    auroc = auc(fpr, tpr)
    auprc = auc(rec, prec)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(fpr, tpr, color="#4C72B0", lw=2, label=f"AUROC = {auroc:.3f}")
    axes[0].plot([0, 1], [0, 1], color="gray", lw=1, linestyle=":")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    baseline = labels.mean()
    axes[1].plot(rec, prec, color="#DD8452", lw=2, label=f"AUPRC = {auprc:.3f}")
    axes[1].axhline(baseline, color="gray", lw=1, linestyle=":",
                    label=f"Random = {baseline:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="upper right")

    plt.suptitle("ChromaFormer — Test Set Performance", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved ROC/PRC curves → {output_path}")


def plot_attention_map(attn_weights: np.ndarray, sequence: str,
                       tokens: list, layer: int, output_path: str,
                       max_tokens: int = 40):
    """
    Visualize average attention from [CLS] token across heads for one sequence.

    attn_weights: (heads, seq_len, seq_len)
    """
    cls_attn = attn_weights[:, 0, 1:]
    avg_attn = cls_attn.mean(axis=0)[:max_tokens]
    token_labels = [t[:4] for t in tokens[1:max_tokens + 1]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 6),
                             gridspec_kw={"height_ratios": [3, 1]})

    im = axes[0].imshow(attn_weights[:, 0:1, 1:max_tokens + 1].squeeze(1),
                        aspect="auto", cmap="Blues", vmin=0)
    axes[0].set_xticks(range(len(token_labels)))
    axes[0].set_xticklabels(token_labels, rotation=90, fontsize=7)
    axes[0].set_yticks(range(attn_weights.shape[0]))
    axes[0].set_yticklabels([f"Head {i}" for i in range(attn_weights.shape[0])])
    axes[0].set_title(f"Layer {layer + 1} — Attention from [CLS] per Head", fontsize=11)
    plt.colorbar(im, ax=axes[0], fraction=0.02)

    axes[1].bar(range(len(avg_attn)), avg_attn, color="#4C72B0", alpha=0.8)
    axes[1].set_xticks(range(len(token_labels)))
    axes[1].set_xticklabels(token_labels, rotation=90, fontsize=7)
    axes[1].set_ylabel("Avg attn")
    axes[1].set_title("Mean Attention Across Heads", fontsize=10)

    plt.suptitle(
        f"Attention Map — {sequence[:30]}{'...' if len(sequence) > 30 else ''}",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved attention map → {output_path}")
