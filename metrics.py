"""
Evaluation metrics for binary chromatin accessibility prediction.

"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def compute_metrics(labels: np.ndarray, probs: np.ndarray,
                    threshold: float = 0.5) -> dict:
    preds = (probs >= threshold).astype(int)
    auroc = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    auprc = average_precision_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    f1 = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    acc = (preds == labels).mean()
    cm = confusion_matrix(labels, preds)

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": acc,
        "confusion_matrix": cm,
    }


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    thresholds = np.linspace(0.01, 0.99, 200)
    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t
