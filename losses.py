"""
Focal Loss for binary classification under severe class imbalance.

Lin et al. (2017) "Focal Loss for Dense Object Detection", ICCV.
https://arxiv.org/abs/1708.02002

Standard binary cross-entropy treats all examples equally. For ATAC-seq
prediction the dataset is ~85% closed chromatin, so a model that always
predicts "closed" achieves 85% accuracy trivially. Focal loss multiplies
the per-example loss by (1 - p_t)^gamma, where p_t is the model's
probability for the correct class. This down-weights confidently correct
predictions (easy negatives) and focuses learning on hard examples.
"""

import numpy as np


def focal_loss(logits: np.ndarray, targets: np.ndarray,
               gamma: float = 2.0, alpha: float = 0.25) -> tuple:
    """
    Binary focal loss.

    logits:  (batch,) raw pre-sigmoid scores
    targets: (batch,) binary labels in {0, 1}
    gamma:   focusing parameter (higher = more focus on hard examples)
    alpha:   class weight for positive class

    Returns (loss scalar, d_logits gradient)
    """
    probs = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
    probs = np.clip(probs, 1e-7, 1.0 - 1e-7)

    p_t = np.where(targets == 1, probs, 1.0 - probs)
    alpha_t = np.where(targets == 1, alpha, 1.0 - alpha)

    focal_weight = alpha_t * (1.0 - p_t) ** gamma
    bce = -np.log(p_t)
    loss = (focal_weight * bce).mean()

    d_p_t = -focal_weight / (p_t + 1e-9)
    d_p_t += focal_weight * bce * gamma / (1.0 - p_t + 1e-9)
    d_p_t /= len(logits)

    sign = np.where(targets == 1, 1.0, -1.0)
    d_probs = d_p_t * sign
    d_logits = (d_probs * probs * (1.0 - probs)).astype(np.float32)

    return loss, d_logits


def binary_cross_entropy(logits: np.ndarray, targets: np.ndarray) -> tuple:
    """Plain BCE for comparison / ablation."""
    probs = np.clip(1.0 / (1.0 + np.exp(-logits.astype(np.float64))), 1e-7, 1-1e-7)
    loss = -(targets * np.log(probs) + (1 - targets) * np.log(1 - probs)).mean()
    d_logits = ((probs - targets) / len(logits)).astype(np.float32)
    return loss, d_logits
