"""
Multi-head self-attention with Rotary Positional Embeddings.

All operations are explicit NumPy. Gradients are computed
analytically in the backward pass.
"""

import numpy as np
from models.embeddings import apply_rope


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x_shifted = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(x_shifted)
    return exp_x / (exp_x.sum(axis=axis, keepdims=True) + 1e-9)


class MultiHeadAttention:
    """
    Multi-head self-attention.

    Parameters store weights as (d_model, d_model) matrices. During the forward
    pass they are reshaped to (n_heads, d_model, head_dim) for parallel head
    computation.
    """

    def __init__(self, d_model: int, n_heads: int, rng: np.random.Generator):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        k = np.sqrt(1.0 / d_model)
        self.W_q = rng.uniform(-k, k, (d_model, d_model)).astype(np.float32)
        self.W_k = rng.uniform(-k, k, (d_model, d_model)).astype(np.float32)
        self.W_v = rng.uniform(-k, k, (d_model, d_model)).astype(np.float32)
        self.W_o = rng.uniform(-k, k, (d_model, d_model)).astype(np.float32)

        self.b_q = np.zeros(d_model, dtype=np.float32)
        self.b_k = np.zeros(d_model, dtype=np.float32)
        self.b_v = np.zeros(d_model, dtype=np.float32)
        self.b_o = np.zeros(d_model, dtype=np.float32)

        self._cache = {}

    def parameters(self):
        return {
            "W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o,
            "b_q": self.b_q, "b_k": self.b_k, "b_v": self.b_v, "b_o": self.b_o,
        }

    def forward(self, x: np.ndarray, rope_cache: np.ndarray,
                mask: np.ndarray = None) -> np.ndarray:
        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        Q = (x @ self.W_q + self.b_q).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        K = (x @ self.W_k + self.b_k).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)
        V = (x @ self.W_v + self.b_v).reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        Q = apply_rope(Q, rope_cache)
        K = apply_rope(K, rope_cache)

        attn_scores = (Q @ K.transpose(0, 1, 3, 2)) * self.scale

        if mask is not None:
            attn_scores = attn_scores + mask[np.newaxis, np.newaxis] * -1e9

        attn_weights = softmax(attn_scores, axis=-1)

        context = attn_weights @ V
        context = context.transpose(0, 2, 1, 3).reshape(B, T, D)
        out = context @ self.W_o + self.b_o

        self._cache = {
            "x": x, "Q": Q, "K": K, "V": V,
            "attn_weights": attn_weights, "context": context,
        }
        return out, attn_weights

    def backward(self, d_out: np.ndarray) -> tuple:
        x = self._cache["x"]
        Q = self._cache["Q"]
        K = self._cache["K"]
        V = self._cache["V"]
        attn_weights = self._cache["attn_weights"]
        context = self._cache["context"]

        B, T, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        d_context = d_out @ self.W_o.T
        dW_o = context.reshape(B * T, D).T @ d_out.reshape(B * T, D)
        db_o = d_out.sum(axis=(0, 1))

        d_context = d_context.reshape(B, T, H, Dh).transpose(0, 2, 1, 3)

        d_V = attn_weights.transpose(0, 1, 3, 2) @ d_context
        d_attn = d_context @ V.transpose(0, 1, 3, 2)

        d_attn_scores = attn_weights * (
            d_attn - (d_attn * attn_weights).sum(axis=-1, keepdims=True)
        ) * self.scale

        d_Q = d_attn_scores @ K
        d_K = d_attn_scores.transpose(0, 1, 3, 2) @ Q

        d_Q = d_Q.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_K = d_K.transpose(0, 2, 1, 3).reshape(B, T, D)
        d_V = d_V.transpose(0, 2, 1, 3).reshape(B, T, D)

        x_flat = x.reshape(B * T, D)
        dW_q = x_flat.T @ d_Q.reshape(B * T, D)
        dW_k = x_flat.T @ d_K.reshape(B * T, D)
        dW_v = x_flat.T @ d_V.reshape(B * T, D)
        db_q = d_Q.sum(axis=(0, 1))
        db_k = d_K.sum(axis=(0, 1))
        db_v = d_V.sum(axis=(0, 1))

        dx = d_Q @ self.W_q.T + d_K @ self.W_k.T + d_V @ self.W_v.T

        grads = {
            "W_q": dW_q, "W_k": dW_k, "W_v": dW_v, "W_o": dW_o,
            "b_q": db_q, "b_k": db_k, "b_v": db_v, "b_o": db_o,
        }
        return dx, grads
