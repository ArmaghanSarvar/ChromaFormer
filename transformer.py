"""
ChromaFormer: Transformer encoder for chromatin accessibility prediction.

Architecture:
  Token embedding → RoPE → N × (MHA + LayerNorm + FFN + LayerNorm) → CLS pool → Linear

All forward and backward passes are implemented in NumPy with no autodiff.
"""

import numpy as np
from models.attention import MultiHeadAttention, softmax
from models.embeddings import build_rope_cache


class LayerNorm:
    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps
        self.gamma = np.ones(d_model, dtype=np.float32)
        self.beta = np.zeros(d_model, dtype=np.float32)
        self._cache = {}

    def parameters(self):
        return {"gamma": self.gamma, "beta": self.beta}

    def forward(self, x: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        out = self.gamma * x_norm + self.beta
        self._cache = {"x": x, "x_norm": x_norm, "mean": mean, "var": var}
        return out

    def backward(self, d_out: np.ndarray) -> tuple:
        x = self._cache["x"]
        x_norm = self._cache["x_norm"]
        var = self._cache["var"]
        N = x.shape[-1]

        d_gamma = (d_out * x_norm).sum(axis=(0, 1))
        d_beta = d_out.sum(axis=(0, 1))

        d_x_norm = d_out * self.gamma
        std_inv = 1.0 / np.sqrt(var + self.eps)
        dx = (
            d_x_norm
            - d_x_norm.mean(axis=-1, keepdims=True)
            - x_norm * (d_x_norm * x_norm).mean(axis=-1, keepdims=True)
        ) * std_inv

        return dx, {"gamma": d_gamma, "beta": d_beta}


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, rng: np.random.Generator):
        k1 = np.sqrt(2.0 / d_model)
        k2 = np.sqrt(2.0 / d_ff)
        self.W1 = rng.normal(0, k1, (d_model, d_ff)).astype(np.float32)
        self.b1 = np.zeros(d_ff, dtype=np.float32)
        self.W2 = rng.normal(0, k2, (d_ff, d_model)).astype(np.float32)
        self.b2 = np.zeros(d_model, dtype=np.float32)
        self._cache = {}

    def parameters(self):
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1 + self.b1
        h_relu = np.maximum(0, h)
        out = h_relu @ self.W2 + self.b2
        self._cache = {"x": x, "h": h, "h_relu": h_relu}
        return out

    def backward(self, d_out: np.ndarray) -> tuple:
        x = self._cache["x"]
        h = self._cache["h"]
        h_relu = self._cache["h_relu"]
        B, T, _ = x.shape

        dW2 = h_relu.reshape(B * T, -1).T @ d_out.reshape(B * T, -1)
        db2 = d_out.sum(axis=(0, 1))
        d_h_relu = d_out @ self.W2.T
        d_h = d_h_relu * (h > 0).astype(np.float32)
        dW1 = x.reshape(B * T, -1).T @ d_h.reshape(B * T, -1)
        db1 = d_h.sum(axis=(0, 1))
        dx = d_h @ self.W1.T

        return dx, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 rng: np.random.Generator):
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, rng)
        self.ln2 = LayerNorm(d_model)
        self._cache = {}

    def forward(self, x: np.ndarray, rope_cache: np.ndarray) -> tuple:
        attn_out, attn_weights = self.attn.forward(x, rope_cache)
        x = self.ln1.forward(x + attn_out)
        ffn_out = self.ffn.forward(x)
        x = self.ln2.forward(x + ffn_out)
        self._cache = {"pre_ln2": x}
        return x, attn_weights

    def backward(self, d_out: np.ndarray) -> tuple:
        pre_ln2 = self._cache["pre_ln2"]

        dx_ln2, grads_ln2 = self.ln2.backward(d_out)
        dx_ffn, grads_ffn = self.ffn.backward(dx_ln2)
        dx_res2 = dx_ln2 + dx_ffn
        dx_ln1, grads_ln1 = self.ln1.backward(dx_res2)
        dx_attn, grads_attn = self.attn.backward(dx_ln1)
        dx = dx_ln1 + dx_attn

        all_grads = {
            **{f"ln1_{k}": v for k, v in grads_ln1.items()},
            **{f"ffn_{k}": v for k, v in grads_ffn.items()},
            **{f"ln2_{k}": v for k, v in grads_ln2.items()},
            **{f"attn_{k}": v for k, v in grads_attn.items()},
        }
        return dx, all_grads


class ChromaFormer:
    """
    Full ChromaFormer model.

    Token IDs → embedding lookup → RoPE transformer blocks → CLS head → logit
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        max_seq_len: int = 210,
        seed: int = 42,
    ):
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_seq_len = max_seq_len

        rng = np.random.default_rng(seed)
        scale = np.sqrt(1.0 / d_model)
        self.embedding = rng.normal(0, scale, (vocab_size, d_model)).astype(np.float32)

        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff, rng) for _ in range(n_layers)
        ]

        self.W_cls = rng.normal(0, scale, (d_model, 1)).astype(np.float32)
        self.b_cls = np.zeros(1, dtype=np.float32)

        self.rope_cache = build_rope_cache(max_seq_len, d_model)
        self._cache = {}

    def forward(self, token_ids: np.ndarray) -> tuple:
        """
        token_ids: (batch, seq_len) int32
        returns: logits (batch,), list of attention weight arrays per layer
        """
        x = self.embedding[token_ids]

        all_attn_weights = []
        for block in self.blocks:
            x, attn_weights = block.forward(x, self.rope_cache)
            all_attn_weights.append(attn_weights)

        cls_hidden = x[:, 0, :]
        logits = (cls_hidden @ self.W_cls + self.b_cls).squeeze(-1)

        self._cache = {"token_ids": token_ids, "cls_hidden": cls_hidden}
        return logits, all_attn_weights

    def backward(self, d_logits: np.ndarray) -> dict:
        """
        d_logits: (batch,) gradient of loss w.r.t. logits
        """
        token_ids = self._cache["token_ids"]
        cls_hidden = self._cache["cls_hidden"]

        d_logits = d_logits[:, np.newaxis]
        dW_cls = cls_hidden.T @ d_logits
        db_cls = d_logits.sum(axis=0)
        d_cls_hidden = d_logits @ self.W_cls.T

        B, T = token_ids.shape
        d_x = np.zeros((B, T, self.d_model), dtype=np.float32)
        d_x[:, 0, :] = d_cls_hidden

        all_block_grads = []
        for block in reversed(self.blocks):
            d_x, block_grads = block.backward(d_x)
            all_block_grads.insert(0, block_grads)

        d_embedding = np.zeros_like(self.embedding)
        np.add.at(d_embedding, token_ids, d_x)

        grads = {
            "embedding": d_embedding,
            "W_cls": dW_cls,
            "b_cls": db_cls,
            "blocks": all_block_grads,
        }
        return grads

    def predict_proba(self, token_ids: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(token_ids)
        return 1.0 / (1.0 + np.exp(-logits))

    def count_parameters(self) -> int:
        total = self.embedding.size + self.W_cls.size + self.b_cls.size
        for block in self.blocks:
            for p in block.attn.parameters().values():
                total += p.size
            for p in block.ln1.parameters().values():
                total += p.size
            for p in block.ffn.parameters().values():
                total += p.size
            for p in block.ln2.parameters().values():
                total += p.size
        return total

    def save(self, path: str):
        arrays = {
            "embedding": self.embedding,
            "W_cls": self.W_cls,
            "b_cls": self.b_cls,
        }
        for i, block in enumerate(self.blocks):
            for k, v in block.attn.parameters().items():
                arrays[f"block{i}_attn_{k}"] = v
            for k, v in block.ln1.parameters().items():
                arrays[f"block{i}_ln1_{k}"] = v
            for k, v in block.ffn.parameters().items():
                arrays[f"block{i}_ffn_{k}"] = v
            for k, v in block.ln2.parameters().items():
                arrays[f"block{i}_ln2_{k}"] = v
        np.savez(path, **arrays)

    def load(self, path: str):
        data = np.load(path)
        self.embedding = data["embedding"]
        self.W_cls = data["W_cls"]
        self.b_cls = data["b_cls"]
        for i, block in enumerate(self.blocks):
            for k in block.attn.parameters():
                setattr(block.attn, k, data[f"block{i}_attn_{k}"])
            for k in block.ln1.parameters():
                setattr(block.ln1, k, data[f"block{i}_ln1_{k}"])
            for k in block.ffn.parameters():
                setattr(block.ffn, k, data[f"block{i}_ffn_{k}"])
            for k in block.ln2.parameters():
                setattr(block.ln2, k, data[f"block{i}_ln2_{k}"])
