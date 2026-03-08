"""

Implements Su et al. (2021) "RoFormer: Enhanced Transformer with Rotary
Position Embedding". RoPE encodes position by rotating query and key vectors
in 2D subspaces, naturally encoding *relative* position in the attention dot
product. This is appropriate for genomic sequences than absolute learned
embeddings because regulatory elements have positional preferences that are
relative (e.g., a motif near another motif) rather than absolute.

Reference: https://arxiv.org/abs/2104.09864
"""

import numpy as np


def build_rope_cache(seq_len: int, dim: int, base: float = 10000.0) -> np.ndarray:
    """
    Precompute cos/sin rotation matrices for positions 0..seq_len-1.

    Returns shape: (seq_len, dim) where the last axis stores interleaved
    [cos_0, sin_0, cos_1, sin_1, ...] values.
    """
    assert dim % 2 == 0, "RoPE requires even embedding dimension"
    half = dim // 2
    theta = 1.0 / (base ** (np.arange(0, half, dtype=np.float64) / half))
    positions = np.arange(seq_len, dtype=np.float64)
    angles = np.outer(positions, theta)
    cos = np.cos(angles)
    sin = np.sin(angles)
    cache = np.stack([cos, sin], axis=-1).reshape(seq_len, dim)
    return cache.astype(np.float32)


def rotate_half(x: np.ndarray) -> np.ndarray:
    """Rotate the second half of the last dimension over to the first half."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return np.concatenate([-x2, x1], axis=-1)


def apply_rope(x: np.ndarray, rope_cache: np.ndarray) -> np.ndarray:
    """
    Apply rotary embeddings to queries or keys.

    x:          (batch, heads, seq_len, head_dim)
    rope_cache: (seq_len, head_dim)  — cos/sin interleaved
    """
    seq_len = x.shape[2]
    head_dim = x.shape[3]
    cos = rope_cache[:seq_len, : head_dim // 2]
    sin = rope_cache[:seq_len, : head_dim // 2]
    cos = np.concatenate([cos, cos], axis=-1)[np.newaxis, np.newaxis]
    sin = np.concatenate([sin, sin], axis=-1)[np.newaxis, np.newaxis]
    return x * cos + rotate_half(x) * sin
