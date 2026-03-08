"""
K-mer tokenizer for DNA sequences.

Decomposes a DNA sequence into overlapping k-mers and maps each to an integer
token ID. The vocabulary size is 4^k (plus special tokens). This mirrors
subword tokenization in NLP but is biologically motivated: 6-mers capture
"""

import itertools
from typing import List, Dict
import numpy as np


BASES = ["A", "C", "G", "T"]
PAD_TOKEN = "<PAD>"
CLS_TOKEN = "<CLS>"
UNK_TOKEN = "<UNK>"


def build_kmer_vocab(k: int) -> Dict[str, int]:
    special = [PAD_TOKEN, CLS_TOKEN, UNK_TOKEN]
    vocab = {tok: i for i, tok in enumerate(special)}
    for kmer in itertools.product(BASES, repeat=k):
        token = "".join(kmer)
        vocab[token] = len(vocab)
    return vocab


class KmerTokenizer:
    """
    Sliding-window k-mer tokenizer.

    A sequence of length L produces L - k + 1 overlapping tokens.
    A [CLS] token is prepended so the final CLS hidden state can be used
    for sequence-level classification (BERT-style).
    """

    def __init__(self, k: int = 6, stride: int = 1):
        self.k = k
        self.stride = stride
        self.vocab = build_kmer_vocab(k)
        self.inv_vocab = {v: kk for kk, v in self.vocab.items()}
        self.pad_id = self.vocab[PAD_TOKEN]
        self.cls_id = self.vocab[CLS_TOKEN]
        self.unk_id = self.vocab[UNK_TOKEN]
        self.vocab_size = len(self.vocab)

    def tokenize(self, sequence: str) -> List[int]:
        seq = sequence.upper()
        tokens = [self.cls_id]
        for i in range(0, len(seq) - self.k + 1, self.stride):
            kmer = seq[i : i + self.k]
            tokens.append(self.vocab.get(kmer, self.unk_id))
        return tokens

    def batch_tokenize(self, sequences: List[str], pad: bool = True) -> np.ndarray:
        tokenized = [self.tokenize(s) for s in sequences]
        if not pad:
            return tokenized
        max_len = max(len(t) for t in tokenized)
        padded = np.full((len(tokenized), max_len), self.pad_id, dtype=np.int32)
        for i, tokens in enumerate(tokenized):
            padded[i, : len(tokens)] = tokens
        return padded

    def decode(self, token_ids: List[int]) -> List[str]:
        return [self.inv_vocab.get(tid, UNK_TOKEN) for tid in token_ids]

    def __repr__(self):
        return (f"KmerTokenizer(k={self.k}, stride={self.stride}, "
                f"vocab_size={self.vocab_size})")
