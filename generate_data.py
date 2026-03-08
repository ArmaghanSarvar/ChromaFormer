"""
Synthetic ATAC-seq data generator

Generates DNA sequences with realistic chromatin accessibility labels
Open chromatin sequences are seeded with known regulatory motifs (CTCF, AP-1, SP1,
TATA-box) and closed sequences are sampled from GC-matched random background.

ENCODE data can be substituted via scripts/download_encode.sh
"""

import argparse
import os
import random
import numpy as np
import pandas as pd


REGULATORY_MOTIFS = {
    "CTCF":   ["CCGCGAGGNGGCAG", "CCGCGNGGNGGCAG", "CCACNAGGTGGCAG"],
    "AP1":    ["TGAGTCA", "TGACTCA", "TGAATCA"],
    "SP1":    ["GGGCGG", "CCGCCC", "GGGCGG"],
    "TATA":   ["TATAAA", "TATATA", "TATAAG"],
    "GATA":   ["TGATAG", "AGATAA", "TGATTA"],
}

BASES = ["A", "C", "G", "T"]

AMBIGUOUS = {
    "N": ["A", "C", "G", "T"],
    "S": ["C", "G"],
    "W": ["A", "T"],
    "R": ["A", "G"],
    "Y": ["C", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
}


def resolve_ambiguous(motif: str) -> str:
    resolved = []
    for ch in motif:
        if ch in AMBIGUOUS:
            resolved.append(random.choice(AMBIGUOUS[ch]))
        else:
            resolved.append(ch)
    return "".join(resolved)


def random_sequence(length: int, gc_content: float = 0.5) -> str:
    at = 1.0 - gc_content
    weights = [at / 2, gc_content / 2, gc_content / 2, at / 2]
    return "".join(random.choices(BASES, weights=weights, k=length))


def embed_motifs(seq: str, n_motifs: int = 2) -> str:
    seq = list(seq)
    motif_families = random.sample(list(REGULATORY_MOTIFS.keys()), min(n_motifs, len(REGULATORY_MOTIFS)))
    for family in motif_families:
        motif_raw = random.choice(REGULATORY_MOTIFS[family])
        motif = resolve_ambiguous(motif_raw)
        max_pos = len(seq) - len(motif)
        if max_pos <= 0:
            continue
        pos = random.randint(0, max_pos)
        seq[pos : pos + len(motif)] = list(motif)
    return "".join(seq)


def generate_open_sequence(seq_len: int) -> str:
    gc = random.uniform(0.38, 0.62)
    seq = random_sequence(seq_len, gc_content=gc)
    n_motifs = random.choices([1, 2, 3], weights=[0.3, 0.5, 0.2])[0]
    return embed_motifs(seq, n_motifs=n_motifs)


def generate_closed_sequence(seq_len: int) -> str:
    gc = random.uniform(0.35, 0.65)
    return random_sequence(seq_len, gc_content=gc)


def generate_dataset(
    n_sequences: int,
    seq_len: int = 200,
    open_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    n_open = int(n_sequences * open_fraction)
    n_closed = n_sequences - n_open

    records = []
    for _ in range(n_open):
        records.append({"sequence": generate_open_sequence(seq_len), "label": 1})
    for _ in range(n_closed):
        records.append({"sequence": generate_closed_sequence(seq_len), "label": 0})

    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)

    seq_ids = [f"seq_{i:06d}" for i in range(len(df))]
    df.insert(0, "seq_id", seq_ids)
    return df


def split_dataset(df: pd.DataFrame, train=0.8, val=0.1, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    n = len(df)
    n_train = int(n * train)
    n_val = int(n * val)
    return (
        df.iloc[:n_train].reset_index(drop=True),
        df.iloc[n_train : n_train + n_val].reset_index(drop=True),
        df.iloc[n_train + n_val :].reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic ATAC-seq dataset")
    parser.add_argument("--n_sequences", type=int, default=5000)
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--open_fraction", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Generating {args.n_sequences} sequences (seq_len={args.seq_len}, "
          f"open_fraction={args.open_fraction})...")

    df = generate_dataset(
        n_sequences=args.n_sequences,
        seq_len=args.seq_len,
        open_fraction=args.open_fraction,
        seed=args.seed,
    )

    train_df, val_df, test_df = split_dataset(df, seed=args.seed)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(args.output_dir, f"{split_name}.csv")
        split_df.to_csv(path, index=False)
        n_open = split_df["label"].sum()
        print(f"  {split_name}: {len(split_df)} sequences, "
              f"{n_open} open ({100*n_open/len(split_df):.1f}%)")

    print(f"Saved splits to {args.output_dir}")


if __name__ == "__main__":
    main()
