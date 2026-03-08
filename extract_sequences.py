"""
Extract positive and negative sequences from ENCODE ATAC-seq data.

Requires ENCODE peak BED file (from download_encode.sh)

Positive sequences: ±100bp windows centered on peak summits (col 9 in BED)
Negative sequences: random non-peak windows with matched GC content (±20%)
"""

import argparse
import os
import random
import numpy as np
import pandas as pd


def read_fasta(path: str) -> dict:
    sequences = {}
    current_chrom = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if current_chrom is not None:
                    sequences[current_chrom] = "".join(current_seq).upper()
                current_chrom = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_chrom is not None:
        sequences[current_chrom] = "".join(current_seq).upper()
    return sequences


def gc_content(seq: str) -> float:
    seq = seq.upper()
    return (seq.count("G") + seq.count("C")) / max(len(seq), 1)


def has_only_bases(seq: str) -> bool:
    return all(b in "ACGT" for b in seq.upper())


def extract_sequences(peaks_path: str, fasta_path: str, chrom: str,
                      seq_len: int = 200, seed: int = 42) -> pd.DataFrame:
    random.seed(seed)
    np.random.seed(seed)

    print(f"Loading FASTA ({fasta_path})...")
    genome = read_fasta(fasta_path)
    if chrom not in genome:
        raise ValueError(f"Chromosome {chrom} not found. Available: {list(genome.keys())}")
    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)
    print(f"  {chrom}: {chrom_len:,} bp")

    print(f"Loading peaks ({peaks_path})...")
    peaks_df = pd.read_csv(peaks_path, sep="\t", header=None,
                           usecols=[0, 1, 2, 9],
                           names=["chrom", "start", "end", "summit_offset"])
    peaks_df = peaks_df[peaks_df["chrom"] == chrom].reset_index(drop=True)
    print(f"  {len(peaks_df)} peaks on {chrom}")

    half = seq_len // 2
    positives = []
    for _, row in peaks_df.iterrows():
        summit = row["start"] + int(row["summit_offset"])
        s = summit - half
        e = summit + half
        if s < 0 or e > chrom_len:
            continue
        seq = chrom_seq[s:e]
        if has_only_bases(seq):
            positives.append(seq)

    print(f"  Extracted {len(positives)} valid positive sequences")

    peak_intervals = set()
    for _, row in peaks_df.iterrows():
        for pos in range(row["start"], row["end"]):
            peak_intervals.add(pos)

    n_neg = len(positives)
    pos_gcs = [gc_content(s) for s in positives]
    neg_gc_mean = np.mean(pos_gcs)
    neg_gc_std = np.std(pos_gcs) + 0.02

    negatives = []
    attempts = 0
    while len(negatives) < n_neg and attempts < n_neg * 50:
        attempts += 1
        start = random.randint(0, chrom_len - seq_len)
        end = start + seq_len
        if any(p in peak_intervals for p in range(start, end, 10)):
            continue
        seq = chrom_seq[start:end]
        if not has_only_bases(seq):
            continue
        gc = gc_content(seq)
        if abs(gc - neg_gc_mean) > neg_gc_std * 2:
            continue
        negatives.append(seq)

    print(f"  Sampled {len(negatives)} negative sequences")

    records = (
        [{"sequence": s, "label": 1} for s in positives] +
        [{"sequence": s, "label": 0} for s in negatives]
    )
    df = pd.DataFrame(records).sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "seq_id", [f"seq_{i:06d}" for i in range(len(df))])
    return df


def main():
    parser = argparse.ArgumentParser(description="Extract sequences from ENCODE data")
    parser.add_argument("--peaks", type=str, required=True)
    parser.add_argument("--fasta", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="data/")
    parser.add_argument("--chrom", type=str, default="chr21")
    parser.add_argument("--seq_len", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = extract_sequences(args.peaks, args.fasta, args.chrom,
                           seq_len=args.seq_len, seed=args.seed)

    n = len(df)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train : n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        path = os.path.join(args.output_dir, f"{split_name}.csv")
        split_df.to_csv(path, index=False)
        print(f"  {split_name}: {len(split_df)} sequences saved to {path}")


if __name__ == "__main__":
    main()
