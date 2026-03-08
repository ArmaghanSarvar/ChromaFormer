"""
ChromaFormer evaluation script.

Loads a saved checkpoint, runs inference on the test set, computes metrics,
and generates attention visualizations for a sample of sequences.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from models.tokenizer import KmerTokenizer
from models.transformer import ChromaFormer
from utils.metrics import compute_metrics, find_optimal_threshold
from utils.viz import plot_roc_prc, plot_attention_map


def evaluate(args):
    os.makedirs(args.results_dir, exist_ok=True)

    test_df = pd.read_csv(os.path.join(args.data_dir, "test.csv"))
    print(f"Test set: {len(test_df)} sequences ({int(test_df['label'].sum())} open)")

    tokenizer = KmerTokenizer(k=args.kmer, stride=1)
    seq_len = len(test_df["sequence"].iloc[0])
    max_seq_len = seq_len - args.kmer + 2 + 5

    model = ChromaFormer(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        max_seq_len=max_seq_len,
    )
    model.load(args.checkpoint)
    print(f"Loaded checkpoint: {args.checkpoint}")

    all_probs = []
    all_labels = test_df["label"].values.astype(np.float32)
    all_attn = []

    for start in range(0, len(test_df), args.batch_size):
        batch = test_df.iloc[start : start + args.batch_size]
        token_ids = tokenizer.batch_tokenize(batch["sequence"].tolist(), pad=True)
        logits, attn_weights = model.forward(token_ids)
        probs = 1.0 / (1.0 + np.exp(-logits))
        all_probs.extend(probs.tolist())
        if start == 0:
            all_attn = attn_weights

    all_probs = np.array(all_probs)

    threshold = find_optimal_threshold(all_labels, all_probs)
    metrics = compute_metrics(all_labels, all_probs, threshold=threshold)

    print("\n=== Test Set Metrics ===")
    print(f"  AUROC:     {metrics['auroc']:.4f}")
    print(f"  AUPRC:     {metrics['auprc']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Threshold: {threshold:.3f}")
    print(f"  Confusion matrix:\n{metrics['confusion_matrix']}")

    plot_roc_prc(
        all_labels, all_probs,
        os.path.join(args.results_dir, "roc_prc.png"),
    )

    open_seqs = test_df[test_df["label"] == 1].head(args.n_attn_viz)
    for idx, (_, row) in enumerate(open_seqs.iterrows()):
        seq = row["sequence"]
        token_ids = tokenizer.batch_tokenize([seq], pad=True)
        _, attn_weights_single = model.forward(token_ids)

        tokens = tokenizer.decode(token_ids[0].tolist())

        for layer_idx, layer_attn in enumerate(attn_weights_single):
            out_path = os.path.join(
                args.results_dir,
                f"attn_seq{idx}_layer{layer_idx}.png",
            )
            plot_attention_map(
                attn_weights=layer_attn[0],
                sequence=seq,
                tokens=tokens,
                layer=layer_idx,
                output_path=out_path,
            )

    print(f"\nAll outputs saved to {args.results_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ChromaFormer")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--checkpoint", type=str, default="results/best_model.npz")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--kmer", type=int, default=6)
    parser.add_argument("--n_attn_viz", type=int, default=3,
                        help="Number of open-chromatin sequences to visualize attention for")
    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
