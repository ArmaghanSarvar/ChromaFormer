# ChromaFormer 

**Predicting Chromatin Accessibility from DNA Sequence with a From-Scratch Transformer**

ChromaFormer is a lightweight transformer model built from scratch in NumPy that learns to predict whether a DNA sequence region is in "open" (accessible) chromatin state — a key indicator of active gene regulation. It is trained on synthetic ENCODE-style ATAC-seq data that faithfully replicates known biological motif distributions.

---

## Why This Problem Matters

Chromatin accessibility determines which genomic regions are available for transcription factor binding and gene activation. ATAC-seq (Assay for Transposase-Accessible Chromatin) is the gold-standard experimental method for measuring this. Training a model to predict accessibility purely from sequence is valuable for:

- Annotating genomes of organisms without ATAC-seq data
- Predicting the regulatory impact of genetic variants (eQTLs, GWAS hits)
- Discovering novel regulatory elements

---

## Model Architecture

```
DNA Sequence (200 bp)
      │
      ▼
 K-mer Tokenizer (k=6, stride=1)
      │  ~195 tokens per sequence
      ▼
 Learned Embedding (dim=64)
      │
      ▼
 Rotary Positional Embedding (RoPE)
      │
      ▼
 ┌─────────────────────────────┐
 │  Transformer Encoder Block  │  × 2 layers
 │  ┌───────────────────────┐  │
 │  │  Multi-Head Attention │  │  4 heads
 │  │  (with RoPE keys/     │  │
 │  │   queries)            │  │
 │  └───────────────────────┘  │
 │  ┌───────────────────────┐  │
 │  │  Feed-Forward Network │  │
 │  │  (ReLU, dim=128)      │  │
 │  └───────────────────────┘  │
 └─────────────────────────────┘
      │
      ▼
 [CLS] token pooling
      │
      ▼
 Linear → Sigmoid → P(open chromatin)
```

**design choices:**
- **K-mer tokenization** — treats overlapping 6-mers as vocabulary tokens (4^6 = 4096 possible tokens), analogous to subword tokenization in NLP but grounded in known biological sequence motifs
- **Rotary Positional Embeddings (RoPE)** — encodes relative positions in attention, more robust than learned absolute positional embeddings for variable-length sequences
- **Focal Loss** — addresses severe class imbalance (~15% open chromatin in real ATAC-seq data) by down-weighting easy negatives
- **From-scratch NumPy implementation** — no deep learning framework; every forward pass, backward pass, and optimizer step is explicit

---

## Data

The project ships with a synthetic data generator (`data/generate_data.py`) that mimics real ENCODE ATAC-seq patterns:

- **Open chromatin regions** are seeded with real regulatory motifs: CTCF (`CCGCGNGGNGGCAG`), AP-1 (`TGASTCA`), SP1 (`GGGCGG`), and TATA-box (`TATAAA`)
- **Closed regions** are random genomic background sequences
- Class ratio mirrors real ATAC-seq (≈15% open)

To use real ENCODE data instead, see `scripts/download_encode.sh`.

---

## Project Structure

```
chromaformer/
├── README.md
├── requirements.txt
├── data/
│   └── generate_data.py       # Synthetic ATAC-seq data generator
├── models/
│   ├── tokenizer.py           # K-mer DNA tokenizer
│   ├── embeddings.py          # RoPE positional embeddings
│   ├── attention.py           # Multi-head self-attention
│   ├── transformer.py         # Full transformer encoder + classifier
│   └── losses.py              # Focal loss implementation
├── utils/
│   ├── metrics.py             # AUROC, AUPRC, F1
│   └── viz.py                 # Attention map & training curve plots
├── scripts/
│   └── download_encode.sh     # Script to fetch real ENCODE ATAC-seq data
├── train.py                   # Training entrypoint
└── evaluate.py                # Evaluation & visualization entrypoint
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/chromaformer
cd chromaformer
pip install -r requirements.txt

# Generate synthetic training data
python data/generate_data.py --n_sequences 5000 --output_dir data/

# Train the model
python train.py --data_dir data/ --epochs 20 --lr 0.001 --batch_size 32

# Evaluate and visualize attention maps
python evaluate.py --data_dir data/ --checkpoint results/best_model.npz
```

---

## Using ENCODE Data

```bash
bash scripts/download_encode.sh
```

This downloads ATAC-seq peak BED files and genome FASTA for K562 cells from the ENCODE portal (https://www.encodeproject.org). Positive sequences are extracted from peak summits ±100bp; negatives are sampled from non-peak regions with matched GC content.

---

## Requirements

```
numpy>=1.24
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.6
seaborn>=0.12
tqdm>=4.64
```

No PyTorch or TensorFlow required. The entire model runs on NumPy.

---

## Citation & Data Sources

- ENCODE Project: https://www.encodeproject.org
- ATAC-seq methodology: Buenrostro et al., *Nature Methods* 2013
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", 2021
- Focal Loss: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
