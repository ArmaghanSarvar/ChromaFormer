#!/usr/bin/env bash
#   - ATAC-seq narrow peak BED file (ENCFF496BTM) — 200k+ peak calls
# Cell line: K562
# Assay: https://www.encodeproject.org ATAC-seq (ENCSR868FGK)
#   bash scripts/download_encode.sh
#   python scripts/extract_sequences.py

set -euo pipefail

DATA_DIR="data/encode"
mkdir -p "$DATA_DIR"

echo "=== Downloading ENCODE ATAC-seq peaks for K562 (hg38) ==="
PEAKS_URL="https://www.encodeproject.org/files/ENCFF496BTM/@@download/ENCFF496BTM.bed.gz"
wget -q --show-progress -O "$DATA_DIR/K562_ATACseq_peaks.bed.gz" "$PEAKS_URL"
gunzip -f "$DATA_DIR/K562_ATACseq_peaks.bed.gz"
echo "Peaks saved to $DATA_DIR/K562_ATACseq_peaks.bed"

echo ""
echo "=== Downloading hg38 chr21 FASTA (lightweight) ==="
CHR21_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg38/chromosomes/chr21.fa.gz"
wget -q --show-progress -O "$DATA_DIR/chr21.fa.gz" "$CHR21_URL"
gunzip -f "$DATA_DIR/chr21.fa.gz"
echo "FASTA saved to $DATA_DIR/chr21.fa"

echo ""
echo "=== Done. Now run: ==="
echo "  python scripts/extract_sequences.py --peaks $DATA_DIR/K562_ATACseq_peaks.bed \\"
echo "      --fasta $DATA_DIR/chr21.fa --output_dir data/ --chrom chr21"
