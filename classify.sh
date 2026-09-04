#!/usr/bin/env bash
# Classify protein sequences attributes from a trained ProXYZ model.
#
# Usage:
#   bash classify.sh [additional options]
#
# Examples:
#   # Classify sequences from seqs.fasta with batch_size=8
#   bash classify.sh --data_files seqs.fasta --batch_size 8
#
# Output: a timestamped .pt file in ./classify_sequences/

PYTHONPATH=src python src/proxyz/classify.py \
  "$@" \
  -v
