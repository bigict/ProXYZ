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

# Setting TOKENIZERS_PARALLELISM=false forces the tokenizer to use a single thread per
# process, clearing the deadlock risk entirely.
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PYTHONPATH=src TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} python src/proxyz/classify.py \
  "$@" \
  -v
