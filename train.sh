#!/usr/bin/env bash
# Example training script for ProXYZ protein language model
#
# Usage:
#   bash train.sh <data_file> [additional options]
#
# Example with line-based format:
#   bash train.sh protein_seqs.txt
#
# Example with FASTA format:
#   bash train.sh seqs.fasta --data_format fasta
#
# Use '-' to read from stdin:
#   cat seqs.txt | bash train.sh - --max_steps 1000

# Setting TOKENIZERS_PARALLELISM=false forces the tokenizer to use a single thread per
# process, clearing the deadlock risk entirely.
TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

PYTHONPATH=src TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM} python src/proxyz/train.py \
  --tokenizer_file uniref90_30000.json \
  "$@" \
  -v
