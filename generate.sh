#!/usr/bin/env bash
# Generate protein sequences from a trained ProXYZ model.
#
# Usage:
#   bash generate.sh [additional options]
#
# Examples:
#   # Generate 10 sequences of 256 tokens each (defaults)
#   bash generate.sh
#
#   # Generate 50 sequences of 512 tokens from the latest checkpoint
#   bash generate.sh --num_sequences 50 --num_tokens 512
#
#   # Seed from a specific prefix with lower temperature
#   bash generate.sh --prompt MVSK --temperature 0.8 --top_p 0.9
#
# Output: a timestamped FASTA file in ./generated_sequences/

PYTHONPATH=src python src/proxyz/generate.py \
  "$@" \
  -v
