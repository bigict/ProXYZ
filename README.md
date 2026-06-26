# ProXYZ

A protein language model pre-training framework built on DeepSeek-style architectures with support for advanced training techniques.

## Overview

ProXYZ is designed for pre-training protein language models using modern transformer architectures. It supports both standard Llama-style models with Grouped-Query Attention (GQA) and DeepSeek-V2's Multi-head Latent Attention (MLA) for improved memory efficiency.

Key features include:
- **Flexible Architecture**: Switch between Llama GQA and DeepSeek-V2 MLA with a single flag
- **Fill-in-the-Middle (FIM) Training**: DeepSeek-Coder style FIM training for bidirectional context understanding
- **Cluster-based Sampling**: Weighted sampling strategy to balance cluster diversity
- **HuggingFace Integration**: Load datasets from HuggingFace Hub or local files
- **Checkpoint Resumption**: Resume training from previous checkpoints with full state restoration
- **Separate Loss Tracking**: Monitor FIM and standard training losses independently

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd ProXYZ

# Install dependencies
pip install torch transformers datasets click

# Optional: Install flash attention for better performance
pip install flash-attn --no-build-isolation
```

## Quick Start

### Training

Basic training with local data:
```bash
bash train.sh protein_seqs.txt
```

Training with FASTA format:
```bash
bash train.sh seqs.fasta --data_format fasta
```

Training from HuggingFace dataset:
```bash
python src/proxyz/train.py \
  --dataset_name your-dataset/name \
  --dataset_split train \
  --text_column sequence \
  --tokenizer_file tokenizer.json
```

### Using MLA (Multi-head Latent Attention)

Switch to DeepSeek-V2 MLA for better memory efficiency:
```bash
bash train.sh protein_seqs.txt --use_mla
```

Customize MLA parameters:
```bash
bash train.sh protein_seqs.txt \
  --use_mla \
  --kv_lora_rank 256 \
  --qk_rope_head_dim 32
```

### Fill-in-the-Middle (FIM) Training

Enable FIM training for bidirectional context learning:
```bash
# 50% FIM, 50% standard training
bash train.sh protein_seqs.txt --fim_rate 0.5

# 100% FIM training (DeepSeek-Coder style)
bash train.sh protein_seqs.txt --fim_rate 1.0
```

### Cluster-based Sampling

Use cluster information for balanced sampling:
```bash
bash train.sh protein_seqs.txt \
  --cluster_files clusters.txt
```

Cluster file format (two columns: cluster_id, data_row_id):
```
cluster_001 0
cluster_001 1
cluster_002 2
cluster_001 3
```

Sampling weight formula: `n / (1 + log(n))` where n is cluster size.

### Validation

Enable validation during training:
```bash
bash train.sh train.txt \
  --eval_files val.txt \
  --eval_strategy steps \
  --eval_steps 500
```

### Resume from Checkpoint

Continue training from a previous checkpoint:
```bash
bash train.sh protein_seqs.txt \
  --resume_from_checkpoint
```

This automatically loads the latest checkpoint from the output directory, restoring:
- Model weights
- Optimizer state
- Learning rate scheduler
- Training step counter

### Sequence Generation

Generate protein sequences from a trained model:
```bash
bash generate.sh
```

Generate with custom parameters:
```bash
bash generate.sh \
  --num_sequences 50 \
  --num_tokens 512 \
  --temperature 0.8 \
  --top_p 0.9
```

Conditional generation with a prefix:
```bash
bash generate.sh --prompt MVSKGE --num_tokens 256
```

FIM infilling (generate middle portion):
```bash
bash generate.sh \
  --fim_prefix "MVSKGE" \
  --fim_suffix "LKTIKQ" \
  --num_tokens 100
```

## Training Options

### Model Architecture
- `--model_hidden_size`: Model width (default: 2048)
- `--model_intermediate_size`: SwiGLU hidden dimension (default: 5632)
- `--model_num_hidden_layers`: Model depth (default: 24)
- `--model_num_attention_heads`: Attention heads (default: 16)
- `--model_num_key_value_heads`: GQA KV heads (default: 4)
- `--use_mla`: Enable DeepSeek-V2 MLA architecture
- `--max_position_embeddings`: Context window length (default: 4096)

### MLA-specific Parameters
- `--kv_lora_rank`: KV low-rank compression rank (default: 512)
- `--q_lora_rank`: Q low-rank compression rank (default: 0, disabled)
- `--qk_nope_head_dim`: Non-RoPE Q/K head dimension (default: 128)
- `--qk_rope_head_dim`: RoPE Q/K head dimension (default: 64)
- `--v_head_dim`: V head dimension (default: 128)

### Training Hyperparameters
- `--learning_rate`: Peak learning rate (default: 3e-4)
- `--weight_decay`: Weight decay (default: 0.1)
- `--num_train_epochs`: Training epochs (default: 3.0)
- `--max_steps`: Fixed step count (overrides epochs if > 0)
- `--per_device_train_batch_size`: Per-device batch size (default: 4)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 8)

### FIM Training
- `--fim_rate`: FIM transformation probability (0.0-1.0, default: 0.0)
- `--fim_spm_rate`: SPM format fraction among FIM examples (default: 0.5)

### Data Loading
- `--data_format`: Input format - "line" or "fasta" (default: "line")
- `--dataset_name`: HuggingFace dataset name
- `--dataset_config`: Dataset config/subset
- `--dataset_split`: Training split (default: "train")
- `--text_column`: Column name for sequence text (default: "text")
- `--cluster_files`: Clustering files for weighted sampling

### Validation
- `--eval_files`: Validation data files
- `--eval_strategy`: When to validate - "steps", "epoch", or "no" (default: "steps")
- `--eval_steps`: Validation frequency in steps (default: 500)

### Sequence Processing
- `--max_token_length`: Random crop threshold for long sequences
- `--tokenizer_file`: Path to tokenizer JSON file

### Logging & Checkpointing
- `--output_dir`: Checkpoint and model output directory
- `--logging_steps`: Log every N steps (default: 10)
- `--save_steps`: Checkpoint every N steps (default: 500)
- `--report_to`: Logging integrations (default: "swanlab,tensorboard")
- `--run_name`: Run name for logging
- `--resume_from_checkpoint`: Resume from latest checkpoint

### Performance
- `--attn_implementation`: Attention backend - "flash_attention_2", "sdpa", or "eager" (default: "flash_attention_2")
- `--dataloader_num_workers`: DataLoader worker processes (default: 4)

## Project Structure

```
ProXYZ/
├── src/proxyz/
│   ├── train.py          # Main training script
│   ├── generate.py       # Sequence generation
│   ├── evaluate.py       # Model evaluation
│   ├── utils.py          # Utility functions
│   └── data/             # Dataset utilities
├── train.sh              # Training wrapper script
├── generate.sh           # Generation wrapper script
└── README.md
```

## Architecture Details

### Llama with GQA
Standard transformer with Grouped-Query Attention for efficient inference. Uses:
- SwiGLU activation
- RMSNorm
- RoPE positional embeddings
- Separate input/output embeddings

### DeepSeek-V2 with MLA
Multi-head Latent Attention provides better memory efficiency through:
- Low-rank KV compression
- Decoupled RoPE and non-RoPE dimensions
- Reduced KV cache size compared to MHA/GQA

### FIM Training
Fill-in-the-Middle training teaches the model to predict missing content given surrounding context:
- **SPM format**: `<BOS><fim_suffix><suffix><fim_prefix><prefix><fim_middle><middle><EOS>`
- **PSM format**: `<BOS><fim_prefix><prefix><fim_suffix><suffix><fim_middle><middle><EOS>`
- Only the middle portion contributes to loss (prefix/suffix masked with -100)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- transformers 4.40+
- datasets
- click
- flash-attn (optional, recommended for performance)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Citation

If you use ProXYZ in your research, please cite:

```bibtex
@software{proxyz2026,
  title = {ProXYZ: Protein Language Model Pre-training Framework},
  author = {bigict},
  year = {2026},
  url = {<repository-url>}
}
```

## Acknowledgments

This project builds upon:
- [DeepSeek-V2](https://github.com/deepseek-ai/DeepSeek-V2) for MLA architecture
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder) for FIM training methodology
- [HuggingFace Transformers](https://github.com/huggingface/transformers) for the training framework
