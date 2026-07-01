import os
import random
import functools
import math
from collections import defaultdict

import click
import torch
from torch.utils.data import WeightedRandomSampler
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset

from proxyz.utils import dict2object
from proxyz.data import dataset


@click.command(context_settings={'show_default': True})
@click.argument("data_files", type=click.Path(), nargs=-1)
@click.option(
    "--eval_files", type=click.Path(), multiple=True, help="evaluate data files"
)
@click.option(
    "--dataset_name",
    type=str,
    default=None,
    help="HuggingFace dataset name (e.g. 'HuggingFaceH4/gsm8k'). "
    "If provided, loads from HuggingFace instead of DATA_FILES.",
)
@click.option(
    "--dataset_config",
    type=str,
    default=None,
    help="HuggingFace dataset config/subset (e.g. 'main').",
)
@click.option(
    "--dataset_split",
    type=str,
    default="train",
    help="Split to use for training (default: 'train').",
)
@click.option(
    "--dataset_eval_split",
    type=str,
    default=None,
    help="Split to use for validation (e.g. 'validation', 'test'). "
    "If not set, no eval dataset is loaded from HuggingFace.",
)
@click.option(
    "--text_column",
    type=str,
    default="text",
    help="Column name containing the sequence text (default: 'text').",
)
@click.option(
    "--tokenizer_file",
    type=click.Path(),
    default="my_tokenizer.json",
    help="Path to the tokenizer json file.",
)
@click.option(
    "--data_format",
    type=click.Choice(["line", "fasta"]),
    default="line",
    help="Input format: one sequence per line ('line') or FASTA ('fasta').",
)
@click.option("--model_hidden_size", type=int, default=2048, help="Model width.")
@click.option(
    "--model_intermediate_size",
    type=int,
    default=5632,
    help="Model SwiGLU hidden dimension (usually ~8/3 of hidden_size).",
)
@click.option("--model_num_hidden_layers", type=int, default=24, help="Model depth.")
@click.option(
    "--model_num_attention_heads", type=int, default=16, help="Model attention heads."
)
@click.option(
    "--model_num_key_value_heads",
    type=int,
    default=4,
    help="Model Grouped-Query Attention (GQA) for speed.",
)
@click.option(
    "--use_mla",
    is_flag=True,
    help="Use Multi-head Latent Attention (MLA) from DeepSeek-V2 instead of standard Llama attention.",
)
@click.option(
    "--kv_lora_rank",
    type=int,
    default=512,
    help="MLA: Rank for KV low-rank compression.",
)
@click.option(
    "--q_lora_rank",
    type=int,
    default=0,
    help="MLA: Rank for Q low-rank compression (0 to disable).",
)
@click.option(
    "--qk_nope_head_dim",
    type=int,
    default=128,
    help="MLA: Dimension of non-RoPE part in Q/K heads.",
)
@click.option(
    "--qk_rope_head_dim",
    type=int,
    default=64,
    help="MLA: Dimension of RoPE part in Q/K heads.",
)
@click.option(
    "--v_head_dim",
    type=int,
    default=128,
    help="MLA: Dimension of V heads.",
)
@click.option(
    "--max_position_embeddings", type=int, default=4096, help="Context window length."
)
@click.option(
    "--attn_implementation",
    type=click.Choice(["flash_attention_2", "sdpa", "eager"]),
    default="flash_attention_2",
    help="Attention backend. flash_attention_2 is fastest on Ampere/Ada+ GPUs.",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Where checkpoints and the final model are saved.",
)
@click.option(
    "--per_device_train_batch_size", type=int, default=4, help="Per-device batch size."
)
@click.option(
    "--gradient_accumulation_steps", type=int, default=8, help="Grad accumulation steps."
)
@click.option("--learning_rate", type=float, default=3e-4, help="Peak learning rate.")
@click.option("--weight_decay", type=float, default=0.1, help="Weight decay.")
@click.option("--num_train_epochs", type=float, default=3.0, help="Training epochs.")
@click.option(
    "--max_steps",
    type=int,
    default=-1,
    help="If > 0, overrides num_train_epochs with a fixed step count.",
)
@click.option("--logging_steps", type=int, default=10, help="Log every N steps.")
@click.option("--save_steps", type=int, default=500, help="Checkpoint every N steps.")
@click.option(
    "--max_token_length",
    type=int,
    default=None,
    help="If set, randomly crop sequences longer than this to a subsequence of this length. "
    "Useful for controlling memory usage with variable-length inputs.",
)
@click.option(
    "--fim_rate",
    type=float,
    default=0.0,
    help="Probability of applying Fill-in-the-Middle (FIM) transformation to each sequence. "
    "0.0 disables FIM, 1.0 applies FIM to all sequences (DeepSeek-Coder style).",
)
@click.option(
    "--fim_spm_rate",
    type=float,
    default=0.5,
    help="Among FIM examples, fraction using SPM format (suffix-prefix-middle). "
    "Remaining use PSM format (prefix-suffix-middle).",
)
@click.option(
    "--eval_strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="When to run validation: 'steps' (every eval_steps), 'epoch' (end of each epoch), or 'no'.",
)
@click.option("--eval_steps", type=int, default=500, help="Run validation every N steps.")
@click.option(
    "--dataloader_num_workers", type=int, default=4, help="Dataloader worker processes."
)
@click.option(
    "--report_to",
    default="swanlab,tensorboard",
    help="Comma-separated logging integrations (e.g. 'swanlab,tensorboard'). "
    "Use 'none' to disable.",
)
@click.option(
    "--run_name",
    default="proxyz-pretrain",
    help="Run name shown in SwanLab / TensorBoard.",
)
@click.option(
    "--logging_dir",
    type=click.Path(),
    default=None,
    help="TensorBoard log directory (default: <output_dir>/runs).",
)
@click.option(
    "--resume_from_checkpoint",
    is_flag=True,
    help="Load the last checkpoint in args.output_dir as saved by a previous instance of Trainer."
    "Restores model weights, optimizer state, and training step.",
)
@click.option(
    "--cluster_files",
    type=click.Path(),
    multiple=True,
    help="Clustering files for cluster-based sampling. Each file has two columns: "
    "cluster_id and data_row_id. Sampling weight = n / (1 + log(n)) where n is cluster size.",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    # ==========================================
    # 0. CHECK DATA SOURCE IS PROVIDED
    # ==========================================
    if not args.data_files and not args.dataset_name:
        raise click.UsageError(
            "No data source given. Pass one or more sequence files, e.g. "
            "`train.py data.txt --tokenizer_file uniref90_30000.json`, "
            "or use --dataset_name to load from HuggingFace. "
            "Use '-' to read from stdin."
        )

    # ==========================================
    # 1. LOAD YOUR CUSTOM BPE TOKENIZER
    # ==========================================
    # Wrap your standalone BPE json file into the Hugging Face ecosystem.
    # Add [BOS]/[EOS] as new special tokens (extends vocab by 2) so the model
    # can learn sequence start/end and stop generation on its own.
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=args.tokenizer_file,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )

    # Add FIM special tokens if FIM training is enabled
    if args.fim_rate > 0:
        fim_tokens = dataset.FIM_TOKENS
        tokenizer.add_special_tokens({"additional_special_tokens": fim_tokens})
        print(f"Added FIM tokens: {fim_tokens} (vocab size: {len(tokenizer)})")

    # Ensure the embedding layer matches this size exactly
    vocab_size = len(tokenizer)

    # ==========================================
    # 2. CONFIGURE DEEPSEEK-STYLE ARCHITECTURE
    # ==========================================
    use_cuda = torch.cuda.is_available()

    # Shared config parameters
    common_config = dict(
        vocab_size=vocab_size,
        hidden_size=args.model_hidden_size,
        intermediate_size=args.model_intermediate_size,
        num_hidden_layers=args.model_num_hidden_layers,
        num_attention_heads=args.model_num_attention_heads,
        max_position_embeddings=args.max_position_embeddings,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )

    if args.use_mla:
        config = DeepseekV2Config(
            **common_config,
            kv_lora_rank=args.kv_lora_rank,
            q_lora_rank=args.q_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
        )
        model = DeepseekV2ForCausalLM(config)
        attn_type = "MLA (DeepSeek-V2)"
    else:
        config = LlamaConfig(
            **common_config,
            num_key_value_heads=args.model_num_key_value_heads,
            hidden_act="silu",
        )
        model = LlamaForCausalLM(config)
        attn_type = "Llama GQA"

    # Ensure all parameters are bf16 — FlashAttention requires fp16 or bf16
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(torch.bfloat16)

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--- Dense DeepSeek-Style Model ---")
        print(f"Attention:           {attn_type}")
        print(f"Attention backend:   {args.attn_implementation}")
        print(f"Total Parameters:    {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    # ==========================================
    # 3. PREPARE YOUR DATASET
    # ==========================================
    # Pre-resolve FIM token IDs once (not per-batch)
    fim_token_ids = None
    if args.fim_rate > 0:
        fim_token_ids = {
            "prefix": tokenizer.convert_tokens_to_ids(dataset.FIM_PREFIX),
            "suffix": tokenizer.convert_tokens_to_ids(dataset.FIM_SUFFIX),
            "middle": tokenizer.convert_tokens_to_ids(dataset.FIM_MIDDLE),
        }

    max_len = config.max_position_embeddings

    def apply_fim(ids, bos_id, eos_id, content, fim_token_ids):
        """Split content into prefix/middle/suffix and rearrange for FIM training.
        Prefix or suffix may be empty, but middle is always non-empty."""
        n = len(content)
        cut1 = random.randint(0, n - 1)
        cut2 = random.randint(cut1 + 1, n)
        prefix, middle, suffix = content[:cut1], content[cut1:cut2], content[cut2:]

        is_spm = random.random() < args.fim_spm_rate
        if is_spm:
            # SPM: <BOS><fim_suffix><suffix><fim_prefix><prefix><fim_middle><middle><EOS>
            first_tag, second_tag = "suffix", "prefix"
            first, second = suffix, prefix
        else:
            # PSM: <BOS><fim_prefix><prefix><fim_suffix><suffix><fim_middle><middle><EOS>
            first_tag, second_tag = "prefix", "suffix"
            first, second = prefix, suffix

        new_ids = (
            [bos_id, fim_token_ids[first_tag]]
            + first
            + [fim_token_ids[second_tag]]
            + second
            + [fim_token_ids["middle"]]
            + middle
            + [eos_id]
        )
        n_mask = 4 + len(first) + len(second)
        new_labels = [-100] * n_mask + middle + [eos_id]

        if len(new_ids) > max_len:
            new_ids = new_ids[:max_len]
            new_labels = new_labels[:max_len]
        return new_ids, new_labels

    def tokenize_function(examples):
        wrapped = [
            f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
            for text in examples[args.text_column]
        ]
        tokenized = tokenizer(wrapped, truncation=True, max_length=max_len)

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i, ids in enumerate(tokenized["input_ids"]):
            # Apply FIM transformation with probability fim_rate (need ≥5 tokens)
            if fim_token_ids and len(ids) >= 5 and random.random() < args.fim_rate:
                bos_id, eos_id = ids[0], ids[-1]
                content = ids[1:-1]
                new_ids, new_labels = apply_fim(ids, bos_id, eos_id, content, fim_token_ids)
                batch_input_ids.append(new_ids)
                batch_attention_mask.append([1] * len(new_ids))
                batch_labels.append(new_labels)
            else:
                batch_input_ids.append(ids)
                batch_attention_mask.append(tokenized["attention_mask"][i])
                batch_labels.append(ids.copy())

        tokenized["input_ids"] = batch_input_ids
        tokenized["attention_mask"] = batch_attention_mask
        tokenized["labels"] = batch_labels

        # Random crop sequences longer than max_token_length (for non-FIM examples)
        if args.max_token_length:
            for i, ids in enumerate(tokenized["input_ids"]):
                if len(ids) > args.max_token_length:
                    start = random.randint(0, len(ids) - args.max_token_length)
                    tokenized["input_ids"][i] = ids[start:start + args.max_token_length]
                    tokenized["attention_mask"][i] = tokenized["attention_mask"][i][start:start + args.max_token_length]
                    tokenized["labels"][i] = tokenized["labels"][i][start:start + args.max_token_length]

        return tokenized

    # Load dataset from HuggingFace or local files
    if args.dataset_name:
        # Load from HuggingFace Hub
        print(f"Loading dataset from HuggingFace: {args.dataset_name}")
        train_dataset = load_dataset(
            args.dataset_name,
            name=args.dataset_config,
            split=args.dataset_split,
        )
        eval_dataset = None
        if args.dataset_eval_split:
            eval_dataset = load_dataset(
                args.dataset_name,
                name=args.dataset_config,
                split=args.dataset_eval_split,
            )
    else:
        # Load from local files
        iterator = dataset.fasta_iterator if args.data_format == "fasta" else dataset.line_iterator

        # Flatten the batched iterators into one-sequence-per-example records.
        def data_generator(data_files):
            for batch in iterator(data_files):
                for seq in batch:
                    yield {"text": seq}

        train_dataset = Dataset.from_generator(
            functools.partial(data_generator, args.data_files)
        )
        eval_dataset = None
        if args.eval_files:
            eval_dataset = Dataset.from_generator(
                functools.partial(data_generator, args.eval_files)
            )

    # Apply tokenization
    text_col = args.text_column if args.dataset_name else "text"

    def tokenize_dataset(dataset):
        return dataset.map(tokenize_function, batched=True, remove_columns=[text_col])

    train_dataset = tokenize_dataset(train_dataset)
    if eval_dataset:
        eval_dataset = tokenize_dataset(eval_dataset)

    if args.verbose:
        print(f"--- Train dataset ---")
        print(f"Examples: {len(train_dataset):,}")
        if eval_dataset:
            print(f"--- Eval dataset ---")
            print(f"Examples: {len(eval_dataset):,}")

    # Load cluster information and compute sampling weights if cluster files provided
    train_sampler = None
    if args.cluster_files:
        # Load all cluster files and build mapping: data_row_id -> cluster_id
        cluster_map = {}  # data_row_id -> cluster_id
        for cluster_file in args.cluster_files:
            with open(cluster_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        cluster_id = parts[0]
                        data_row_id = int(parts[1])
                        cluster_map[data_row_id] = cluster_id
        
        # Count cluster sizes
        cluster_sizes = defaultdict(int)
        for data_row_id, cluster_id in cluster_map.items():
            if data_row_id < len(train_dataset):  # Only count valid indices
                cluster_sizes[cluster_id] += 1
        
        # Compute sampling weights: n / (1 + log(n)) for each cluster
        cluster_weights = {}
        for cluster_id, n in cluster_sizes.items():
            cluster_weights[cluster_id] = 1 / (1 + math.log(n))
        
        # Assign weights to each sample
        sample_weights = []
        for i in range(len(train_dataset)):
            if i in cluster_map:
                cluster_id = cluster_map[i]
                weight = cluster_weights[cluster_id]
                sample_weights.append(weight)
            else:
                # If no cluster info, use uniform weight (1.0)
                sample_weights.append(1.0)
        
        # Create WeightedRandomSampler
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        if args.verbose:
            print(f"--- Cluster-based sampling ---")
            print(f"Clusters: {len(cluster_sizes):,}")
            print(f"Samples with cluster info: {sum(1 for w in sample_weights if w != 1.0):,}")

    # Data collator that pads input_ids, attention_mask, and labels uniformly
    pad_token_id = tokenizer.pad_token_id

    def data_collator(examples):
        max_len = max(len(ex["input_ids"]) for ex in examples)
        input_ids, attention_mask, labels = [], [], []
        for ex in examples:
            pad_len = max_len - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [pad_token_id] * pad_len)
            attention_mask.append(ex["attention_mask"] + [0] * pad_len)
            label_seq = ex.get("labels", ex["input_ids"])
            labels.append(label_seq + [-100] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    # ==========================================
    # 4. TRAINING ARGUMENTS & EXECUTION
    # ==========================================

    # Custom Trainer for FIM loss tracking: caches batch data only
    class FIMTrainer(Trainer):
        def __init__(self, train_sampler=None, **kwargs):
            super().__init__(**kwargs)

            self.train_sampler = train_sampler
            self._logs = {}
        
        def _get_train_sampler(self, train_dataset: Dataset = None):
            if self.train_sampler is not None:
                return self.train_sampler
            return super()._get_train_sampler(train_dataset)
        
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            # Always cache data for FIM loss tracking (training and eval)
            # Cache data BEFORE calling super (which may modify inputs)
            labels = inputs["labels"].clone()

            # Call parent compute_loss (handles label smoothing, loss scaling, etc.)
            loss, outputs = super().compute_loss(
                model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch
            )

            # ONLY track training metrics if the model is actively training
            if model.training:
                for key, val in self.aux_metric_calculator(
                    (
                        self.aux_preprocess_logits_for_metrics(outputs.logits, labels),
                        labels,
                    ), prefix=""
                ).items():
                    if key in self._logs:
                        self._logs[key].append(val)
                    else:
                        self._logs[key] = [val]

            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            if self._logs:
                for key, val in self._logs.items():
                    if isinstance(val, list):
                        val = sum(val) / len(val)  # Avg.
                    logs[key] = val
                self._logs = {}

            super().log(logs, start_time=start_time)

        @staticmethod
        def aux_preprocess_logits_for_metrics(logits, labels):
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss_per_token.view(-1, shift_labels.size(-1))

        @staticmethod
        def aux_metric_calculator(preds, prefix="eval_"):
            """Computes metrics: n_fim / n_std, loss_fim / loss_std etc"""
            loss_per_token, labels = preds

            logs = {}
            with torch.no_grad():
                # Detect FIM examples: labels start with -100
                is_fim = labels[:, 0] == -100

                for tag, mask in [("fim", is_fim), ("std", ~is_fim)]:
                    # Add FIM/standard counts to logs
                    logs[f"{prefix}n_{tag}"] = mask.sum().item()

                    # Add FIM/standard loss to logs
                    if mask.any():
                        shift_labels = labels[mask][..., 1:]
                        valid = shift_labels.reshape(-1) != -100
                        if valid.any():
                            loss = loss_per_token[mask].reshape(-1)
                            logs[f"{prefix}loss_{tag}"] = loss[valid].mean().item()
            return logs

    # Parse report_to: "swanlab,tensorboard" -> ["swanlab", "tensorboard"]
    report_to = [r.strip() for r in args.report_to.split(",") if r.strip()]
    if report_to == ["none"]:
        report_to = "none"

    # TensorBoard log directory (set via env var; `logging_dir` kwarg is deprecated)
    logging_dir = args.logging_dir or f"{args.output_dir}/runs"
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", logging_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,                              # DeepSeek beta2 standard
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if (args.eval_files or args.dataset_eval_split) else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        eval_accumulation_steps=args.gradient_accumulation_steps,
        prediction_loss_only=True,                    # only returns the loss
        bf16=use_cuda,                                # bf16 is preferred over fp16 on modern GPUs
        ddp_find_unused_parameters=False,             # disabled warning
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=report_to,                          # SwanLab + TensorBoard
        run_name=args.run_name,
    )

    trainer = FIMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,                  # transformers >=5 renamed `tokenizer`
        train_sampler=train_sampler,
        preprocess_logits_for_metrics=FIMTrainer.aux_preprocess_logits_for_metrics,
        compute_metrics=FIMTrainer.aux_metric_calculator,
    )


    # ==========================================
    # 5. INITIALIZE LOGGERS & START TRAINING
    # ==========================================
    # SwanLab >=0.8 requires an explicit init before the callback can get_run().
    # We init it ourselves so the SwanLabCallback's setup() finds an active run.
    _swanlab_active = False
    if report_to != "none" and "swanlab" in report_to:
        try:
            import swanlab

            swanlab_mode = os.environ.get("SWANLAB_MODE", "cloud")
            swanlab.init(
                name=args.run_name,
                project=os.environ.get("SWANLAB_PROJECT", "proxyz"),
                mode=swanlab_mode,
            )
            _swanlab_active = True
        except Exception as e:
            print(f"[warn] SwanLab init failed ({e}), continuing without SwanLab.")

    # Start or resume pre-training
    if args.resume_from_checkpoint:
        print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Finish SwanLab run
    if _swanlab_active:
        import swanlab
        swanlab.finish()

    # Save final weights and configuration
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Clean up distributed process group to avoid resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
