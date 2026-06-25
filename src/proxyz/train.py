import os
import random
import functools

import click
import torch
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    DeepseekV2Config,
    DeepseekV2ForCausalLM,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset, load_dataset

from proxyz.utils import dict2object
from proxyz.data.dataset import line_iterator, fasta_iterator


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
        fim_tokens = ["<fim_prefix>", "<fim_suffix>", "<fim_middle>"]
        tokenizer.add_special_tokens({"additional_special_tokens": fim_tokens})
        print(f"Added FIM tokens: {fim_tokens} (vocab size: {len(tokenizer)})")

    # Ensure the embedding layer matches this size exactly
    vocab_size = len(tokenizer)

    # ==========================================
    # 2. CONFIGURE DEEPSEEK-STYLE ARCHITECTURE
    # ==========================================
    if args.use_mla:
        # Multi-head Latent Attention (MLA) from DeepSeek-V2
        config = DeepseekV2Config(
            vocab_size=vocab_size,
            hidden_size=args.model_hidden_size,
            intermediate_size=args.model_intermediate_size,
            num_hidden_layers=args.model_num_hidden_layers,
            num_attention_heads=args.model_num_attention_heads,
            # MLA-specific parameters
            kv_lora_rank=args.kv_lora_rank,
            q_lora_rank=args.q_lora_rank,
            qk_nope_head_dim=args.qk_nope_head_dim,
            qk_rope_head_dim=args.qk_rope_head_dim,
            v_head_dim=args.v_head_dim,
            # Common parameters
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
        model = DeepseekV2ForCausalLM(config)
        attn_type = "MLA (DeepSeek-V2)"
    else:
        # DeepSeek-V2/V3 use Llama-based primitives (SwiGLU, RMSNorm, RoPE)
        config = LlamaConfig(
            vocab_size=vocab_size,
            hidden_size=args.model_hidden_size,                  # Model width
            intermediate_size=args.model_intermediate_size,      # SwiGLU hidden dimension (usually ~8/3 of hidden_size)
            num_hidden_layers=args.model_num_hidden_layers,      # Depth
            num_attention_heads=args.model_num_attention_heads,  # Attention heads
            num_key_value_heads=args.model_num_key_value_heads,  # Grouped-Query Attention (GQA) for speed
            hidden_act="silu",                                   # SiLU activation for SwiGLU
            max_position_embeddings=args.max_position_embeddings,  # Context window length
            initializer_range=0.02,
            rms_norm_eps=1e-6,                                   # DeepSeek RMSNorm epsilon
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            attn_implementation=args.attn_implementation,
            torch_dtype=torch.bfloat16,
            tie_word_embeddings=False                            # DeepSeek keeps input/output embeddings separate
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
    # Tokenize each sequence independently (no cross-sequence concatenation):
    # every line is wrapped as [BOS] + sequence + [EOS] and truncated, so the
    # model learns where sequences start and end (and when to stop generating).
    # Optionally applies FIM (Fill-in-the-Middle) transformation.
    def tokenize_function(examples):
        wrapped = [
            f"{tokenizer.bos_token}{text}{tokenizer.eos_token}"
            for text in examples[args.text_column]
        ]
        tokenized = tokenizer(
            wrapped,
            truncation=True,
            max_length=config.max_position_embeddings,
        )

        # Get FIM token IDs if FIM is enabled
        fim_prefix_id = None
        fim_suffix_id = None
        fim_middle_id = None
        if args.fim_rate > 0:
            fim_prefix_id = tokenizer.convert_tokens_to_ids("<fim_prefix>")
            fim_suffix_id = tokenizer.convert_tokens_to_ids("<fim_suffix>")
            fim_middle_id = tokenizer.convert_tokens_to_ids("<fim_middle>")

        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []

        for i, ids in enumerate(tokenized["input_ids"]):
            mask = tokenized["attention_mask"][i]

            # Apply FIM transformation with probability fim_rate
            # Need at least 4 tokens: BOS + 1 prefix + 1 middle + 1 suffix + EOS
            if args.fim_rate > 0 and len(ids) >= 5 and random.random() < args.fim_rate:
                # Split: [BOS] prefix middle suffix [EOS]
                bos_id = ids[0]
                eos_id = ids[-1]
                content = ids[1:-1]  # tokens between BOS and EOS

                # Random split into 3 parts
                n = len(content)
                cut1 = random.randint(1, n - 1)
                cut2 = random.randint(cut1, n - 1)
                prefix = content[:cut1]
                middle = content[cut1:cut2]
                suffix = content[cut2:]

                if random.random() < args.fim_spm_rate:
                    # SPM format: <BOS><fim_suffix><suffix><fim_prefix><prefix><fim_middle><middle><EOS>
                    new_ids = [bos_id, fim_suffix_id] + suffix + [fim_prefix_id] + prefix + [fim_middle_id] + middle + [eos_id]
                    # Labels: -100 for everything except middle
                    new_labels = [-100] * (3 + len(suffix) + len(prefix)) + middle + [eos_id]
                else:
                    # PSM format: <BOS><fim_prefix><prefix><fim_suffix><suffix><fim_middle><middle><EOS>
                    new_ids = [bos_id, fim_prefix_id] + prefix + [fim_suffix_id] + suffix + [fim_middle_id] + middle + [eos_id]
                    # Labels: -100 for everything except middle
                    new_labels = [-100] * (3 + len(prefix) + len(suffix)) + middle + [eos_id]

                # Truncate if too long
                max_len = config.max_position_embeddings
                if len(new_ids) > max_len:
                    new_ids = new_ids[:max_len]
                    new_labels = new_labels[:max_len]

                batch_input_ids.append(new_ids)
                batch_attention_mask.append([1] * len(new_ids))
                batch_labels.append(new_labels)
            else:
                # Normal training: predict all tokens
                batch_input_ids.append(ids)
                batch_attention_mask.append(mask)
                # Labels are input_ids shifted by 1 (handled by DataCollator)
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
        iterator = fasta_iterator if args.data_format == "fasta" else line_iterator

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
    columns_to_remove = [args.text_column] if args.dataset_name else ["text"]
    train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=columns_to_remove,
    )
    if eval_dataset:
        eval_dataset = eval_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=columns_to_remove,
        )

    if args.verbose:
        print(f"--- Train dataset ---")
        print(f"Examples: {len(train_dataset):,}")
        if eval_dataset:
            print(f"--- Eval dataset ---")
            print(f"Examples: {len(eval_dataset):,}")

    # Data collator pads each batch and shifts labels internally for Causal LM.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # ==========================================
    # 4. TRAINING ARGUMENTS & EXECUTION
    # ==========================================

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
        bf16=use_cuda,                                # bf16 is preferred over fp16 on modern GPUs
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        report_to=report_to,                         # SwanLab + TensorBoard
        run_name=args.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        processing_class=tokenizer,                  # transformers >=5 renamed `tokenizer`
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
