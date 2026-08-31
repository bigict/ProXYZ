import os
import functools
import random

import click
from datasets import Dataset, load_dataset
import torch
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments

from proxyz.data import dataset, sampler
from proxyz.models import XYZConfig, XYZForCausalLM, XYZProcessor
from proxyz.utils import dict2object


@click.command(context_settings={'show_default': True})
@click.argument("data_files", type=click.Path(), nargs=-1)
@click.option(
    "--eval_files", type=click.Path(), multiple=True, help="Evaluate data files"
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
    "--tokenizer_bpe_dropout",
    type=float,
    default=0,
    help="Stochastic BPE, typically implemented as BPE-Dropout, is a subword "
    "regularization method that randomly drops valid subword merges during "
    "tokenization.",
)
@click.option(
    "--data_format",
    type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line",
    help="Input format: one sequence per line ('line'), FASTA ('fasta'), "
    "Foldcomp ('foldcomp') or PDB ('pdb/cif').",
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
    "--gradient_accumulation_steps",
    type=int,
    default=8,
    help="Grad accumulation steps."
)
@click.option("--learning_rate", type=float, default=3e-4, help="Peak learning rate.")
@click.option(
    "--warmup_steps",
    type=int,
    default=0,
    help="Number of steps for a linear warmup from 0 to `learning_rate`."
)
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
    "--max_sequence_length",
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
    "--fim_sft_style",
    is_flag=True,
    help="Only the tokens after <middle> has loss.",
)
@click.option(
    "--eval_strategy",
    type=click.Choice(["no", "steps", "epoch"]),
    default="steps",
    help="When to run validation: 'steps' (every eval_steps), 'epoch' (end of each epoch), or 'no'.",
)
@click.option(
    "--eval_steps", type=int, default=500, help="Run validation every N steps."
)
@click.option(
    "--dataloader_num_workers", type=int, default=4, help="Dataloader worker processes."
)
@click.option(
    "--dataloader_prefetch_factor",
    type=int,
    default=None,
    help="Number of batches loaded in advance by each worker."
)
@click.option(
    "--report_to",
    multiple=True,
    default=("swanlab", "tensorboard"),
    help="Logging integration(s). Repeat for multiple, e.g. "
    "--report_to swanlab --report_to tensorboard. Use 'none' to disable.",
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
@click.option(
    "--random_seed",
    type=float,
    default=None,
    help="Initializes the underlying pseudo-random number generator (PRNG).",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    random.seed(args.random_seed)

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
    processor = XYZProcessor(
        tokenizer=tokenizer, text_column=args.text_column
    )

    # Ensure the embedding layer matches this size exactly
    vocab_size = len(tokenizer)

    # ==========================================
    # 2. CONFIGURE DEEPSEEK-STYLE ARCHITECTURE
    # ==========================================
    use_cuda = torch.cuda.is_available()

    # Shared config parameters
    config = XYZConfig(
        vocab_size=vocab_size,
        hidden_size=args.model_hidden_size,
        intermediate_size=args.model_intermediate_size,
        num_hidden_layers=args.model_num_hidden_layers,
        num_attention_heads=args.model_num_attention_heads,
        num_key_value_heads=args.model_num_key_value_heads,
        max_position_embeddings=args.max_position_embeddings,
        hidden_act="silu",
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )
    model = XYZForCausalLM(config)

    # Ensure all parameters are bf16 — FlashAttention requires fp16 or bf16
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        model = model.to(torch.bfloat16)

    if args.verbose:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"--- Dense DeepSeek-Style Model ---")
        print(f"Attention backend:   {args.attn_implementation}")
        print(f"Total Parameters:    {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

    # ==========================================
    # 3. PREPARE YOUR DATASET
    # ==========================================
    max_sequence_length = args.max_sequence_length
    if max_sequence_length is None:
        max_sequence_length = args.max_position_embeddings

    def tokenize_function(examples):
        fim_apply = random.random() < args.fim_rate

        transform = dataset.data_transform(args.data_format)
        if transform is not None:
            examples = transform(examples)

        batch_size = len(examples[args.text_column])
        examples = processor(
            examples,
            bpe_dropout=args.tokenizer_bpe_dropout,
            fim_apply=fim_apply,
            fim_spm_rate=args.fim_spm_rate,
            fim_sft_style=args.fim_sft_style,
            max_length=max_sequence_length - (5 if fim_apply else 2),
        )
        examples["is_fim"] = [fim_apply] * batch_size
        return examples

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
        iterator = dataset.data_iterator(args.data_format)

        # Flatten the batched iterators into one-sequence-per-example records.
        def data_generator(data_files):
            for batch in iterator(data_files):
                yield from batch

        train_dataset = Dataset.from_generator(
            functools.partial(data_generator, args.data_files)
        )
        eval_dataset = None
        if args.eval_files:
            eval_dataset = Dataset.from_generator(
                functools.partial(data_generator, args.eval_files)
            )

    # Load cluster information and compute sampling weights if cluster files provided
    train_sampler = None
    if args.cluster_files:
        train_sampler = sampler.from_cluster_files(train_dataset, args.cluster_files)
        if args.verbose:
            print(f"--- Cluster-based sampling ---")
            print(f"Clusters: {len(train_sampler):,}")

    # Apply tokenization
    def tokenize_dataset(dataset):
        return dataset.with_transform(tokenize_function)

    train_dataset = tokenize_dataset(train_dataset)
    if eval_dataset:
        eval_dataset = tokenize_dataset(eval_dataset)

    if args.verbose:
        print(f"--- Train dataset ---")
        print(f"Examples: {len(train_dataset):,}")
        if eval_dataset:
            print(f"--- Eval dataset ---")
            print(f"Examples: {len(eval_dataset):,}")

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
            # Call parent compute_loss (handles label smoothing, loss scaling, etc.)
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )

            # ONLY track training metrics if the model is actively training
            if not hasattr(self, "_last_logged"):
                self._last_logged = self._globalstep_last_logged

            if (
                self.args.average_tokens_across_devices
                and (self.model_accepts_loss_kwargs or self.compute_loss_func)
                and num_items_in_batch is not None
            ):
                # TP and EP-as-TP ranks see replicated batches; `num_processes` over-counts
                # them by `tp_size`. Mirror the divisor used in `_get_num_items_in_batch`.
                loss_scale = self.accelerator.num_processes
                if (pc := getattr(self.accelerator, "parallelism_config", None)) is not None:
                    loss_scale //= pc.tp_size
                loss_scale = loss_scale if self.args.n_gpu <= 1 else self.args.n_gpu
            else:
                loss_scale = 1

            def logs_update(key, val):
                if self.model.training:
                    if key in self._logs:
                        self._logs[key] += val
                    else:
                        self._logs[key] = val
                else:
                    key = f"eval_{key}"
                    if key in self._logs:
                        self._logs[key].append(val)
                    else:
                        self._logs[key] = [val]

            n_fim, n_std = inputs["is_fim"].sum().item(), (~inputs["is_fim"]).sum().item()
            for tag, n in [("fim", n_fim), ("std", n_std)]:
                logs_update(f"n_{tag}", n * loss_scale)
                if n > 0:
                    val = loss.detach().item()
                    logs_update(f"loss_{tag}", val * n / (n_fim + n_std))
                    for key, val in outputs.items():
                        if key.endswith("_loss") and val is not None:
                            val = val.detach().item() * loss_scale
                            logs_update(f"{key}_{tag}", val * n / (n_fim + n_std))

            return (loss, outputs) if return_outputs else loss

        def log(self, logs, start_time=None):
            if self._logs:
                for key, val in self._logs.items():
                    if isinstance(val, list):
                        val = sum(val) / len(val)  # Avg.
                    logs[key] = val / max(self.state.global_step - self._last_logged, 1)

                self._last_logged = self._globalstep_last_logged
                self._logs.clear()

            super().log(logs, start_time=start_time)

    report_to = list(args.report_to) if args.report_to else []
    if report_to == ["none"] or report_to == ["all"]:
        report_to = "".join(report_to)

    # TensorBoard log directory (set via env var; `logging_dir` kwarg is deprecated)
    logging_dir = args.logging_dir or f"{args.output_dir}/runs"
    os.environ.setdefault("TENSORBOARD_LOGGING_DIR", logging_dir)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,                              # DeepSeek beta2 standard
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_strategy=args.eval_strategy if (args.eval_files or args.dataset_eval_split) else "no",
        eval_steps=args.eval_steps if args.eval_strategy == "steps" else None,
        per_device_eval_batch_size=args.per_device_train_batch_size,
        eval_accumulation_steps=args.gradient_accumulation_steps,
        eval_on_start=True if args.eval_files else False,
        prediction_loss_only=True,
        bf16=use_cuda,                                # bf16 is preferred over fp16 on modern GPUs
        ddp_find_unused_parameters=False,             # disabled warning
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_persistent_workers=True,
        remove_unused_columns=False,
        report_to=report_to,                          # SwanLab + TensorBoard
        run_name=args.run_name,
    )

    trainer = FIMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,                  # transformers >=5 renamed `tokenizer`
        train_sampler=train_sampler,
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
    processor.save_pretrained(args.output_dir)

    # Clean up distributed process group to avoid resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
