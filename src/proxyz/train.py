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
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, load_dataset

from proxyz.data import dataset, sampler
from proxyz.models import XYZConfig, XYZForCausalLM, XYZProcessor
from proxyz.utils import dict2object, compose


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
    type=click.Choice(["line", "fasta", "pdb"]),
    default="line",
    help="Input format: one sequence per line ('line'), FASTA ('fasta') or PDB ('pdb/cif').",
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
    "--use_unet",
    is_flag=True,
    help="Use U-net style XYZForCausalLM instead of standard Llama attention.",
)
@click.option(
    "--model_char_hidden_size",
    type=int,
    default=768,
    help="Character: Model width.",
)
@click.option(
    "--model_char_intermediate_size",
    type=int,
    default=2064,
    help="Character: Model SwiGLU hidden dimension (usually ~8/3 of hidden_size).",
)
@click.option(
    "--model_char_num_hidden_layers",
    type=int,
    default=2,
    help="Character: Model depth.",
)
@click.option(
    "--model_char_num_attention_heads",
    type=int,
    default=6,
    help="Character: Model attention heads.",
)
@click.option(
    "--model_use_char_position_ids",
    is_flag=True,
    help="Character: Model gather position_ids from char_position_ids",
)
@click.option(
    "--model_has_char_lm_head",
    is_flag=True,
    help="Character: Model has char_lm_head",
)
@click.option(
    "--model_has_cle_lm_head",
    is_flag=True,
    help="Character: Model has cle_lm_head",
)
@click.option(
    "--model_has_distogram_lm_head",
    is_flag=True,
    help="Character: Model has distogram_lm_head",
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
    "--dataloader_num_workers",
    type=int,
    default=4,
    help="Dataloader worker processes."
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

    features = []
    label_names = [("labels", 1)]
    keys_to_ignore_at_inference = ["past_key_values", "char_past_key_values"]
    if args.model_has_char_lm_head:
        label_names += [("char_labels", 1)]
    else:
        keys_to_ignore_at_inference += ["char_logits"]
    if args.model_has_cle_lm_head:
        label_names += [("cle_labels", 0)]
        features += ["cle_labels"]
    else:
        keys_to_ignore_at_inference += ["cle_logits"]
    if args.model_has_distogram_lm_head:
        label_names += [("distogram_labels", 0)]
        features += ["distogram_labels"]
    else:
        keys_to_ignore_at_inference += ["distogram_logits"]

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
        tokenizer=tokenizer,
        text_column=args.text_column,
        features=features,
    )

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
        keys_to_ignore_at_inference=keys_to_ignore_at_inference
    )

    if args.use_unet:
        config = XYZConfig(
            **common_config,
            char_hidden_size=args.model_char_hidden_size,
            char_intermediate_size=args.model_char_intermediate_size,
            char_num_hidden_layers=args.model_char_num_hidden_layers,
            char_num_attention_heads=args.model_char_num_attention_heads,
            use_char_position_ids=args.model_use_char_position_ids,
            has_char_lm_head=args.model_has_char_lm_head,
            has_cle_lm_head=args.model_has_cle_lm_head,
            has_distogram_lm_head=args.model_has_distogram_lm_head,
        )
        model = XYZForCausalLM(config)
    else:
        config = LlamaConfig(**common_config, )
        model = LlamaForCausalLM(config)

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
        if args.data_format == "pdb":
            examples = dataset.pdb_transform(os.environ["DATA_PATH"], examples)
        examples = processor(
            examples,
            bpe_dropout=args.tokenizer_bpe_dropout,
            char_apply=args.use_unet,
            fim_apply=fim_apply,
            fim_spm_rate=args.fim_spm_rate,
            fim_sft_style=args.fim_sft_style,
            max_length=max_sequence_length - (5 if fim_apply else 2),
        )
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
            # Always cache data for FIM loss tracking (training and eval)
            # Cache data BEFORE calling super (which may modify inputs)
            labels = tuple(inputs[k].clone() for k in self.args.label_names)
            if len(labels) == 1:
                labels = labels[0]

            # Call parent compute_loss (handles label smoothing, loss scaling, etc.)
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch
            )

            # ONLY track training metrics if the model is actively training
            if model.training and self.compute_metrics is not None:
                ignore_keys = []

                module = model
                # FIX: DistributedDataParallel
                if not hasattr(module, "config") and hasattr(module, "module"):
                    module = module.module
                if hasattr(module, "config"):
                    ignore_keys = getattr(
                        module.config,
                        "keys_to_ignore_at_inference",
                        ["past_key_values"]
                    )

                logits = tuple(
                    v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                )
                if len(logits) == 1:
                    logits = logits[0]
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                for key, val in self.compute_metrics(
                    (logits, labels), update_metrics=False, prefix=""
                ).items():
                    if key in self._logs:
                        self._logs[key].append(val)
                    else:
                        self._logs[key] = [val]

            return (loss, outputs) if return_outputs else loss

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys):
            loss, logits, labels = super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys
            )

            def pad_across_processes(tensors, dims=None):
                if isinstance(tensors, tuple):
                    return tuple(
                        map(functools.partial(pad_across_processes, dims=dims), tensors)
                    )
                if dims is None:
                    dims = tensors.dim()
                elif dims < 0:
                    dims = dims + tensors.dim()
                for dim in range(2, dims):
                    tensors = self.accelerator.pad_across_processes(
                        tensors, dim=dim, pad_index=-100
                    )
                return tensors

            if logits is not None:
                logits = pad_across_processes(logits, dims=-1)
            if labels is not None:
                labels = pad_across_processes(labels)
            return loss, logits, labels

        def log(self, logs, start_time=None):
            if self._logs:
                for key, val in self._logs.items():
                    if isinstance(val, list):
                        val = sum(val) / len(val)  # Avg.
                    logs[key] = val
                self._logs = {}

            super().log(logs, start_time=start_time)

        @classmethod
        def aux_preprocess_logits_for_metrics(cls, logits, labels, label_names=None):
            loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
            if isinstance(logits, tuple) and isinstance(labels, tuple):
                return tuple(
                    cls.aux_preprocess_logits_for_metrics(
                        p, l, label_names=label_names[i] if label_names else None
                    ) for i, (p, l) in enumerate(zip(logits, labels))
                )
            assert not isinstance(logits, tuple), len(logits)
            assert not isinstance(labels, tuple), len(labels)

            n_shift = 0
            if label_names is None or label_names[1]:
                n_shift = labels.dim() - 1
            logits_slices = [slice(0, -1)] * n_shift
            labels_slices = [slice(1, None)] * n_shift

            shift_logits = logits[..., *logits_slices, :].contiguous()
            shift_labels = labels[..., *labels_slices].contiguous()

            loss_per_token = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            return loss_per_token.view(-1, *shift_labels.shape[1:])

        @classmethod
        def aux_metric_calculator(
            cls, metrics, preds, compute_result=False, update_metrics=True, prefix="eval_", label_names=None
        ):
            """Computes metrics: n_fim / n_std, loss_fim / loss_std etc"""
            loss_per_token, labels = preds

            logs = {}

            if isinstance(loss_per_token, tuple) and isinstance(labels, tuple):
                assert len(label_names) == len(loss_per_token)
                for i, (p, l) in enumerate(zip(loss_per_token, labels)):
                    label, _ = label_names[i]
                    if label.endswith("labels"):
                        label = label[:-len("labels")]
                    logs.update(
                        cls.aux_metric_calculator(
                            metrics, (p, l),
                            update_metrics=False,
                            prefix=f"{prefix}{label}",
                            label_names=label_names[i],
                        )
                    )
            elif isinstance(loss_per_token, list) and isinstance(labels, list):
                assert len(loss_per_token) % len(label_names) == 0, ([l.shape for l in loss_per_token], [l.shape for l in labels], label_names)
                gather_logs = defaultdict(list)

                for g in range(len(loss_per_token) // len(label_names)):
                    i, j = g * len(label_names), (g + 1) * len(label_names)
                    for key, value in cls.aux_metric_calculator(
                        metrics, (tuple(loss_per_token[i:j]), tuple(labels[i:j])),
                        update_metrics=False,
                        prefix=prefix,
                        label_names=label_names,
                    ).items():
                        gather_logs[key].append(value)

                logs.update({k: sum(v)/len(v) for k, v in gather_logs.items()})
            else:
                with torch.no_grad():
                    # Detect FIM examples: labels start with -100
                    if args.fim_sft_style:
                        is_fim = (labels[..., 0:1] == -100)
                    else:
                        is_fim = (
                            labels == processor.tokenizer.convert_tokens_to_ids(
                                processor.FIM_MIDDLE
                            )
                        )
                    is_fim = is_fim.any(tuple(range(1, is_fim.dim())))

                    eps = 1e-8
                    n_shift = 0
                    if label_names is None or label_names[1]:
                        n_shift = labels.dim() - 1
                    labels_slices = [slice(1, None)] * n_shift
                    for tag, mask in [("fim", is_fim), ("std", ~is_fim)]:
                        # Add FIM/standard counts to logs
                        logs[f"{prefix}n_{tag}"] = mask.sum().item()

                        # Add FIM/standard loss to logs
                        if mask.any():
                            shift_labels = labels[mask][..., *labels_slices]
                            valid = (shift_labels != -100).reshape(-1)
                            if valid.any():
                                loss = loss_per_token[mask].reshape(-1)
                                logs[f"{prefix}loss_{tag}"] = loss[valid].mean().item()

            if update_metrics:
                for k, v in logs.items():
                    metrics[k].append(v)

                if compute_result:
                    logs = {k: sum(v)/len(v) for k, v in metrics.items()}
                    metrics.clear()
                    return logs

            return logs

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
        eval_use_gather_object=True,
        eval_on_start=True if args.eval_files else False,
        batch_eval_metrics=True,
        label_names=[label for label, _ in label_names],
        bf16=use_cuda,                                # bf16 is preferred over fp16 on modern GPUs
        ddp_find_unused_parameters=False,             # disabled warning
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_drop_last=True,
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
        preprocess_logits_for_metrics=functools.partial(
            FIMTrainer.aux_preprocess_logits_for_metrics,
            label_names=label_names,
        ),
        compute_metrics=functools.partial(
            FIMTrainer.aux_metric_calculator,
            defaultdict(list),
            label_names=label_names,
        ),
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
