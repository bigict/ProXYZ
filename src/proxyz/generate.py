import os
from datetime import datetime
import functools

import click
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)
from tqdm import tqdm

from proxyz.data import dataset
from proxyz.utils import data_utils, dict2object, model_utils


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Trained model dir. If it has no weights at the root, the latest "
    "checkpoint-* subdirectory is used automatically.",
)
@click.option(
    "--num_tokens",
    type=int,
    default=100,
    help="Max number of tokens to generate per sequence. With [EOS] the model "
    "may stop earlier; use --force_length to always generate exactly this many.",
)
@click.option(
    "-n", "--num_sequences", type=int, default=10, help="How many sequences to generate."
)
@click.option(
    "--prompt",
    type=str,
    default="",
    help="Optional seed residues to start generation from (after [BOS]). "
    "Empty means unconditional generation from [BOS] only.",
)
@click.option(
    "--force_length",
    is_flag=True,
    help="Disable [EOS] stopping and generate exactly --num_tokens tokens.",
)
@click.option("--temperature", type=float, default=1.0, help="Sampling temperature.")
@click.option("--top_p", type=float, default=0.95, help="Nucleus (top-p) sampling.")
@click.option("--top_k", type=int, default=0, help="Top-k sampling (0 disables).")
@click.option("--batch_size", type=int, default=8, help="Sequences generated per batch.")
@click.option(
    "--output_dir",
    type=click.Path(),
    default="./generated_sequences",
    help="Directory where the output FASTA file is written.",
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--attn_implementation",
    type=click.Choice(["flash_attention_2", "sdpa", "eager"]),
    default="flash_attention_2",
    help="Attention backend used for inference.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device to run on (default: cuda if available else cpu).",
)
@click.option(
    "--use_cache",
    is_flag=True,
    help="Enable model to compute and store the key/value hidden states for past tokens",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    torch.manual_seed(args.seed)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.startswith("cuda") and torch.cuda.is_available()

    # ==========================================
    # 1. RESOLVE MODEL & LOAD TOKENIZER
    # ==========================================
    model_path = model_utils.resolve_model_path(args.model_dir)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    if processor.tokenizer.bos_token_id is None or processor.tokenizer.eos_token_id is None:
        raise click.UsageError(
            "Tokenizer has no [BOS]/[EOS]. Retrain with the updated train.py "
            "(which adds them) or point --model_dir at such a model."
        )

    # ==========================================
    # 2. LOAD TRAINED MODEL
    # ==========================================
    # FlashAttention needs half precision; bf16 is also the fastest path on CUDA.
    dtype = torch.bfloat16 if use_cuda else torch.float32
    attn_impl = args.attn_implementation if use_cuda else "eager"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    if args.verbose:
        print(f"--- ProXYZ Generation ---")
        print(f"Model:       {model_path}")
        print(f"Device:      {device}  dtype={dtype}  attn={attn_impl}")
        print(f"Vocab size:  {len(processor.tokenizer)}  bos={processor.tokenizer.bos_token_id} eos={processor.tokenizer.eos_token_id}")
        print(f"Prompt:      {args.prompt!r}")
        mode = "exactly" if args.force_length else "up to"
        print(f"Target:      {mode} {args.num_tokens} new tokens x {args.num_sequences} seqs")

    # ==========================================
    # 3. BUILD GENERATION CONFIG
    # ==========================================
    # By default the model may emit [EOS] early (variable-length generation).
    # With --force_length we disable [EOS] and pad to exactly num_tokens.
    gen_config = GenerationConfig(
        do_sample=True,
        max_new_tokens=args.num_tokens,
        min_new_tokens=args.num_tokens if args.force_length else None,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else None,
        eos_token_id=None if args.force_length else processor.tokenizer.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        use_cache=args.use_cache,
    )

    # ==========================================
    # 4. ENCODE PROMPT & GENERATE IN BATCHES
    # ==========================================
    # Ban all tokens whose string representation contains "X" (unknown residue).
    # This prevents the model from emitting ambiguous amino-acid placeholders.
    vocab = processor.tokenizer.get_vocab()
    suppress_ids = [tid for token, tid in vocab.items()
                    if "X" in token and not token.startswith("[")]
    logits_processor = LogitsProcessorList()
    if suppress_ids:
        logits_processor.append(
            SuppressTokensLogitsProcessor(suppress_tokens=suppress_ids, device=device)
        )
        if args.verbose:
            print(f"Suppressed {len(suppress_ids)} tokens containing 'X'")

    def data_generator():
        for idx in range(args.num_sequences):
            yield {processor.text_column: args.prompt}

    # Apply tokenization
    def tokenize_dataset(prompt_dataset):
        return prompt_dataset.with_transform(
            functools.partial(
                data_utils.tokenize_function,
                processor,
                "line",
                char_apply=model.config.has_characterization,
                generate=True,
            )
        )

    prompt_dataset = Dataset.from_generator(data_generator)
    prompt_dataset = tokenize_dataset(prompt_dataset)

    prompt_dataloader = DataLoader(
        prompt_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    # Normal generation mode
    sequences = []
    for input_ids in tqdm(prompt_dataloader, desc="generation"):
        input_ids = {
            k: v.to(device) for k, v in input_ids.items() if torch.is_tensor(v)
        }
        input_ids = data_utils.prepare_inputs(processor, input_ids)
        with torch.no_grad():
            out = model.generate(
                **input_ids,
                processor=processor,
                generation_config=gen_config,
                logits_processor=logits_processor,
            )
        for row in out:
            # Decode keeping FIM special tokens, removing only [BOS]/[EOS]/[PAD]/[UNK]
            decoded = processor.tokenizer.decode(
                row.tolist(), skip_special_tokens=False
            )
            # Remove non-FIM special tokens
            for special in [
                processor.tokenizer.bos_token,
                processor.tokenizer.eos_token,
                processor.tokenizer.pad_token,
                processor.tokenizer.unk_token
            ]:
                if special:
                    decoded = decoded.replace(special, "")
            # Remove whitespace (FIM tokens like <fim_prefix> don't contain spaces)
            seq = decoded.replace(" ", "")
            sequences.append(seq)
        if args.verbose:
            print(f"  generated {len(sequences)}/{args.num_sequences}")

    # ==========================================
    # 5. WRITE FASTA OUTPUT
    # ==========================================
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"generated_{timestamp}.fasta")

    if sequences:
        with open(output_path, "w") as f:
            for i, seq in enumerate(sequences):
                header = f">proxyz_gen_{i} length={len(seq)}"
                f.write(f"{header}\n{dataset.fasta_wrap(seq)}\n")

    print(f"Wrote {len(sequences)} sequences to {output_path}")


if __name__ == "__main__":
    main()

