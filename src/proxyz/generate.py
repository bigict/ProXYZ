import os
from datetime import datetime
import glob
import re

import click
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    GenerationConfig,
    LogitsProcessorList,
    SuppressTokensLogitsProcessor,
)

from proxyz.data import dataset
from proxyz.utils import dict2object


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
    def resolve_model_path(model_dir: str) -> str:
        """Return model_dir if it holds a model directly, else the latest checkpoint-*."""
        if os.path.isfile(os.path.join(model_dir, "model.safetensors")) or os.path.isfile(
            os.path.join(model_dir, "pytorch_model.bin")
        ):
            return model_dir

        checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
        checkpoints = [c for c in checkpoints if re.search(r"checkpoint-(\d+)$", c)]
        if not checkpoints:
            raise click.UsageError(
                f"No model weights found in {model_dir} and no checkpoint-* subdirectories. "
                "Pass --model_dir pointing at a trained model or checkpoint."
            )
        latest = max(checkpoints, key=lambda c: int(re.search(r"checkpoint-(\d+)$", c).group(1)))
        return latest


    model_path = resolve_model_path(args.model_dir)
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

    # Normal generation mode
    prompt_ids = processor({processor.text_column: [args.prompt]}, generate=True)

    sequences = []
    remaining = args.num_sequences
    while remaining > 0:
        n = min(args.batch_size, remaining)
        input_ids = {k: v.repeat(n, *[1]*(v.dim() - 1)).to(device) for k, v in prompt_ids.items()}
        with torch.no_grad():
            out = model.generate(
                **input_ids,
                processor=processor,
                generation_config=gen_config,
                logits_processor=logits_processor,
            )
        for row in out:
            # Decode keeping FIM special tokens, removing only [BOS]/[EOS]/[PAD]/[UNK]
            decoded = processor.tokenizer.decode(row.tolist(), skip_special_tokens=False)
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
        remaining -= n
        if args.verbose:
            print(f"  generated {len(sequences)}/{args.num_sequences}")

    # ==========================================
    # 5. WRITE FASTA OUTPUT
    # ==========================================
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"generated_{timestamp}.fasta")

    with open(output_path, "w") as f:
        for i, seq in enumerate(sequences):
            header = f">proxyz_gen_{i} length={len(seq)}"
            f.write(f"{header}\n{dataset.fasta_wrap(seq)}\n")

    print(f"Wrote {len(sequences)} sequences to {output_path}")


if __name__ == "__main__":
    main()

