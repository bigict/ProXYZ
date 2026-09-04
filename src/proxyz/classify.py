import os
from datetime import datetime
import functools
import glob
import re

import click
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoProcessor)
from tqdm import tqdm

from proxyz.data import dataset
from proxyz.utils import data_utils, dict2object


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Trained model dir. If it has no weights at the root, the latest "
    "checkpoint-* subdirectory is used automatically.",
)
@click.option(
    "-n", "--num_sequences", type=int, default=10, help="How many sequences to generate."
)
@click.option(
    "--data_files", type=click.Path(), multiple=True, help="Data files to classify. "
)
@click.option(
    "--data_format",
    type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line",
    help="Input format: one sequence per line ('line'), FASTA ('fasta'), "
    "Foldcomp ('foldcomp') or PDB ('pdb/cif').",
)
@click.option(
    "--max_sequence_length",
    type=int,
    default=None,
    help="If set, randomly crop sequences longer than this to a subsequence of this length. "
    "Useful for controlling memory usage with variable-length inputs.",
)
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
        print(f"--- ProXYZ Classification ---")
        print(f"Model:       {model_path}")
        print(f"Device:      {device}  dtype={dtype}  attn={attn_impl}")
        print(f"Vocab size:  {len(processor.tokenizer)}  bos={processor.tokenizer.bos_token_id} eos={processor.tokenizer.eos_token_id}")
        print(f"Prompt files:{','.join(args.prompt_data_files)}")
        print(f"Data format: {args.data_format}")

    # ==========================================
    # 3. ENCODE PROMPT & CLASSIFY IN BATCHES # ==========================================
    # Ban all tokens whose string representation contains "X" (unknown residue).
    # This prevents the model from emitting ambiguous amino-acid placeholders.

    # Flatten the batched iterators into one-sequence-per-example records.
    def data_generator():
        iterator = dataset.data_iterator(args.data_format)
        for batch in iterator(args.prompt_data_files):
            yield from batch

    # Apply tokenization
    def tokenize_dataset(prompt_dataset):
        return prompt_dataset.with_transform(
            functools.partial(
                data_utils.tokenize_function,
                processor,
                args.data_format,
                char_apply=model.config.has_characterization,
                max_length=args.max_sequence_length,
                generate=True,
            )
        )

    prompt_dataset = Dataset.from_generator(data_generator)
    prompt_dataset = tokenize_dataset(prompt_dataset)

    prompt_dataloader = DataLoader(
        prompt_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )

    # Normal generation mode
    sequences, features = [], []
    for input_ids in tqdm(prompt_dataloader):
        input_ids = {
            k: v.to(device) for k, v in input_ids.items() if torch.is_tensor(v)
        }
        input_ids = data_utils.prepare_inputs(processor, input_ids)
        with torch.no_grad():
            outputs = model(**input_ids, use_cache=False)
        features.append(
            {
                "cle_logits":
                    outputs.cle_logits.cpu()
                    if outputs.cle_logits is not None else None,
                "distogram_logits": (
                    outputs.distogram_logits.cpu()
                    if outputs.distogram_logits is not None else None
                ),
            }
        )

    # ==========================================
    # 4. WRITE EMBED OUTPUT
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
