import os
from datetime import datetime
import functools

from accelerate import Accelerator
from accelerate.utils import gather_object
import click
from datasets import Dataset
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoProcessor)
from tqdm import tqdm

from proxyz.data import dataset
from proxyz.utils import data_utils, dict2object, model_utils, structure_utils


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Trained model dir. If it has no weights at the root, the latest "
    "checkpoint-* subdirectory is used automatically.",
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
@click.option("--batch_size", type=int, default=8, help="Sequences classified per batch.")
@click.option(
    "--output_dir",
    type=click.Path(),
    default="./classify_sequences",
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
    "--torch_compile",
    is_flag=True,
    help="Compiles PyTorch code to fused kernels to make it run faster.",
)
@click.option(
    "--save_features",
    is_flag=True,
    help="Run a forward pass over each sequence and save the auxiliary "
    "token-classification outputs (CLE logits, distogram logits) as a .pt file.",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    torch.manual_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device
    use_cuda = torch.cuda.is_available()

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
        dtype=dtype,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model = model.to(device).eval()

    if args.torch_compile:
        model = torch.compile(model)

    if args.verbose:
        print(f"--- ProXYZ Classification ---")
        print(f"Model:       {model_path}")
        print(f"Device:      {device}  dtype={dtype}  attn={attn_impl}")
        print(f"Vocab size:  {len(processor.tokenizer)}  bos={processor.tokenizer.bos_token_id} eos={processor.tokenizer.eos_token_id}")
        print(f"Data files:  {','.join(args.data_files)}")
        print(f"Data format: {args.data_format}")

    # ==========================================
    # 3. ENCODE PROMPT & CLASSIFY IN BATCHES # ==========================================
    # Ban all tokens whose string representation contains "X" (unknown residue).
    # This prevents the model from emitting ambiguous amino-acid placeholders.

    # Flatten the batched iterators into one-sequence-per-example records.
    def data_generator():
        iterator = dataset.data_iterator(args.data_format)
        for batch in iterator(args.data_files):
            yield from batch

    # Apply tokenization
    def tokenize_dataset(eval_dataset):
        return eval_dataset.with_transform(
            functools.partial(
                data_utils.tokenize_function,
                processor,
                args.data_format,
                char_apply=model.config.has_characterization,
                max_length=args.max_sequence_length,
            )
        )

    eval_dataset = Dataset.from_generator(data_generator)
    eval_dataset = tokenize_dataset(eval_dataset)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # TODO: provided by args
    # distogram to contact
    distogram_cutoff_idx = int((processor.distogram_bins <= 8).sum(-1))

    # Normal clasification mode
    results, features = [], []
    for input_ids in tqdm(eval_dataloader, desc="classification"):
        input_ids = {
            k: v.to(device) if torch.is_tensor(v) else v for k, v in input_ids.items()
        }
        input_ids = data_utils.prepare_inputs(processor, input_ids)
        with torch.no_grad():
            outputs = model(**input_ids, use_cache=False)

        if outputs.distogram_logits is not None and "distogram_labels" in input_ids:
            distogram_metrics = contact_precision(
                outputs.distogram_logits,
                input_ids["distogram_labels"],
                distogram_cutoff_idx,
                lengths=input_ids["attention_mask"].sum(-1),
                ignore_index=processor.ignore_index,
            )
        else:
            distogram_metrics = None

        batched_result = []
        for idx in range(len(input_ids["id"])):
            result = {
                "id": input_ids["id"][idx],
                "lenght": input_ids["attention_mask"][idx].sum().item()
            }
            if distogram_metrics is not None:
                for key, value in distogram_metrics.items():
                    result[key] = value[idx].item()
            batched_result.append(result)

        # Gather python strings/objects across all GPUs safely. This returns a list of
        # all strings from all GPUs combined
        batched_result = gather_object(batched_result)
        if accelerator.is_main_process:
            results += batched_result

        if args.save_features:
            batched_feature = []
            for idx in range(len(input_ids["id"])):
                feat = {
                    "id": input_ids["id"][idx], "mask": input_ids["attention_mask"][idx]
                }
                if outputs.cle_logits is not None:
                    feat["cle_logits"] = outputs.cle_logits[idx].cpu()
                if outputs.distogram_logits is not None:
                    feat["distogram_logits"] = outputs.distogram_logits[idx].cpu()
                batched_feature.append(feat)

            batched_feature = gather_object(batched_feature)
            if accelerator.is_main_process:
                features += batched_feature

    # ==========================================
    # 4. WRITE EMBED OUTPUT
    # ==========================================
    if accelerator.is_main_process:
        # Truncate the padded items to match your exact original dataset size
        total_samples = len(eval_dataset)

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if features and args.save_features:
            output_path = os.path.join(args.output_dir, f"classify_{timestamp}.pt")

            with open(output_path, "wb") as f:
                torch.save(features[:total_samples], f)

            print(f"Wrote {len(features)} features to {output_path}")

        output_path = os.path.join(args.output_dir, f"classify_{timestamp}.csv")
        df = pd.DataFrame(results[:total_samples])
        if args.verbose:
            print("Classification Summary:")
            print(df.describe())
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(results)} results to {output_path}")

    # Clean up distributed process group to avoid resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def contact_precision(
    distogram_logits: torch.FloatTensor,
    distogram_labels: torch.LongTensor,
    distogram_cutoff_idx: int,
    lengths: torch.Tensor | None = None,
    ignore_index: int = -100,
) -> torch.FloatTensor:
    predictions = F.softmax(distogram_logits, dim=-1)
    predictions = predictions[..., : distogram_cutoff_idx + 1].sum(-1)
    targets = (distogram_labels <= distogram_cutoff_idx).where(
        distogram_labels != ignore_index, ignore_index
    )
    # lengths = (targets != ignore_index).any(-1).sum(-1) + 2  # [BOS] + [EOS]

    return structure_utils.contact_precision(predictions, targets, lengths)


if __name__ == "__main__":
    main()
