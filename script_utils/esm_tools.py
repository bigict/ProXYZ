import os
from datetime import datetime
import functools

from accelerate import Accelerator
from accelerate.utils import gather_object
import click
from datasets import Dataset
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, EsmForMaskedLM
from tqdm import tqdm

from proxyz.data import dataset, sampler
from proxyz.models import XYZProcessor
from proxyz.utils import data_utils, dict2object, structure_utils


@click.group()
def main():
    pass


@main.command("contact_precision", context_settings={"show_default": True})
@click.option(
    "--model_id",
    type=click.Choice(
        ["facebook/esm2_t33_650M_UR50D", "facebook/esm2_t36_3B_UR50D"]
    ),
    default="facebook/esm2_t33_650M_UR50D",
    help="Pretrained model id hosted inside a model repo on huggingface.co.",
)
@click.option(
    "--data_files", type=click.Path(), multiple=True, help="Data files to classify. "
)
@click.option(
    "--cluster_files",
    type=click.Path(),
    multiple=True,
    help="Clustering files for cluster-based sampling. Each file has two columns: "
    "cluster_id and data_row_id. Sampling weight = n / (1 + log(n)) where n is cluster size.",
)
@click.option(
    "--data_format",
    type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line",
    help="Input format: one sequence per line ('line'), FASTA ('fasta'), "
    "Foldcomp ('foldcomp') or PDB ('pdb/cif').",
)
@click.option(
    "--text_column",
    type=str,
    default="text",
    help="Column name containing the sequence text (default: 'text').",
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
@click.option(
    "--contact_ranges",
    type=click.Choice(structure_utils.contact_ranges.keys()),
    multiple=True,
    default=structure_utils.contact_ranges.keys(),
    help="Minimum separation distance to consider. We often want to measure contacts "
    "at a certain range. Typical ranges are short [6, 12], medium [12, 24], and long "
    "[24, inf)."
)
@click.option("--seed", type=int, default=42, help="Random seed for reproducibility.")
@click.option(
    "--attn_implementation",
    type=click.Choice(["flash_attention_2", "sdpa", "eager"]),
    default="flash_attention_2",
    help="Attention backend used for inference.",
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def contact_precision(**args):
    args = dict2object(**args)

    torch.manual_seed(args.seed)
    # 1. Initialize Accelerator
    accelerator = Accelerator()
    device = accelerator.device
    use_cuda = torch.cuda.is_available()

    # 2. Load the model and tokenizer (Must use MaskedLM architecture for contact prediction)
    # FlashAttention needs half precision; bf16 is also the fastest path on CUDA.
    dtype = torch.bfloat16 if use_cuda else torch.float32
    attn_impl = args.attn_implementation if use_cuda else "eager"
    dtype = torch.float32
    attn_impl = "eager"

    model = EsmForMaskedLM.from_pretrained(
        args.model_id, dtype=dtype, attn_implementation=attn_impl, trust_remote_code=True
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.bos_token = "<cls>"  # FIX: <cls> as <bos>
    features = ["distogram_labels"]
    processor = XYZProcessor(
        tokenizer=tokenizer, text_column=args.text_column, features=features
    )
    if args.verbose:
        print(f"--- ESM Classification ---")
        print(f"Model:       {args.model_id}")
        print(f"Device:      {device}  dtype={dtype}  attn={attn_impl}")
        print(f"Vocab size:  {len(processor.tokenizer)}  bos={processor.tokenizer.bos_token_id} eos={processor.tokenizer.eos_token_id}")
        print(f"Data files:  {','.join(args.data_files)}")
        print(f"Data format: {args.data_format}")

    # 3. Your sample protein sequences (Replace with your actual dataset)
    eval_dataset = data_utils.create_dataset(args.data_files, args.data_format)

    # Apply tokenization
    tokenize_function = functools.partial(
        data_utils.tokenize_function,
        processor,
        args.data_format,
        max_length=args.max_sequence_length,
    )
    eval_dataset = eval_dataset.with_transform(tokenize_function)

    # Load cluster information and compute sampling weights if cluster files provided
    eval_sampler = None
    if args.cluster_files:
        eval_sampler = sampler.from_cluster_files(eval_dataset, args.cluster_files)
        if args.verbose:
            print(f"--- Cluster-based sampling ---")
            print(f"Clusters: {len(eval_sampler):,}")

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=args.batch_size, sampler=eval_sampler, pin_memory=True
    )
    eval_dataloader = accelerator.prepare(eval_dataloader)

    # TODO: provided by args
    # distogram to contact: bin k covers (bins[k-1], bins[k]], so the last bin
    # fully within 8 A is the last boundary <= 8, i.e. count(bins <= 8) - 1.
    distogram_cutoff_idx = int((processor.distogram_bins <= 8).sum(-1)) - 1

    # 4. Use accelerator to distribute sequences across your GPUs
    results = []
    for input_ids in tqdm(
        eval_dataloader, desc="contact", disable=not accelerator.is_main_process
    ):
        input_ids = data_utils.prepare_inputs(processor, input_ids)
        
        # 5. Inference loop
        with torch.no_grad():
            # Extract the unsupervised contact map matrix
            # Outputs a matrix shape: [Batch, Sequence_Length, Sequence_Length]
            contact_matrix = model.predict_contacts(
                input_ids["input_ids"], input_ids["attention_mask"],
            )

        attention_mask_key = "char_attention_mask"
        if attention_mask_key not in input_ids:
            attention_mask_key = "attention_mask"
        assert attention_mask_key in input_ids

        if "distogram_labels" in input_ids:
            distogram_labels = input_ids["distogram_labels"]

            valid_length = (distogram_labels != processor.ignore_index).any(-1).sum(-1)

            contact_target = (distogram_labels <= distogram_cutoff_idx).where(
                distogram_labels != processor.ignore_index, processor.ignore_index
            )

            distogram_metrics = structure_utils.contact_precision(
                contact_matrix, contact_target, src_lengths=valid_length
            )
            for contact_range in args.contact_ranges:
                minsep, maxsep = structure_utils.contact_ranges[contact_range]
                distogram_metrics.update(
                    {
                        f"{key}/{contact_range}": value
                        for key, value in structure_utils.contact_precision(
                            contact_matrix,
                            contact_target,
                            src_lengths=valid_length,
                            minsep=minsep,
                            maxsep=maxsep,
                        ).items()
                    }
                )
        else:
            distogram_metrics = None

        batched_result = []
        for idx in range(len(input_ids["id"])):
            result = {
                "id": input_ids["id"][idx],
                "length": input_ids[attention_mask_key][idx].sum().item() - 2
            }
            if distogram_metrics is not None:
                result["valid"] = valid_length[idx].item()
                for key, value in distogram_metrics.items():
                    result[key] = value[idx].item()
            batched_result.append(result)

        # Gather python strings/objects across all GPUs safely. This returns a list of
        # all strings from all GPUs combined
        batched_result = gather_object(batched_result)
        if accelerator.is_main_process:
            results += batched_result

    # ==========================================
    # 4. WRITE EMBED OUTPUT
    # ==========================================
    if accelerator.is_main_process:
        # Drop the padded duplicates instead of positional truncation: they sit
        # at the end of each process stream, i.e. mid-list after gathering.
        results = list(data_utils.deduplicate(results))
        if len(results) != len(eval_dataset):
            print(
                f"Warning: classified {len(results)} of {len(eval_dataset)} sequences"
            )

        os.makedirs(args.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(args.output_dir, f"classify_{timestamp}.csv")
        df = pd.DataFrame(results)
        if args.verbose:
            print("Classification Summary:")
            print(df.describe())
        df.to_csv(output_path, index=False)
        print(f"Wrote {len(results)} results to {output_path}")

    # Clean up distributed process group to avoid resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
