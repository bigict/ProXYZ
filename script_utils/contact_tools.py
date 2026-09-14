import os
import functools

import click
from datasets import Dataset
import pandas as pd
from transformers import PreTrainedTokenizerFast
import torch
from tqdm import tqdm

from proxyz.data import dataset
from proxyz.data.utils import opener
from proxyz.models import XYZProcessor
from proxyz.utils import data_utils, dict2object, structure_utils


@click.command(context_settings={"show_default": True})
@click.argument("data_files", type=click.Path(), nargs=-1)
@click.option(
    "--data_format",
    type=click.Choice(["foldcomp", "pdb"]),
    default="pdb",
    help="Input format: one sequence per FOLDCOMP ('foldcomp') or PDB ('pdb/cif').",
)
@click.option(
    "--tokenizer_file",
    type=click.Path(),
    default="my_tokenizer.json",
    help="Path to the tokenizer json file.",
)
@click.option(
    "--text_column",
    type=str,
    default="text",
    help="Column name containing the sequence text (default: 'text').",
)
@click.option(
    "--output_file",
    type=click.Path(),
    default="-",
    help="Path to save csv file.",
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
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    features = ["distogram_labels"]

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

    # Load from local files
    iterator = dataset.data_iterator(args.data_format)

    # Flatten the batched iterators into one-sequence-per-example records.
    def data_generator(data_files):
        for batch in iterator(data_files):
            yield from batch

    eval_dataset = Dataset.from_generator(
        functools.partial(data_generator, args.data_files)
    )
    if args.verbose:
        print(f"--- Evaluation-dataset ---")
        print(f"Dataset: {len(eval_dataset):,}")

    # Apply tokenization
    def tokenize_dataset(dataset):
        return dataset.with_transform(
            functools.partial(
                data_utils.tokenize_function,
                processor,
                args.data_format,
                char_apply=True,
            )
        )

    eval_dataset = tokenize_dataset(eval_dataset)

    # distogram to contact
    distogram_cutoff_idx = int((processor.distogram_bins <= 8).sum(-1)) - 1

    results = []
    for data in tqdm(eval_dataset):
        data = data_utils.prepare_inputs(processor, data)
        contact_labels = (data["distogram_labels"] <= distogram_cutoff_idx).where(
            data["distogram_labels"] != processor.ignore_index, processor.ignore_index
        )
        attention_mask_key = "char_attention_mask"
        if attention_mask_key not in data:
            attention_mask_key = "attention_mask"
        assert attention_mask_key in data
        length = data[attention_mask_key].sum(-1).item()

        result = {"id": data["id"], "length": length}

        seqlen_range = torch.arange(length)
        sep = seqlen_range[None, :] - seqlen_range[:, None]
        for contact_range in args.contact_ranges:
            minsep, maxsep = structure_utils.contact_ranges[contact_range]
            valid_mask = (sep >= minsep) & (contact_labels != processor.ignore_index)
            if maxsep is not None:
                valid_mask &= sep < maxsep
            # upper-triangle mask counts each contacting pair exactly once
            contact = ((contact_labels == 1) & valid_mask).sum().item()
            result[f"contact_{contact_range}"] = contact

        results.append(result)

    df = pd.DataFrame(results)
    if args.verbose:
        print("Summary:")
        print(df.describe())
    with opener(args.output_file, "w") as f:
        df.to_csv(f, index=False)


if __name__ == "__main__":
    main()
