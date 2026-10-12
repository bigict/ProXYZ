"""
proteingym_eval.py
Zero-shot VEP on ProteinGym via ProXYZ's native next-char/token head.
"""
import os
from datetime import timedelta
import functools

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs, gather_object
import click
from datasets import Dataset
import pandas as pd
from scipy import stats
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor
from tqdm import tqdm

from proxyz.utils import data_utils, dict2object, model_utils
from proxyz.data.utils import opener


def apply_mutations(wt_seq, mutant_str):
    """Apply 'A12V' or 'A12V:M30K' (1-based positions) to the WT sequence."""
    seq = list(wt_seq)
    for mut in mutant_str.split(":"):
        wt_aa, pos, mut_aa = mut[0], int(mut[1:-1]), mut[-1]
        assert seq[pos - 1] == wt_aa, f"WT mismatch at {pos}: {seq[pos-1]} != {wt_aa}"
        seq[pos - 1] = mut_aa
    return "".join(seq)


def dataset_transform(processor, examples, char_apply=True):
    tokenized = data_utils.tokenize_function(
        processor, "line", examples, char_apply=char_apply
    )
    tokenized["DMS_score"] = examples["DMS_score"]
    return tokenized


def evaluate_assay(model, processor, wt_seq, assay_csv, accelerator, batch_size=1):
    """Per-assay Spearman between model scores and DMS fitness."""
    df = pd.read_csv(assay_csv)  # columns: mutant, DMS_score

    df["id"] = df["mutant"]
    df[processor.text_column] = df["mutant"].apply(
        lambda mut: apply_mutations(wt_seq, mut)
    )

    df_dataset = Dataset.from_pandas(df)
    df_dataset = df_dataset.with_transform(
        functools.partial(
            dataset_transform, processor, char_apply=model.config.has_characterization
        )
    )
    df_dataloader = DataLoader(
        df_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=1
    )
    df_dataloader = accelerator.prepare(df_dataloader)

    model_scores = []

    loss_fn = nn.CrossEntropyLoss(ignore_index=processor.ignore_index, reduction="mean")
    for input_ids in tqdm(
        df_dataloader, desc=os.path.basename(assay_csv), disable=not accelerator.is_main_process
    ):
        # input_ids = {
        #     k: v.to(device) if torch.is_tensor(v) else v for k, v in input_ids.items()
        # }
        input_ids = data_utils.prepare_inputs(processor, input_ids)

        with torch.no_grad():
            outputs = model(**input_ids, use_cache=False)

        for idx in range(len(input_ids["id"])):
            result = {
                "id": input_ids["id"][idx],
                "DMS_score": input_ids["DMS_score"][idx].item(),
            }
            loss = loss_fn(outputs.logits[idx][:-1, :], input_ids["labels"][idx][1:])
            if outputs.char_logits is not None:
                char_loss = loss_fn(
                    outputs.char_logits[idx][:-1, :], input_ids["char_labels"][idx][1:]
                )
                result["Model_score_char"] = -char_loss.item()
                result["Model_score_token"] = -loss.item()
                loss += char_loss
            result["Model_score"] = -loss.item()
            model_scores.append(result)

    return model_scores
    df = pd.DataFrame(model_scores)

    r = {"spearman": stats.spearmanr(df["Model_score"], df["DMS_score"]).statistic}
    for key in ("char", "token"):
        column = f"Model_score_{key}"
        if column in df:
            r[f"spearman_{key}"] = stats.spearmanr(
                df[column], df["DMS_score"]
            ).statistic
    return r


def run_benchmark(
    model, processor, assay_table, accelerator, batch_size=1, output_file=None
):
    """
    assay_table: csv/tsv with columns [assay_id, assay_csv_path, wt_sequence]
    (export from official ProteinGym metadata).
    Returns per-assay Spearman table + mean.
    """
    rows = []

    for idx, r in tqdm(
        assay_table.iterrows(), total=len(assay_table), disable=not accelerator.is_main_process
    ):
        model_scores = evaluate_assay(
            model,
            processor,
            r.wt_sequence,
            r.assay_csv_path,
            accelerator,
            batch_size=batch_size,
        )
        model_scores = gather_object(model_scores)
        if accelerator.is_main_process:
            df = pd.DataFrame(data_utils.deduplicate(model_scores))

            rho = {
                "spearman": stats.spearmanr(
                    df["Model_score"], df["DMS_score"]
                ).statistic
            }
            for key in ("char", "token"):
                column = f"Model_score_{key}"
                if column in df:
                    rho[f"spearman_{key}"] = stats.spearmanr(
                        df[column], df["DMS_score"]
                    ).statistic
            print(
                f"{r.assay_id}: " + ", ".join([f"{k}: {v:.3f}" for k, v in rho.items()])
            )
            rho["assay"] = r.assay_id

            rows.append(rho)

        # Optional: explicitly sync before moving to the next task loop
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        df = pd.DataFrame(rows)
        print(df.describe())
        if output_file is not None:
            with opener(output_file, "w") as f:
                df.to_csv(f, index=False)


@click.command(context_settings={"show_default": True})
@click.option(
    "--model_dir",
    type=click.Path(),
    default="./deepseek_style_model",
    help="Trained model dir. If it has no weights at the root, the latest "
    "checkpoint-* subdirectory is used automatically.",
)
@click.option(
    "--DMS_reference_file_path",
    type=click.Path(),
    help="Reference file with list of DMSs to consider."
)
@click.option(
    "--DMS_data_folder",
    type=click.Path(),
    default=os.path.join(
        os.path.expanduser("~"), ".cache", "ProteinGym", "DMS_ProteinGym_substitutions"
    ),
    help="Path to folder that contains all DMS datasets."
)
@click.option(
    "--batch_size", type=int, default=8, help="Sequences classified per batch."
)
@click.option(
    "--output_file",
    type=click.Path(),
    default="-",
    help="File where the results are written.",
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
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)

    torch.manual_seed(args.seed)
    timeout_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(kwargs_handlers=[timeout_kwargs])
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

    dms_mapping = pd.read_csv(args.dms_reference_file_path)
    dms_mapping["MSA_Neff_L_category"] = dms_mapping["MSA_Neff_L_category"].apply(
        lambda x: x[0].upper() + x[1:] if isinstance(x, str) else x
    )
    num_DMS = len(dms_mapping)
    print(f"There are {num_DMS} DMSs in mapping file")

    dms_mapping["assay_id"] = dms_mapping["DMS_id"]
    dms_mapping["wt_sequence"] = dms_mapping["target_seq"]
    dms_mapping["assay_csv_path"] = dms_mapping["DMS_filename"].apply(
        lambda x: os.path.join(args.dms_data_folder, x)
    )

    run_benchmark(
        model,
        processor,
        dms_mapping,
        accelerator,
        batch_size=args.batch_size,
        output_file=args.output_file,
    )

    # Clean up distributed process group to avoid resource leaks
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    main()
