from typing import Dict, List, Optional

import click

from transformers import AutoTokenizer, EsmForProteinFolding

from proxyz.utils import dict2object


def run_esmfold(
    sequences: List[str],
    path_to_esmfold_out: str,
    name: str,
    suffix: str,
    cache_dir: Optional[str] = None,
    keep_outputs: bool = False,
) -> List[str]:
    """
    Runs ESMFold on sequences and stores results as PDB files.

    For now, runs with a single GPU, though not a big deal if we parallelie jobs (easily
    done with our inference pipeline).

    Args:
        sequences: List of protein sequences to predict
        path_to_esmfold_out: Root directory to store outputs of ESMFold as PDBs
        name: name to use when storing
        suffix: to use as suffix when storing files
        cache_dir: Cache directory for model weights
        keep_outputs: Whether to keep output directories

    Returns:
        List of paths (list of str) to PDB files
    """
    is_cluster_run = os.environ.get("SLURM_JOB_ID") is not None

    # Use provided cache_dir or fallback to environment/cluster logic
    final_cache_dir = cache_dir
    if final_cache_dir is None and is_cluster_run:
        final_cache_dir = os.environ.get("CACHE_DIR")

    tokenizer = AutoTokenizer.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir
    )
    esm_model = EsmForProteinFolding.from_pretrained(
        "facebook/esmfold_v1", cache_dir=final_cache_dir
    )
    esm_model = esm_model.cuda()

    # Run ESMFold
    list_of_strings_pdb = []
    if len(sequences) == 8:
        max_nres = max([len(x) for x in sequences])
        if max_nres > 700:
            batch_size = 1
            num_batches = 8
        elif max_nres > 500:
            batch_size = 2
            num_batches = 4
        elif max_nres > 200:
            batch_size = 4
            num_batches = 2
        else:
            batch_size = 8
            num_batches = 1
    elif len(sequences) == 1:
        batch_size = 8
        num_batches = 1
    else:
        raise IOError(
            "We can only run ESMFold with 1 or 8 sequences... We should fix this..."
        )

    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        inputs = tokenizer(
            sequences[start_idx:end_idx],
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
        )
        inputs = {k: inputs[k].cuda() for k in inputs}

        with torch.no_grad():
            _outputs = esm_model(**inputs)

        _list_of_strings_pdb = _convert_esm_outputs_to_pdb(_outputs)
        list_of_strings_pdb.extend(_list_of_strings_pdb)

    # Create out directory if not there
    if not os.path.exists(path_to_esmfold_out):
        os.makedirs(path_to_esmfold_out)

    # Store generations for each sequence
    out_esm_paths = []
    for i, pdb in enumerate(list_of_strings_pdb):
        fname = f"esm_{i+1}.pdb_esm_{suffix}"
        fdir = os.path.join(path_to_esmfold_out, fname)
        with open(fdir, "w") as f:
            f.write(pdb)
            out_esm_paths.append(fdir)

    if not keep_outputs:
        # Clean up individual FASTA files directory
        try:
            shutil.rmtree(os.path.dirname(os.path.dirname(fdir)))
        except Exception as e:
            logger.warning(f"Could not clean up FASTA directory: {e}")

    return out_esm_paths


@click.command(context_settings={'show_default': True})
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def main(**args):
    args = dict2object(**args)


if __name__ == "__main__":
    main()
