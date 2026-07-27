import itertools
import pathlib
from typing import Literal, Sequence, Union

from Bio.Data.PDBData import protein_letters_3to1
from datasets import Dataset
import torch

from proxyz.data.utils import lines, opener


FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_TOKENS = [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]


def line_iterator(file_paths: Sequence[str], batch_size=64):
    batch = []

    i = 0
    for file_path in file_paths:
        with opener(file_path) as f:
            for line in lines(f):
                batch.append({"id": i, "text": line})
                i += 1
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def fasta_parse(file_path: str):
    with opener(file_path) as f:
        description, text = "", ""
        for line in lines(f):
            if line.startswith(">"):
                if text:
                    yield description, text
                description, text = line[1:], ""
            else:
                text += line
        if text:
            yield description, text


def fasta_wrap(seq: str, width: int = 60) -> str:
    """Wrap a sequence string to FASTA line width."""
    return "\n".join(seq[i : i + width] for i in range(0, len(seq), width))


def fasta_iterator(file_paths: Sequence[str], batch_size: int = 64):
    batch = []
    for file_path in file_paths:
        for description, text in fasta_parse(file_path):
            if len(batch) >= batch_size:
                yield batch
                batch = []

            batch.append({"id": description, "text": text})
    if batch:
        yield batch


def pdb_iterator(file_paths: Sequence[str], batch_size: int = 64):
    for file_path in file_paths:
        data = Dataset.from_csv(file_path)
        for batch in data.iter(batch_size=batch_size):
            yield [dict(zip(batch, v)) for v in zip(*batch.values())]


def pdb_transform(data_dir: Union[pathlib.Path, str], examples: dict):
    processed_dir = pathlib.Path(data_dir) / "processed"
    assert "id" in examples

    coord, coord_mask, residue_idx, seq = [], [], [], []
    for pid in examples["id"]:
        graph = torch.load(processed_dir / f"{pid}.pt", weights_only=False)
        coord.append(graph.coords)
        coord_mask.append(graph.coord_mask)
        residue_idx.append(graph.residue_pdb_idx - graph.residue_pdb_idx[0])
        seq.append("".join(protein_letters_3to1[r] for r in graph.residues))

    return {
        "text": seq,
        "residue_idx": residue_idx,
        "coord": coord,
        "coord_mask": coord_mask
    }


def data_iterator(data_format: Literal["line", "fasta", "pdb"]):
    iterators = {
        "line": line_iterator,
        "fasta": fasta_iterator,
        "pdb": pdb_iterator,
    }
    return iterators[data_format]
