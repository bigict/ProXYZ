from array import array
import bisect
import itertools
import pathlib
import random
from typing import Literal, Sequence
from urllib.parse import urlparse

from Bio.Data.PDBData import protein_letters_3to1
from biotite.structure import alphabet
from datasets import Dataset
import torch
from tqdm import tqdm

from proxyz.data.utils import lines, opener, semaphore
from proxyz.utils import cache, env


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
    return "\n".join(seq[i:i + width] for i in range(0, len(seq), width))


def fasta_iterator(file_paths: Sequence[str], batch_size: int = 64):
    batch = []
    for file_path in tqdm(file_paths, desc="fasta_iterator"):
        for description, text in fasta_parse(file_path):
            if len(batch) >= batch_size:
                yield batch
                batch = []

            batch.append({"id": description, "text": text})
    if batch:
        yield batch


def pdb_iterator(file_paths: Sequence[str], batch_size: int = 64):
    for file_path in tqdm(file_paths, desc="pdb_iterator"):
        data = Dataset.from_csv(file_path)
        for batch in data.iter(batch_size=batch_size):
            batch["dataset"] = [file_path] * len(batch["id"])
            yield [dict(zip(batch, v)) for v in zip(*batch.values())]


def pdb_transform(examples: dict):
    batch = []
    for pid, file_path in zip(examples["id"], examples["dataset"]):
        processed_dir = pathlib.Path(file_path).parent / "processed"
        graph = torch.load(processed_dir / f"{pid}.pt", weights_only=False)
        batch.append(graph)
    return pyg_transform(batch)


@cache
def foldcomp_dataset(file_path: str):
    from graphein.ml.datasets.foldcomp_dataset import FoldCompDataset as FCDatasetBase
    from loguru import logger as log

    class FoldCompDataset(FCDatasetBase):
        def _get_indices(self):
            """Get indices for the dataset."""
            # Read in look up file
            LOOKUP_FILE = pathlib.Path(self.root) / f"{self.database}.lookup"
            if not LOOKUP_FILE.exists():
                self.download()
            with open(LOOKUP_FILE, "r") as f:
                accessions = f.readlines()
            # Extract accessions
            accessions = [x.strip().split("\t")[1] for x in tqdm(accessions)]
            # Get indices
            if self.ids is None:
                self.ids = accessions
            # Exclude indices
            if self.exclude_ids is not None:
                log.info(f"Excluding {len(self.exclude_ids)} chains...")
                self.ids = [
                    acc for acc in tqdm(self.ids) if acc not in self.exclude_ids
                ]
            # Sub sample
            if self.fraction < 1:
                log.info(f"Sampling fraction: {self.fraction}...")
                self.ids = random.sample(
                    self.ids, int(len(self.ids) * self.fraction)
                )
            log.info("Creating index...")
            # indices = dict(enumerate(accessions))
            # self.idx_to_protein = indices
            self.idx_to_protein = accessions
            # self.protein_to_idx = {v: k for k, v in indices.items()}
            self.protein_to_idx = array(
                "I",
                sorted(
                    range(len(accessions)), key=lambda idx: self.idx_to_protein[idx]
                )
            )
            log.info(f"Dataset contains {len(self.protein_to_idx)} chains.")

        def process(self):
            ids = self.ids

            self.ids = None  # Trigger to load the whole db
            super().process()

            self.ids = ids

        def len(self) -> int:
            """Returns length of the dataset"""
            return len(self.ids)

        def get(self, idx):
            """Retrieves a protein from the dataset. Can idx on either the protein
            ID or its index."""
            if isinstance(idx, int):
                idx = self.ids[idx]
            idx = self.protein_to_idx[
                bisect.bisect_left(
                    self.protein_to_idx, idx, key=lambda x: self.idx_to_protein[x]
                )
            ]
            return super().get(idx)


    o = urlparse(file_path)
    if o.fragment:
        with opener(o.fragment) as f:
            ids = [line for line in lines(f) if not line.startswith("#")]
    else:
        ids = None
    file_path = pathlib.Path(o.path)
    with semaphore(
        "proxyz_dataset_foldcomp_parallel", env("proxyz_dataset_foldcomp_parallel", -1)
    ):
        return FoldCompDataset(
            root=file_path.parent,
            database=file_path.name,  # name of the dataset. See: https://github.com/steineggerlab/foldcomp
            ids=ids,
            fraction=1,
        )


def foldcomp_iterator(file_paths: Sequence[str], batch_size: int = 64):
    batch = []

    for file_path in tqdm(file_paths, desc="foldcomp_iterator"):
        data = foldcomp_dataset(file_path, use_cache=False)
        for pid in data.ids:
            if len(batch) >= batch_size:
                yield batch
                batch = []
            batch.append({"id": pid, "dataset": file_path})
        del data  # gc

    if batch:
        yield batch

def foldcomp_transform(examples: dict):
    batch = []
    for pid, file_path in zip(examples["id"], examples["dataset"]):
        # load from foldcomp db
        data = foldcomp_dataset(file_path)
        graph = data.get(pid)
        assert pid == graph.id

        if not hasattr(graph, "coord_mask"):
            graph.coord_mask = (graph.coords != graph.fill_value)[..., 0]
        if not hasattr(graph, "residue_pdb_idx"):
            graph.residue_pdb_idx = torch.tensor(
                [int(s.split(":")[2]) for s in graph.residue_id], dtype=torch.long
            )
        batch.append(graph)
    return pyg_transform(batch)


def pyg_transform(batch: list) -> dict:
    coord, coord_mask, residue_idx, seq = [], [], [], []
    cle, pseudo_beta, pseudo_beta_mask = [], [], []

    for graph in batch:
        coord.append(graph.coords)
        coord_mask.append(graph.coord_mask)
        residue_idx.append(graph.residue_pdb_idx - graph.residue_pdb_idx[0])
        seq.append("".join(protein_letters_3to1.get(r, "A") for r in graph.residues))

        # atom indices
        n_idx, ca_idx, c_idx, cb_idx = 0, 1, 2, 3

        # pseudo_beta
        is_gly = (graph.residue_type == 7)
        pseudo_beta.append(
            torch.where(
                is_gly[:, None], graph.coords[:, ca_idx, :], graph.coords[:, cb_idx, :]
            )
        )
        pseudo_beta_mask.append(
            torch.where(
                is_gly, graph.coord_mask[:, ca_idx], graph.coord_mask[:, cb_idx]
            )
        )

        # 3di
        nan = torch.full((3, ), torch.nan)
        bbxyz = torch.stack(
            (
                torch.where(
                    graph.coord_mask[:, ca_idx, None], graph.coords[:, ca_idx, :], nan
                ),
                torch.where(
                    graph.coord_mask[:, cb_idx, None], graph.coords[:, cb_idx, :], nan
                ),
                torch.where(
                    graph.coord_mask[:,  n_idx, None], graph.coords[:,  n_idx, :], nan
                ),
                torch.where(
                    graph.coord_mask[:,  c_idx, None], graph.coords[:,  c_idx, :], nan
                ),
            )
        )
        cle.append(
            torch.from_numpy(
                alphabet.i3d.Encoder().encode(*bbxyz.numpy()).filled()
            ).long()
        )

    return {
        "text": seq,
        "residue_idx": residue_idx,
        "coord": coord,
        "coord_mask": coord_mask,
        "distogram_labels": (pseudo_beta, pseudo_beta_mask),
        "cle_labels": cle,
    }


def data_iterator(data_format: Literal["line", "fasta", "foldcomp", "pdb"]):
    iterators = {
        "line": line_iterator,
        "fasta": fasta_iterator,
        "foldcomp": foldcomp_iterator,
        "pdb": pdb_iterator,
    }
    return iterators[data_format]


def data_transform(data_format: Literal["line", "fasta", "foldcomp", "pdb"]):
    transform = {
        "foldcomp": foldcomp_transform,
        "pdb": pdb_transform,
    }
    return transform.get(data_format)
