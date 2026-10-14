from array import array
from collections import defaultdict
from collections.abc import Iterator
import math
import pickle
from typing import Literal, Sequence

from datasets import Dataset
import torch
from torch.utils.data import Sampler, WeightedRandomSampler
from tqdm import tqdm

from proxyz.data.utils import lines, opener
from proxyz.utils import env


# FIX: number of categories cannot exceed 2^24
class HierarchicalWeightedRandomSampler(WeightedRandomSampler):
    def __iter__(self) -> Iterator[int]:
        chunk_size = env("proxyz_dataset_sampler_chunk_size", 1<<23)
        if len(self.weights) <= chunk_size:
            yield from super().__iter__()
        else:
            # Split to chunks
            rows = self.weights.split(chunk_size)

            # Sample which chunks to draw from
            cnt = torch.bincount(
                torch.multinomial(
                    torch.stack([row.sum() for row in rows]),
                    self.num_samples,
                    replacement=True,
                    generator=self.generator
                ), minlength=len(rows)
            )

            # Sample within each chunk
            for idx, row in enumerate(rows):
                replacement = self.replacement if cnt[idx] <= row.size(0) else True
                col_indices = torch.multinomial(
                    row + 1e-8, cnt[idx].item(), replacement, generator=self.generator
                )
                rand_indices = array("I", idx * chunk_size + col_indices)
                yield from iter(rand_indices)


def from_cluster_files(
    dataset: Dataset, file_paths: Sequence[str], generator: torch.Generator = None
) -> Sampler:
    def reweighting(
        data_row_ids: Sequence[str],
        method: Literal["inverse_log", "log"],
        power: float = 1.0,
    ) -> float:
        assert len(data_row_ids) > 0
        if method == "inverse_log":
            # Compute sampling weights: n / (1 + log(n)) for each cluster
            w = len(data_row_ids) / (1 + math.log(len(data_row_ids)))
        elif method == "log":
            # Compute sampling weights: (1 + log(n)) for each cluster
            w = (1 + math.log(len(data_row_ids)))
        else:
            assert False, "Only `inverse_log` and `log` reweighting is supported."
        return math.pow(w, power)

    # Incredibly fast: Extracts the column directly as a PyArrow ChunkedArray
    seqeuence_to_idx = {
        seq: idx for idx, seq in enumerate(dataset.data["id"].to_pylist())
    }

    num_samples = 0

    # Assign weights to each sample
    sample_weights = [0.0] * len(dataset)  # disabled by default
    for cluster_id, data_row_ids in iter_cluster_map(file_paths):
        # Filtering
        data_row_ids = [
            data_row_id for data_row_id in data_row_ids if data_row_id in seqeuence_to_idx
        ]
        if data_row_ids:
            weight = reweighting(
                data_row_ids=data_row_ids,
                method=env("proxyz_dataset_sampler_reweighting_method", "log"),
                power=env("proxyz_dataset_sampler_reweighting_power", 1.0),
            ) / len(data_row_ids)
            for data_row_id in data_row_ids:
                sample_weights[seqeuence_to_idx[data_row_id]] = weight
            num_samples += 1

    # Create WeightedRandomSampler
    sampler = HierarchicalWeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,  # FIX: use #cluster_size instead of #samples
        replacement=False,
        generator=generator,
    )
    return sampler


def iter_cluster_map(file_paths: Sequence[str]):
    def cluster_map_from_file(file_path):
        # Load the cluster file and build mapping: data_row_id -> cluster_id
        cluster_map = defaultdict(list)  # cluster_id -> [data_row_id]

        with opener(file_path) as f:
            for line in lines(f):
                cluster_id, *data_row_ids = line.split("\t")
                cluster_map[cluster_id] += data_row_ids

        yield from cluster_map.items()

    def cluster_map_from_db(file_path):
        import lmdb

        with lmdb.open(file_path, readonly=True, lock=False, max_dbs=1) as env:
            db = env.open_db(b"cluster_map", dupsort=True)
            with env.begin() as txn:
                cursor = txn.cursor(db=db)
                if cursor.first():
                    while True:
                        key = pickle.loads(cursor.key())
                        values = [
                            pickle.loads(value) for value in cursor.iternext_dup()
                        ]
                        yield key, values

                        if not cursor.next_nodup():
                            break

    for file_path in tqdm(file_paths, desc="iter_cluster_map"):
        if file_path.endswith(".db"):
            yield from cluster_map_from_db(file_path)
        else:
            yield from cluster_map_from_file(file_path)
