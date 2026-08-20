from collections import defaultdict
import math
import pickle
from typing import Sequence

from datasets import Dataset
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from proxyz.data.utils import lines, opener


def from_cluster_files(
    dataset: Dataset, file_paths: Sequence[str], generator: torch.Generator = None
) -> Sampler:
    seqeuence_to_idx = {}
    for idx, example in enumerate(dataset):
        seqeuence_to_idx[example["id"]] = idx

    num_samples = 0

    # Assign weights to each sample
    sample_weights = [0.0] * len(dataset)  # disabled by default
    for cluster_id, data_row_ids in iter_cluster_map(file_paths):
        # Filtering
        data_row_ids = [
            data_row_id for data_row_id in data_row_ids if data_row_id in seqeuence_to_idx
        ]
        if data_row_ids:
            # Compute sampling weights: n / (1 + log(n)) for each cluster
            weight = 1 / (1 + math.log(len(data_row_ids)))
            for data_row_id in data_row_ids:
                sample_weights[seqeuence_to_idx[data_row_id]] = weight
            num_samples += 1

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=num_samples,  # FIX: use #cluster_size instead of #samples
        replacement=True,
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


    for file_path in file_paths:
        if file_path.endswith(".db"):
            yield from cluster_map_from_db(file_path)
        else:
            yield from cluster_map_from_file(file_path)
