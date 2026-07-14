from collections import defaultdict
import math
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

    # Load all cluster files and build mapping: data_row_id -> cluster_id
    cluster_map = defaultdict(list)  # cluster_id -> [data_row_id]
    for file_path in file_paths:
        with opener(file_path) as f:
            for line in lines(f):
                cluster_id, *data_row_ids = line.split("\t")
                cluster_map[cluster_id] += data_row_ids

    # Assign weights to each sample
    sample_weights = [0.0] * len(dataset)  # disabled by default
    for cluster_id, data_row_ids in cluster_map.items():
        # Compute sampling weights: n / (1 + log(n)) for each cluster
        weight = 1 / (1 + math.log(len(data_row_ids)))
        for data_row_id in data_row_ids:
            if data_row_id in seqeuence_to_idx:
                sample_weights[seqeuence_to_idx[data_row_id]] = weight

    # Create WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(cluster_map),  # FIX: use #cluster_size instead of #samples
        replacement=True,
        generator=generator,
    )
    return sampler
