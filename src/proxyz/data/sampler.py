from collections import defaultdict
import math
from typing import Sequence

from datasets import Dataset
import torch
from torch.utils.data import Sampler, WeightedRandomSampler

from proxyz.data.utils import lines, opener


class DistributedSamplerWrapper(Sampler):
    """
    Wraps a non-distributed Sampler (like WeightedRandomSampler) 
    to make it compatible with Distributed Data Parallel (DDP).
    """
    def __init__(
        self,
        sampler: Sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = None,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        # Automatically detect DDP environment details if not provided
        if num_replicas is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available() or not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
                
        self.sampler = sampler
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.sampler) % self.num_replicas != 0:
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.sampler) - self.num_replicas) / self.num_replicas
            )
        else:
            self.num_samples = math.ceil(len(self.sampler) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        # Get the underlying sample order from the WeightedRandomSampler
        indices = list(self.sampler)

        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            # Shuffle the underlying indices globally before splitting
            indices = [indices[i] for i in torch.randperm(len(indices), generator=g).tolist()]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        if len(indices) != self.total_size:
            raise AssertionError(
                f"Number of indices ({len(indices)}) does not match total_size ({self.total_size})"
            )

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise AssertionError(
                f"Number of subsampled indices ({len(indices)}) does not match num_samples ({self.num_samples})"
            )

        # pyrefly: ignore [bad-return]
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def from_cluster_files(
    dataset: Dataset, file_paths: Sequence[str], generator: torch.Generator = None
):
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
    sample_weights = [1.0] * len(dataset)
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
