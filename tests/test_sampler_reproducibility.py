"""Smoke test: cluster-based eval sampling in classify.py is seed-reproducible.

Run with:
    PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
        python tests/test_sampler_reproducibility.py
"""

import os
import tempfile

import torch
from datasets import Dataset

from proxyz.data import sampler


def make_fixture(tmpdir, n_seqs=64, n_clusters=8):
    ids = [f"seq_{i:03d}" for i in range(n_seqs)]
    dataset = Dataset.from_dict({"id": ids})
    cluster_file = os.path.join(tmpdir, "clusters.tsv")
    per_cluster = n_seqs // n_clusters
    with open(cluster_file, "w", encoding="utf-8") as f:
        for c in range(n_clusters):
            rows = ids[c * per_cluster : (c + 1) * per_cluster]
            f.write("\t".join([f"cluster_{c}"] + rows) + "\n")
    return dataset, cluster_file, n_clusters


def draw(dataset, cluster_file, seed, consume_global_rng=False):
    generator = torch.Generator().manual_seed(seed)
    s = sampler.from_cluster_files(dataset, [cluster_file], generator=generator)
    if consume_global_rng:
        torch.rand(1000)
    return list(s)


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset, cluster_file, n_clusters = make_fixture(tmpdir)

        # 1. Same seed -> identical sampling order (the classify.py pattern).
        baseline = draw(dataset, cluster_file, seed=42)
        assert len(baseline) == n_clusters, (len(baseline), n_clusters)
        assert baseline == draw(dataset, cluster_file, seed=42)
        print("PASS: same seed reproduces identical sampling order")

        # 2. Order is independent of global RNG consumption (model loading,
        #    accelerate.prepare, etc. may consume the global RNG).
        assert baseline == draw(dataset, cluster_file, seed=42, consume_global_rng=True)
        print("PASS: sampling order unaffected by global RNG consumption")

        # 3. Different seed -> different order (sanity: seed is actually used).
        assert baseline != draw(dataset, cluster_file, seed=43)
        print("PASS: different seed changes sampling order")

        # 4. Regression demo: without a dedicated generator (old behavior),
        #    global RNG consumption changes the order.
        torch.manual_seed(42)
        old_a = list(sampler.from_cluster_files(dataset, [cluster_file]))
        torch.manual_seed(42)
        torch.rand(1000)
        old_b = list(sampler.from_cluster_files(dataset, [cluster_file]))
        assert old_a != old_b
        print("PASS: confirmed old generator=None behavior was not reproducible")

        # 5. Regression: chunked path with zero-count chunks used to crash
        # ("cannot sample n_sample <= 0 samples"). 8 draws over 8 chunks leave
        # some chunk unselected for this seed.
        chunk_size = 8
        os.environ["proxyz_dataset_sampler_chunk_size"] = str(chunk_size)
        try:
            chunked = draw(dataset, cluster_file, seed=42)
            assert len(chunked) == n_clusters
            assert len({i // chunk_size for i in chunked}) < n_clusters
            assert chunked == draw(dataset, cluster_file, seed=42)
            assert chunked == draw(
                dataset, cluster_file, seed=42, consume_global_rng=True
            )
        finally:
            del os.environ["proxyz_dataset_sampler_chunk_size"]
        print("PASS: chunked path skips zero-count chunks, seed-reproducible")

        # 6. Dense chunked fixture (num_samples >> num_chunks) exercises the
        # within-chunk replacement flip; also seed-reproducible.
        dense_dataset, dense_file, dense_clusters = make_fixture(
            tmpdir, n_seqs=64, n_clusters=64
        )
        os.environ["proxyz_dataset_sampler_chunk_size"] = "16"
        try:
            chunked = draw(dense_dataset, dense_file, seed=42)
            assert len(chunked) == dense_clusters
            assert chunked == draw(dense_dataset, dense_file, seed=42)
            assert chunked == draw(
                dense_dataset, dense_file, seed=42, consume_global_rng=True
            )
        finally:
            del os.environ["proxyz_dataset_sampler_chunk_size"]
        print("PASS: dense chunked path is seed-reproducible")

    print("All sampler reproducibility checks passed.")


if __name__ == "__main__":
    main()