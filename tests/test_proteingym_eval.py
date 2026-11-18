"""Smoke test: script_utils/proteingym_eval.py zero-shot VEP scoring.

Builds a tiny tokenizer/processor/model fixture and a fake ProteinGym
substitution assay, then checks:
  1. apply_mutations handles single/multi mutants and rejects WT mismatches.
  2. evaluate_assay scores every mutant exactly once (token-only and char models).
  3. Per-mutant scores are invariant to batch size (right-padding + ignore_index).
  4. run_benchmark writes per-assay Spearman matching a manual scipy computation.
  5. A 2-rank CPU (gloo) run shards, gathers and deduplicates to the same scores.

Run with:
    PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
        python tests/test_proteingym_eval.py
"""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd
from scipy import stats
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from proxyz.models import XYZConfig, XYZForCausalLM, XYZProcessor

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "script_utils",
    "proteingym_eval.py",
)

TINY = dict(
    hidden_size=16,
    intermediate_size=32,
    num_hidden_layers=1,
    num_attention_heads=2,
    num_key_value_heads=2,
    max_position_embeddings=128,
    attn_implementation="eager",
)
CHAR_TINY = dict(
    char_hidden_size=16,
    char_intermediate_size=32,
    char_num_hidden_layers=1,
    char_num_attention_heads=2,
    char_num_key_value_heads=2,
)

WT = "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK"
AA = "ACDEFGHIKLMNPQRSTVWY"


def load_script():
    spec = importlib.util.spec_from_file_location("proteingym_eval", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_processor():
    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=64,
        min_frequency=1,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"],
    )
    corpus = [
        "".join(AA[(i * 7 + j * 3) % len(AA)] for j in range(40))
        for i in range(64)
    ]
    tok.train_from_iterator(corpus, trainer)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="[BOS]",
        eos_token="[EOS]",
    )
    return XYZProcessor(tokenizer)


def make_model(processor, with_char):
    kwargs = dict(TINY, vocab_size=len(processor.tokenizer))
    if with_char:
        kwargs.update(CHAR_TINY, has_char_lm_head=True)
    torch.manual_seed(0)
    return XYZForCausalLM(XYZConfig(**kwargs)).eval()


def make_assay(tmpdir):
    rows = []
    for pos in (1, 5, 12, 27, len(WT)):
        for aa in ("A", "W"):
            if aa == WT[pos - 1]:
                continue
            rows.append({"mutant": f"{WT[pos - 1]}{pos}{aa}", "DMS_score": 0.0})
    rows.append({"mutant": f"M1A:{WT[11]}12W", "DMS_score": 0.0})
    rng = torch.Generator().manual_seed(42)
    for i, row in enumerate(rows):
        row["DMS_score"] = float(torch.rand(1, generator=rng).item())
    path = os.path.join(tmpdir, "assay.csv")
    pd.DataFrame(rows).to_csv(path, index=False)
    return path, rows


def check_apply_mutations(mod):
    assert mod.apply_mutations(WT, "M1A") == "A" + WT[1:]
    assert mod.apply_mutations(WT, f"M1A:{WT[11]}12W")[0] == "A"
    assert mod.apply_mutations(WT, f"M1A:{WT[11]}12W")[11] == "W"
    try:
        mod.apply_mutations(WT, "V1A")
    except AssertionError:
        pass
    else:
        raise AssertionError("WT mismatch not rejected")
    print("PASS: apply_mutations single/multi/WT-check")


def scores_by_id(results):
    by_id = {}
    for r in results:
        assert r["id"] not in by_id, f"duplicate id {r['id']}"
        by_id[r["id"]] = r
    return by_id


def check_assay(mod, accelerator, tmpdir, with_char):
    processor = make_processor()
    model = make_model(processor, with_char)
    assay_path, rows = make_assay(tmpdir)
    expected_ids = {r["mutant"] for r in rows}

    runs = {}
    for batch_size in (1, 4):
        results = mod.evaluate_assay(
            model, processor, WT, assay_path, accelerator, batch_size=batch_size
        )
        by_id = scores_by_id(results)
        assert set(by_id) == expected_ids, (set(by_id), expected_ids)
        for r in results:
            assert torch.isfinite(torch.tensor(r["Model_score"]))
            expected = next(
                x["DMS_score"] for x in rows if x["mutant"] == r["id"]
            )
            assert abs(r["DMS_score"] - expected) < 1e-6
            if with_char:
                assert "Model_score_char" in r and "Model_score_token" in r
            else:
                assert "Model_score_char" not in r
        runs[batch_size] = by_id

    for mut in expected_ids:
        a, b = runs[1][mut], runs[4][mut]
        for key in a:
            if key.startswith("Model_score"):
                assert abs(a[key] - b[key]) < 1e-5, (mut, key, a[key], b[key])
    kind = "char+token" if with_char else "token-only"
    print(f"PASS: {kind} assay scored once per mutant, batch-size invariant")
    return processor, model, assay_path, runs[1]


def check_benchmark(mod, accelerator, processor, model, assay_path, by_id):
    with tempfile.TemporaryDirectory() as tmpdir:
        out = os.path.join(tmpdir, "results.csv")
        table = pd.DataFrame(
            [
                {
                    "assay_id": "fake_assay",
                    "wt_sequence": WT,
                    "assay_csv_path": assay_path,
                }
            ]
        )
        mod.run_benchmark(
            model, processor, table, accelerator, batch_size=4, output_file=out
        )
        df = pd.read_csv(out)
        assert list(df["assay"]) == ["fake_assay"]
        ids = list(by_id)
        manual = stats.spearmanr(
            [by_id[i]["Model_score"] for i in ids],
            [by_id[i]["DMS_score"] for i in ids],
        ).statistic
        assert abs(df["spearman"].iloc[0] - manual) < 1e-9
        if "Model_score_char" in by_id[ids[0]]:
            for key in ("char", "token"):
                manual_k = stats.spearmanr(
                    [by_id[i][f"Model_score_{key}"] for i in ids],
                    [by_id[i]["DMS_score"] for i in ids],
                ).statistic
                assert abs(df[f"spearman_{key}"].iloc[0] - manual_k) < 1e-9
        print("PASS: run_benchmark Spearman matches manual computation, CSV written")


def check_distributed(reference_path):
    """2-rank gloo/CPU run: sharding + gather_object + deduplicate.

    Launched by main() via torchrun with CUDA_VISIBLE_DEVICES="" and
    PROTEINGYM_EVAL_DIST_REF pointing at single-process reference scores.
    """
    from accelerate import Accelerator
    from accelerate.utils import gather_object

    mod = load_script()
    accelerator = Accelerator()
    assert accelerator.num_processes == 2, accelerator.num_processes

    processor = make_processor()
    model = make_model(processor, with_char=False)
    with open(reference_path, encoding="utf-8") as f:
        reference = json.load(f)

    with tempfile.TemporaryDirectory() as tmpdir:
        assay_path, rows = make_assay(tmpdir)
        results = mod.evaluate_assay(
            model, processor, WT, assay_path, accelerator, batch_size=4
        )
        gathered = gather_object(results)
        unique = list(mod.data_utils.deduplicate(gathered))
        assert len(gathered) > len(rows), "even_batches padding never exercised"
        assert len(unique) == len(rows), (len(unique), len(rows))
        assert {r["id"] for r in unique} == set(reference)
        for r in unique:
            assert abs(r["Model_score"] - reference[r["id"]]["Model_score"]) < 1e-5

        out = os.path.join(tmpdir, "dist_results.csv")
        table = pd.DataFrame(
            [
                {
                    "assay_id": "fake_assay",
                    "wt_sequence": WT,
                    "assay_csv_path": assay_path,
                }
            ]
        )
        mod.run_benchmark(
            model, processor, table, accelerator, batch_size=4, output_file=out
        )
        if accelerator.is_main_process:
            df = pd.read_csv(out)
            ids = list(reference)
            expected = stats.spearmanr(
                [reference[i]["Model_score"] for i in ids],
                [reference[i]["DMS_score"] for i in ids],
            ).statistic
            assert abs(df["spearman"].iloc[0] - expected) < 1e-9
            print("PASS: 2-rank gather/dedup scores and Spearman match single-process")
    accelerator.wait_for_everyone()


def main():
    from accelerate import Accelerator

    if "PROTEINGYM_EVAL_DIST_REF" in os.environ:
        check_distributed(os.environ["PROTEINGYM_EVAL_DIST_REF"])
        return

    torch.manual_seed(0)
    mod = load_script()
    accelerator = Accelerator(cpu=True)

    check_apply_mutations(mod)
    with tempfile.TemporaryDirectory() as tmpdir:
        token_reference = None
        for with_char in (False, True):
            subdir = os.path.join(tmpdir, "char" if with_char else "token")
            os.makedirs(subdir)
            processor, model, assay_path, by_id = check_assay(
                mod, accelerator, subdir, with_char=with_char
            )
            check_benchmark(mod, accelerator, processor, model, assay_path, by_id)
            if not with_char:
                token_reference = by_id

        reference_path = os.path.join(tmpdir, "reference.json")
        with open(reference_path, "w", encoding="utf-8") as f:
            json.dump(token_reference, f)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["ACCELERATE_USE_CPU"] = "1"
        env["PROTEINGYM_EVAL_DIST_REF"] = reference_path
        env["PYTHONPATH"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
        )
        env["TOKENIZERS_PARALLELISM"] = "false"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=2",
                os.path.abspath(__file__),
            ],
            env=env,
            check=True,
        )

    print("All proteingym_eval checks passed.")


if __name__ == "__main__":
    main()