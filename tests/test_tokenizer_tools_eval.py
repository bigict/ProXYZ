"""Smoke test: script_utils/tokenizer_tools.py evaluate/train/viz streaming reads.

The commands stream sequences lazily via dataset.data_iterator(data_format)
(one {"id", "text"} record per example, flattened from batches) instead of the
raw line_iterator that once fed dicts into tokenizer.encode (TypeError), and
without create_dataset's eager Arrow materialization (so huge corpora scale).
evaluate still collects rows into a pandas DataFrame for the CSV. Checks:
  1. evaluate: per-sequence CSV rows match direct tokenizer.encode pointwise.
  2. evaluate: aggregate summary equals the totals over all sequences.
  3. evaluate: no -o -> summary only, no file; '-' -> CSV on stdout.
  4. train: streams records and saves a working tokenizer.
  5. viz: decodes each record's text without TypeError.
  6. evaluate --data_format fasta: parses headers into the id column.
  7. no Arrow cache is materialized (streaming scale guard).

Run with:
    PYTHONPATH=src TOKENIZERS_PARALLELISM=false \
        python tests/test_tokenizer_tools_eval.py
"""

import atexit
import csv
import importlib.util
import io
import os
import shutil
import tempfile

from click.testing import CliRunner
from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SCRIPT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "script_utils",
    "tokenizer_tools.py",
)
AA = "ACDEFGHIKLMNPQRSTVWY"
SEQS = [
    "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLK",
    "ACDEFGHIKLMNPQRSTVWY",
    "AAAAACCCCCGGGGGTTTTT",
    "M" * 120,
    "WY" * 7 + "X",  # 'X' may be out-of-vocab -> exercises unk_count
]


def load_script():
    spec = importlib.util.spec_from_file_location("tokenizer_tools", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_tokenizer(path):
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
    tok.save(path)
    return tok


def write_seqs(path):
    with open(path, "w", encoding="utf-8") as f:
        for s in SEQS:
            f.write(s + "\n")


def expected_rows(tok):
    unk_id = tok.token_to_id("[UNK]")
    rows = []
    for i, s in enumerate(SEQS):
        ids = tok.encode(s).ids
        n_tokens, n_chars = len(ids), len(s)
        n_unk = ids.count(unk_id)
        tpw = n_tokens / n_chars if n_chars else 0
        ur = (n_unk / n_tokens) * 100 if n_tokens else 0
        rows.append(
            {
                "id": str(i),
                "length": str(n_chars),
                "num_tokens": str(n_tokens),
                "unk_count": str(n_unk),
                "tokens_per_word": f"{tpw:.4f}",
                "unk_rate": f"{ur:.4f}",
            }
        )
    return rows


def check_csv(mod, tmpdir):
    tok_path = os.path.join(tmpdir, "tok.json")
    seq_path = os.path.join(tmpdir, "seqs.txt")
    out_path = os.path.join(tmpdir, "metrics.csv")
    make_tokenizer(tok_path)
    write_seqs(seq_path)
    tok = Tokenizer.from_file(tok_path)

    result = CliRunner().invoke(
        mod.main, ["evaluate", seq_path, "-m", tok_path, "-o", out_path]
    )
    assert result.exit_code == 0, result.output + str(result.exception)

    with open(out_path, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    exp = expected_rows(tok)
    assert len(got) == len(exp), (len(got), len(exp))
    for g, e in zip(got, exp):
        assert g == e, (g, e)
    print("PASS: per-sequence CSV rows match direct encoding pointwise")

    # Aggregate summary must equal totals over all sequences.
    total_tokens = sum(int(r["num_tokens"]) for r in exp)
    total_chars = sum(int(r["length"]) for r in exp)
    total_unk = sum(int(r["unk_count"]) for r in exp)
    tpw = total_tokens / total_chars
    ur = total_unk / total_tokens * 100
    assert f"Tokens per Word: {tpw:.2f}" in result.output, result.output
    assert f"UNK Token Rate:  {ur:.2f}%" in result.output, result.output
    print("PASS: aggregate summary matches totals")


def check_no_output(mod, tmpdir):
    tok_path = os.path.join(tmpdir, "tok2.json")
    seq_path = os.path.join(tmpdir, "seqs2.txt")
    make_tokenizer(tok_path)
    write_seqs(seq_path)
    result = CliRunner().invoke(mod.main, ["evaluate", seq_path, "-m", tok_path])
    assert result.exit_code == 0, result.output + str(result.exception)
    assert "Tokens per Word:" in result.output
    assert not os.path.exists(os.path.join(tmpdir, "should_not_exist.csv"))
    print("PASS: no -o -> summary only, no CSV file")


def check_stdout(mod, tmpdir):
    tok_path = os.path.join(tmpdir, "tok3.json")
    seq_path = os.path.join(tmpdir, "seqs3.txt")
    make_tokenizer(tok_path)
    write_seqs(seq_path)
    result = CliRunner().invoke(
        mod.main, ["evaluate", seq_path, "-m", tok_path, "-o", "-"]
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    # CSV header + rows are emitted on stdout alongside the summary.
    assert "id,length,num_tokens,unk_count,tokens_per_word,unk_rate" in result.output
    csv_text = result.output.split(
        "id,length,num_tokens,unk_count,tokens_per_word,unk_rate\n", 1
    )[1]
    # Stop at the summary line that follows the CSV body.
    csv_text = csv_text.split("Tokens per Word:")[0]
    got = list(csv.DictReader(io.StringIO(
        "id,length,num_tokens,unk_count,tokens_per_word,unk_rate\n" + csv_text
    )))
    assert len(got) == len(SEQS), (len(got), len(SEQS))
    print("PASS: '-' writes CSV to stdout")


def check_train(mod, tmpdir):
    seq_path = os.path.join(tmpdir, "train_seqs.txt")
    out_path = os.path.join(tmpdir, "trained.json")
    write_seqs(seq_path)
    result = CliRunner().invoke(
        mod.main,
        ["train", seq_path, "-o", out_path, "--vocab_size", "64", "--min_frequency", "1"],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    assert os.path.exists(out_path), "train did not save a tokenizer"
    tok = Tokenizer.from_file(out_path)
    assert len(tok.encode(SEQS[0]).ids) > 0
    print("PASS: train streams data_iterator records and saves a working tokenizer")


def check_viz(mod, tmpdir):
    tok_path = os.path.join(tmpdir, "viz_tok.json")
    seq_path = os.path.join(tmpdir, "viz_seqs.txt")
    make_tokenizer(tok_path)
    write_seqs(seq_path)
    result = CliRunner().invoke(mod.main, ["viz", seq_path, "-m", tok_path])
    assert result.exit_code == 0, result.output + str(result.exception)
    for s in SEQS:
        assert f"Original: {s}" in result.output, s
    assert "Decoded:" in result.output
    print("PASS: viz decodes each record's text without TypeError")


def check_fasta(mod, tmpdir):
    tok_path = os.path.join(tmpdir, "fasta_tok.json")
    fa_path = os.path.join(tmpdir, "seqs.fasta")
    out_path = os.path.join(tmpdir, "fasta_metrics.csv")
    make_tokenizer(tok_path)
    descriptions = ["prot1 desc", "prot2", "prot3 long description"]
    with open(fa_path, "w", encoding="utf-8") as f:
        for desc, s in zip(descriptions, SEQS[:3]):
            f.write(f">{desc}\n{s}\n")
    result = CliRunner().invoke(
        mod.main,
        ["evaluate", fa_path, "-m", tok_path, "--data_format", "fasta", "-o", out_path],
    )
    assert result.exit_code == 0, result.output + str(result.exception)
    tok = Tokenizer.from_file(tok_path)
    with open(out_path, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == len(descriptions), len(got)
    for row, desc, s in zip(got, descriptions, SEQS[:3]):
        assert row["id"] == desc, (row["id"], desc)
        assert int(row["length"]) == len(s)
        assert int(row["num_tokens"]) == len(tok.encode(s).ids)
    print("PASS: evaluate --data_format fasta parses headers into ids")


def main():
    # Point any incidental HF cache at an isolated dir so we can assert the
    # streaming data_iterator path materializes no Arrow files (scale guard).
    cache_dir = tempfile.mkdtemp(prefix="tokenizer_tools_hf_cache_")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    atexit.register(shutil.rmtree, cache_dir, ignore_errors=True)
    mod = load_script()
    with tempfile.TemporaryDirectory() as tmpdir:
        check_csv(mod, tmpdir)
        check_no_output(mod, tmpdir)
        check_stdout(mod, tmpdir)
        check_train(mod, tmpdir)
        check_viz(mod, tmpdir)
        check_fasta(mod, tmpdir)
        arrows = [
            f for _, _, fs in os.walk(cache_dir) for f in fs if f.endswith(".arrow")
        ]
        assert not arrows, f"streaming path materialized Arrow cache: {arrows}"
    print("PASS: streaming data_iterator materializes no Arrow cache")
    print("All tokenizer_tools checks passed.")


if __name__ == "__main__":
    main()