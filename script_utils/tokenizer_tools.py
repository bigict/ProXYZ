from typing import Sequence

import click
import pandas as pd
from loguru import logger
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from proxyz.data import dataset
from proxyz.data.utils import opener
from proxyz.utils import dict2object

"""
SentencePiece BPE Tokenizer
as outlined in Kudo 2018 Subword Regularization: Improving Neural Network Translation Modelswith Multiple Subword Candidates

The central idea is to `virtually augment training data with on-the-fly subword  sampling`,
which  helps to improve the accuracy as well as robustness of NMT models.
For better subword sampling they use the unigram language model, which unlike the greedy BPE approach
(takes two tokens, looks at the frequency of each pair and then merges the pairs that have
the highest combined frequency count) chooses the most likely likely combination.

Algorithm is performed in Expectation Maximization (EM) setting:
    0) convert all the input into unicode, even spaces (as underscores, '_')
    1) calculate probabilities (frequency-based) of each subword token (can seed the subword token set with BPE)
    2) with EM estimate a loss which would result if each subword token was discarded
    3) discard tokens with the largest loss (can adjust the fraction of the worst tokens to drop with param )
    <-- insert fraction param
    4) repeat steps 1-3 until reached final vocabulary size or until there is no change in token numbers after successive iterations

Pecularities:
    - spaces encoded as "_", or symbol U+2581
"""

@click.group()
def main():
    pass

@main.command("train", context_settings={"show_default": True})
@click.argument("files", type=click.Path(), nargs=-1)
@click.option(
    "--data_format", type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line", help="Input format of the sequence files."
)
@click.option(
    "-o", "--out", type=click.Path(), default="tokens.json", help="Path to the output directory, where the files will be saved"
)
@click.option("--vocab_size", type=int, default=30000, help="Vocabulary size.")
@click.option(
    "--min_frequency", type=int, default=0, help="The minimum frequency a pair should have in order to be merged."
)
@click.option(
    "--max_token_length", type=int, default=None, help="Prevents create tokens logger than the specified size."
)
@click.option(
    "--limit_alphabet", type=int, default=None, help="The size of alphabet character set (e.g., for English, |alphabet|=26)"
)
@click.option(
    "--pretty", is_flag=True, help="Whether the JSON file should be pretty formatted."
)
@click.option(
    "--batch_size", type=int, default=5000, help="The batch size."
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def train(**args):
    args = dict2object(**args)

    # Initialize an empty tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Setup trainer
    trainer = BpeTrainer(
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        max_token_length=args.max_token_length,
        special_tokens=[
            "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]", "[BOS]", "[EOS]"
        ],
    )

    tokenizer.train_from_iterator(
        line_iterator(args.files, args.data_format, batch_size=args.batch_size), trainer
    )

    # Save the files
    tokenizer.save(args.out, pretty=args.pretty)

    # Restoring model from learned vocab/merges
    tokenizer = Tokenizer.from_file(args.out)

    # Test encoding
    logger.info(
        "Tokens and their ids from SentencePiece with GFP protein sequence:\n >4EUL_1|Chain A|Green fluorescent protein|Aequorea victoria (6100)\n MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    )
    encoded = tokenizer.encode(
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    )
    logger.info(encoded.tokens)
    logger.info(encoded.ids)
    logger.info('done!')

@main.command("evaluate", context_settings={"show_default": True})
@click.argument("files", type=click.Path(), nargs=-1)
@click.option(
    "--data_format", type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line", help="Input format of the sequence files."
)
@click.option(
    "-m", "--model", type=click.Path(), default="tokens.json", help="Path to the tokenizer.json"
)
@click.option(
    "--batch_size", type=int, default=5000, help="The batch size."
)
@click.option(
    "-o", "--output", type=click.Path(), default=None,
    help="Write per-sequence metrics to this CSV (supports .gz; '-' for stdout). "
    "Columns: id, length, num_tokens, unk_count, tokens_per_word, unk_rate. "
    "If omitted, only the aggregate summary is printed."
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def evaluate(**args):
    args = dict2object(**args)

    # Restoring model from learned vocab/merges
    tokenizer = Tokenizer.from_file(args.model)

    total_tokens = 0
    total_words = 0
    unk_count = 0

    unk_id = tokenizer.token_to_id("[UNK]")

    rows = []
    data_iterator = dataset.data_iterator(args.data_format)
    for batch in data_iterator(args.files, batch_size=args.batch_size):
        for record in batch:
            encoded = tokenizer.encode(record["text"])

            n_tokens = len(encoded.ids)
            n_chars = len(record["text"])
            n_unk = encoded.ids.count(unk_id)

            total_tokens += n_tokens
            total_words += n_chars
            unk_count += n_unk

            if args.output is not None:
                # Per-sequence tokens per residue (char) and UNK rate (%).
                tpw = n_tokens / n_chars if n_chars > 0 else 0
                ur = (n_unk / n_tokens) * 100 if n_tokens > 0 else 0
                rows.append(
                    {
                        "id": record["id"],
                        "length": n_chars,
                        "num_tokens": n_tokens,
                        "unk_count": n_unk,
                        "tokens_per_word": tpw,
                        "unk_rate": ur,
                    }
                )

    if rows and args.output is not None:
        df = pd.DataFrame(rows)
        with opener(args.output, "w") as f:
            df.to_csv(f, index=False, float_format="%.4f")

    # 1. Tokens per Word (Lower is usually better, ideally 1.1 to 1.4 for native text)
    tokens_per_word = total_tokens / total_words if total_words > 0 else 0

    # 2. UNK Rate (Should be as close to 0% as possible)
    unk_rate = (unk_count / total_tokens) * 100 if total_tokens > 0 else 0

    print(f"Tokens per Word: {tokens_per_word:.2f}")
    print(f"UNK Token Rate:  {unk_rate:.2f}%")


@main.command("viz", context_settings={"show_default": True})
@click.argument("files", type=click.Path(), nargs=-1)
@click.option(
    "--data_format", type=click.Choice(["line", "fasta", "foldcomp", "pdb"]),
    default="line", help="Input format of the sequence files."
)
@click.option(
    "-m", "--model", type=click.Path(), default="tokens.json", help="Path to the tokenizer.json"
)
@click.option(
    "--batch_size", type=int, default=5000, help="The batch size."
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def viz(**args):
    args = dict2object(**args)

    # Restoring model from learned vocab/merges
    tokenizer = Tokenizer.from_file(args.model)

    for batch in line_iterator(args.files, args.data_format, batch_size=args.batch_size):
        for line in batch:
            encoded = tokenizer.encode(line)
            decoded = tokenizer.decode(encoded.ids)

            print(f"Original: {line}")
            print(f"Decoded:  {decoded}")

@main.command("vocab", context_settings={"show_default": True})
@click.option(
    "-m", "--model", type=click.Path(), default="tokens.json", help="Path to the tokenizer.json"
)
@click.option("-v", "--verbose", is_flag=True, help="verbose output.")
def vocab(**args):
    args = dict2object(**args)

    # Restoring model from learned vocab/merges
    tokenizer = Tokenizer.from_file(args.model)

    if args.verbose:
        print(f"{args.model}: vocab_size={tokenizer.get_vocab_size()}")

    vocab = tokenizer.get_vocab(with_added_tokens=False)
    for token, tid in vocab.items():
        print(f"{token}\t{tid}")


def line_iterator(data_files: Sequence[str], data_format: str, batch_size: int = 64):
    iterator = dataset.data_iterator(data_format)
    for batch in iterator(data_files, batch_size=batch_size):
        yield [record["text"] for record in batch]


if __name__ == "__main__":
    main()

