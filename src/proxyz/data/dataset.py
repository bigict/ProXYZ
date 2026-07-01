import sys
import contextlib
import gzip


FIM_PREFIX = "<fim_prefix>"
FIM_SUFFIX = "<fim_suffix>"
FIM_MIDDLE = "<fim_middle>"
FIM_TOKENS = [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]


def lines(f):
    for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
        yield line


@contextlib.contextmanager
def fopen(file_path):
    if file_path == "-":
        yield sys.stdin
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            yield f
    else:
        with open(file_path, "r") as f:
            yield f

def line_iterator(file_paths, batch_size=64):
    batch = []
    for file_path in file_paths:
        with fopen(file_path) as f:
            for line in lines(f):
                batch.append(line)
                if len(batch) >= batch_size:
                    yield batch
                    batch = []
    if batch:
        yield batch


def fasta_parse(file_path):
    with fopen(file_path) as f:
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


def fasta_iterator(file_paths, batch_size=64):
    batch = []
    for file_path in file_paths:
        for _, text in fasta_parse(file_path):
            if len(batch) >= batch_size:
                yield batch
                batch = []

            batch.append(text)
    if batch:
        yield batch
