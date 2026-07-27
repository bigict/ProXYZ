import sys
import contextlib
import gzip


def lines(f):
    for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
        yield line


@contextlib.contextmanager
def opener(file_path: str):
    if file_path == "-":
        yield sys.stdin
    elif file_path.endswith(".gz"):
        with gzip.open(file_path, "rt") as f:
            yield f
    else:
        with open(file_path, "r") as f:
            yield f
