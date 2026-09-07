import sys
import contextlib
import gzip
import logging

logger = logging.getLogger(__name__)


def lines(f):
    for line in filter(lambda x: x, map(lambda x: x.strip(), f)):
        yield line


@contextlib.contextmanager
def opener(file_path: str, mode: str = "r"):
    if file_path == "-":
        if "w" in mode:
            yield sys.stdout
        else:
            yield sys.stdin
    elif file_path.endswith(".gz"):
        if "b" not in mode:
            mode = f"{mode}t"  # text mode by default
        with gzip.open(file_path, mode) as f:
            yield f
    else:
        with open(file_path, mode) as f:
            yield f


@contextlib.contextmanager
def semaphore(name: str, value: int = 0):
    if value >= 0:  # disabled when value < 0
        try:
            import posix_ipc
            with posix_ipc.Semaphore(
                f"/{name}", flags=posix_ipc.O_CREAT, initial_value=value
            ) as sem:
                yield sem
            return
        except Exception as e:
            logger.warning(f"Create semaphore {name} failed: {e}")
    yield None
