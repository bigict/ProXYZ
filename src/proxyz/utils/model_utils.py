import os
import glob
import re

from torch import nn


def apply_trainable_keys(
    model: nn.Module, trainable_keys: list[str]
) -> tuple[int, int]:
    """Freeze/defreeze parameters based on name substrings."""
    n_total, n_trainable = 0, 0
    for name, param in model.named_parameters():
        n_total += 1
        keep = any(k in name for k in trainable_keys)
        param.requires_grad = keep
        if keep:
            n_trainable += 1
    return n_total, n_trainable


def resolve_model_path(model_dir: str) -> str:
    """Return model_dir if it holds a model directly, else the latest checkpoint-*."""
    if os.path.isfile(os.path.join(model_dir, "model.safetensors")) or os.path.isfile(
        os.path.join(model_dir, "pytorch_model.bin")
    ):
        return model_dir

    checkpoints = glob.glob(os.path.join(model_dir, "checkpoint-*"))
    checkpoints = [c for c in checkpoints if re.search(r"checkpoint-(\d+)$", c)]
    if not checkpoints:
        raise ValueError(
            f"No model weights found in {model_dir} and no checkpoint-* subdirectories. "
            "Pass --model_dir pointing at a trained model or checkpoint."
        )
    latest = max(
        checkpoints, key=lambda c: int(re.search(r"checkpoint-(\d+)$", c).group(1))
    )
    return latest
