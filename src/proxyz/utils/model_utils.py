import os
import glob
import re


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
