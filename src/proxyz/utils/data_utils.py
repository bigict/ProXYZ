import random
from typing import Any

import torch

from proxyz.data import dataset


def tokenize_function(
    processor, data_format: str, examples: dict[str, torch.Tensor | Any], **kwargs
) -> dict[str, torch.Tensor | Any]:
    if "fim_rate" in kwargs:
        kwargs["fim_apply"] = random.random() < kwargs.pop("fim_rate")
    if "max_length" in kwargs and kwargs["max_length"] is not None:
        kwargs["max_length"] = (
            kwargs["max_length"] - (5 if kwargs.get("fim_apply", False) else 2)
        )

    transform = dataset.data_transform(data_format)
    if transform is not None:
        examples = transform(examples)

    batch_size = len(examples[processor.text_column])
    examples = processor(examples, **kwargs)
    if "fim_apply" in kwargs:
        examples["is_fim"] = [kwargs["fim_apply"]] * batch_size

    return examples


def prepare_inputs(
    processor, inputs: dict[str, torch.Tensor | Any]
) -> dict[str, torch.Tensor | Any]:
    if "distogram_labels" in inputs:
        inputs["distogram_labels"] = processor.to_distogram(
            inputs["distogram_labels"][..., :-1],
            inputs["distogram_labels"][...,  -1],
        )
    return inputs
