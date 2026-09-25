import random
from typing import Any, Iterable

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
    tokenized = processor(examples, **kwargs)
    if "fim_apply" in kwargs:
        tokenized["is_fim"] = [kwargs["fim_apply"]] * batch_size
    if "id" in examples:
        tokenized["id"] = examples["id"]

    return tokenized


def prepare_inputs(
    processor, inputs: dict[str, torch.Tensor | Any]
) -> dict[str, torch.Tensor | Any]:
    if "distogram_labels" in inputs:
        inputs["distogram_labels"] = processor.to_distogram(
            inputs["distogram_labels"][..., :-1],
            inputs["distogram_labels"][...,  -1],
        )
    if "residue_idx" in inputs and "char_position_ids" not in inputs:
        inputs["char_position_ids"] = inputs["residue_idx"]
    return inputs


def deduplicate(
    records: Iterable[dict[str, Any]], key: str = "id"
) -> Iterable[dict[str, Any]]:
    seen = set()

    for record in records:
        if record[key] in seen:
            continue
        seen.add(record[key])
        yield record
