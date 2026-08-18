import random

import torch

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaForSequenceClassification,
    LlamaForTokenClassification,
    LlamaMLP,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    eager_attention_forward,
)
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.configuration_utils import PreTrainedConfig
from transformers.generation import GenerationMixin
from transformers.integrations import (
    use_kernel_forward_from_hub, use_kernel_func_from_hub, use_kernelized_func
)
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import (
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPast, CausalLMOutputWithPast, TokenClassifierOutput
)
from transformers.modeling_rope_utils import (
    ROPE_INIT_FUNCTIONS, dynamic_rope_update, RopeParameters
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.processing_utils import ProcessorMixin, Unpack
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple, logging
from transformers.utils.generic import maybe_autocast, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs
from transformers.utils.type_validators import interval

from proxyz.utils import attr

logger = logging.get_logger(__name__)


@auto_docstring(checkpoint="bigict/ProXYZ")
@strict
class XYZConfig(LlamaConfig):
    pass
class XYZForCausalLM(LlamaForCausalLM):
    pass


class XYZForSequenceClassification(LlamaForSequenceClassification):
    pass


class XYZForTokenClassification(LlamaForTokenClassification):
    pass


class XYZProcessor(ProcessorMixin):
    """Processes protein sequences at both amino-acid and BPE-token granularity.

    Inherits ``ProcessorMixin`` so it supports ``save_pretrained`` /
    ``from_pretrained`` out of the box.  The wrapped BPE tokenizer is
    persisted alongside the processor config.

    ``AutoProcessor.from_pretrained(dir)`` works as long as *proxyz* is
    importable (the ``auto_map`` entry in *preprocessor_config.json*
    points to ``proxyz.processor.XYZProcessor``).
    """

    attributes = ["tokenizer"]
    tokenizer_class = "AutoTokenizer"

    FIM_PREFIX = "<fim_prefix>"
    FIM_SUFFIX = "<fim_suffix>"
    FIM_MIDDLE = "<fim_middle>"
    FIM_TOKENS = [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE]

    def __init__(self, tokenizer, text_column: str = "text", **kwargs) -> None:
        tokenizer.add_special_tokens({"additional_special_tokens": self.FIM_TOKENS})
        logger.info(
            f"Added FIM tokens: {self.FIM_TOKENS} to tokenizer. "
            f"(vocabu size: {len(tokenizer)})"
        )

        special_tokens = set(tokenizer.all_special_tokens)
        self.max_chars_within_token = max(
            len(token) for token in tokenizer.get_vocab() if token not in special_tokens
        )

        super().__init__(tokenizer, **kwargs)

        self.text_column = text_column
        self.ignore_index = kwargs.get("ignore_index", -100)

    def __call__(
        self,
        examples: dict,
        *,
        bpe_dropout: float | None = None,
        fim_apply: bool = False,
        fim_spm_rate: float = 0.5,
        fim_sft_style: bool = False,
        max_length: int | None = None,
        generate: bool = False,
        **kwargs,
    ) -> dict:
        examples = self.apply_crop(examples, max_length=max_length)
        if fim_apply:
            examples = self.apply_fim(examples, spm_rate=fim_spm_rate)
        examples = self.apply_wrap(examples, add_eos_token=not generate)

        tokenized = self.to_tokenization(
            examples,
            bpe_dropout=bpe_dropout,
            fim_apply=fim_apply,
            fim_sft_style=fim_sft_style,
            generate=generate,
        )

        return tokenized

    def to_tokenization(
        self,
        examples: dict,
        tokenized: dict = None,
        bpe_dropout: float | None = None,
        fim_apply: bool = False,
        fim_sft_style: bool = False,
        generate: bool = False,
    ) -> dict:
        if tokenized is None:
            tokenized = {}
        tokenized.update(
            self._tokenize_with_dropout(
                examples,
                bpe_dropout=bpe_dropout,
                fim_apply=fim_apply,
                fim_sft_style=fim_sft_style,
                generate=generate,
            )
        )
        return tokenized

    def to_example(self, input_ids: torch.LongTensor) -> dict:
        text = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        return {self.text_column: [t.replace(" ", "") for t in text]}

    def _tokenize_with_dropout(
        self,
        examples: dict,
        *,
        bpe_dropout: float | None = None,
        prefix: str = "",
        fim_apply: bool = False,
        fim_sft_style: bool = False,
        generate: bool = False,
        **kwargs,
    ) -> dict:
        with attr(self.tokenizer.backend_tokenizer.model, dropout=bpe_dropout):
            tokenized = self.tokenizer(
                examples[self.text_column],
                truncation=True,
                return_tensors="pt",
                padding=True,
                return_offsets_mapping=True,
                *kwargs,
            )
        if not generate:
            tokenized["labels"] = tokenized["input_ids"].where(tokenized["attention_mask"] > 0, self.ignore_index)
            if fim_apply and fim_sft_style:
                middle_pos = (tokenized["labels"] == self.tokenizer.convert_tokens_to_ids(self.FIM_MIDDLE)).cumsum(
                    1
                )
                tokenized["labels"] = tokenized["labels"].where(
                    middle_pos.cumsum(1) > 1,
                    self.ignore_index,  # the <fim_middle> is excluded
                )
        return {f"{prefix}{k}": v for k, v in tokenized.items()}

    def apply_fim(self, examples: dict, spm_rate: float = 0.5) -> dict:
        """Split content into prefix/middle/suffix and rearrange for FIM training.
        Prefix or suffix may be empty, but middle is always non-empty."""
        for idx, text in enumerate(examples[self.text_column]):
            n = len(text)
            cut1 = random.randint(0, n - 1)
            cut2 = random.randint(cut1 + 1, n)
            prefix, middle, suffix = text[:cut1], text[cut1:cut2], text[cut2:]

            is_spm = random.random() < spm_rate
            if is_spm:
                # SPM: <BOS><fim_suffix><suffix><fim_prefix><prefix><fim_middle><middle><EOS>
                first_tag, second_tag = self.FIM_SUFFIX, self.FIM_PREFIX
                first, second = suffix, prefix
            else:
                # PSM: <BOS><fim_prefix><prefix><fim_suffix><suffix><fim_middle><middle><EOS>
                first_tag, second_tag = self.FIM_PREFIX, self.FIM_SUFFIX
                first, second = prefix, suffix

            examples[self.text_column][idx] = (
                first_tag + first + second_tag + second + self.FIM_MIDDLE + middle
            )
        return examples

    def apply_crop(self, examples: dict, max_length: int | None = None) -> dict:
        for idx, text in enumerate(examples[self.text_column]):
            n = len(text)
            if max_length and max_length < n:
                cut = random.randint(0, n - max_length)
                text = text[cut:cut + max_length]
                examples[self.text_column][idx] = text
        return examples

    def apply_wrap(self, examples: dict, add_eos_token: bool = True) -> dict:
        for idx, text in enumerate(examples[self.text_column]):
            text = f"{self.tokenizer.bos_token}{text}"
            if add_eos_token:
                text = f"{text}{self.tokenizer.eos_token}"
            examples[self.text_column][idx] = text
        return examples


__all__ = [
    "XYZPreTrainedModel",
    "XYZModel",
    "XYZForCausalLM",
    "XYZForSequenceClassification",
    "XYZForTokenClassification",
    "XYZConfig",
    "XYZProcessor",
]
