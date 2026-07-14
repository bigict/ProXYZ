"""U-Net style language model.

Three-stage architecture:
  -> char encoder       (character granularity)
  -> token transformmer (token granularity)
  -> char decoder       (character granularity)
  => predict next token
"""
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import functools
import random

import torch
from torch import nn

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
    """
    Configuration for the XYZ U-Net style protein language model.

    A three-stage architecture:

    1. **Char encoder** — transformer at character (residue) granularity
    2. **Token trunk** — transformer at BPE-token granularity
    3. **Char decoder** — transformer at character granularity with U-Net skip
       connections from the encoder

    The char encoder/decoder share the same ``head_dim`` as the trunk so that
    a single ``RotaryEmbedding`` instance can be used across all three stacks.

    This configuration extends [`LlamaConfig`].  All token-level fields
    (``hidden_size``, ``intermediate_size``, ``num_hidden_layers``, etc.) are
    inherited and control the *trunk* transformer.  The char-level fields
    below control the encoder and decoder.

    Constraint: ``char_hidden_size == char_num_attention_heads × head_dim``.

    char_hidden_size (`int`, *optional*, defaults to 2048):
        Hidden size of the char encoder / decoder transformer.
    char_intermediate_size (`int`, *optional*, defaults to 5504):
        FFN intermediate size for char-level layers.
    char_num_hidden_layers (`int`, *optional*, defaults to 4):
        Number of transformer layers in the char encoder and char decoder.
    char_num_attention_heads (`int`, *optional*, defaults to 16):
        Number of attention heads for char-level self-attention.
    char_num_key_value_heads (`int`, *optional*):
        Number of KV heads for GQA at char level.  Defaults to
        ``char_num_attention_heads`` (i.e. multi-head attention) when
        not specified.

    use_char_position_ids (`bool`, *optional*, defaults to `False`):
        If ``True``, token-level ``position_ids`` are gathered from
        ``char_position_ids`` via ``repr_char_idx`` instead of being
        computed independently.  This keeps RoPE aligned between the
        char and token stacks.

    has_char_lm_head (`bool`, *optional*, defaults to `False`):
        If ``True``, create a next-character prediction head
        (``char_lm_head``) on top of the char decoder output.

    has_cle_lm_head (`bool`, *optional*, defaults to `False`):
        If ``True``, create a CLE (Cα–Local–Environment) classification
        head on top of the char decoder output.

    has_distogram_lm_head (`bool`, *optional*, defaults to `False`):
        If ``True``, create a distogram prediction head on top of the
        char decoder output.  Predicts pairwise residue–residue distance
        bins via an outer-sum MLP.

    distogram_bins_num (`int`, *optional*, defaults to 64):
        Number of distance bins for the distogram head.

    distogram_intermediate_size (`int`, *optional*, defaults to 32):
        Inner dimension for the distogram pair representation.  Each
        residue is projected to this size by separate left/right linear
        layers; the outer product yields a ``distogram_intermediate_size²``
        feature per residue pair before the final classification layer.

    distogram_chunk_size (`int`, *optional*, defaults to 0):
        If > 0, compute the distogram in chunks of this size along the
        first sequence dimension to reduce peak memory.  ``0`` disables
        chunking (compute the full ``L×L`` matrix at once).

    Note:
        ``char_head_dim`` is intentionally omitted — the char stacks reuse the
        token-level ``head_dim`` so that RoPE can be shared.

    Example:
    ```python
    >>> from proxyz.models import XYZConfig, XYZForCausalLM
    >>> config = XYZConfig()
    >>> model = XYZForCausalLM(config)
    ```
    """

    model_type = "xyz"
    keys_to_ignore_at_inference = ["past_key_values", "char_past_key_values"]

    # ---- Tensor-parallel / pipeline-parallel plans (inherited from Llama) ----
    base_model_tp_plan = {
        ".*.layers.*.self_attn.q_proj": "colwise",
        ".*.layers.*.self_attn.k_proj": "colwise",
        ".*.layers.*.self_attn.v_proj": "colwise",
        ".*.layers.*.self_attn.o_proj": "rowwise",
        ".*.layers.*.mlp.gate_proj": "colwise",
        ".*.layers.*.mlp.up_proj": "colwise",
        ".*.layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        ".*.layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        ".*.norm": (["hidden_states"], ["hidden_states"]),
    }

    char_hidden_size: int = 2048
    char_intermediate_size: int = 5504
    char_num_hidden_layers: int = 4
    char_num_attention_heads: int = 16
    char_num_key_value_heads: int | None = None

    use_char_position_ids: bool = False

    has_char_lm_head: bool = False
    has_cle_lm_head: bool = False
    has_distogram_lm_head: bool = False

    distogram_bins_num: int = 64
    distogram_intermediate_size: int = 32
    distogram_chunk_size: int = 0  # 0 = no chunking (compute full L×L at once)

    def __post_init__(self, **kwargs):
        # Default char KV heads to char query heads (MHA) when unspecified.
        if self.char_num_key_value_heads is None:
            self.char_num_key_value_heads = self.char_num_attention_heads
        super().__post_init__(**kwargs)

    def validate_architecture(self):
        # Ensure char_hidden_size is compatible with (char_num_attention_heads, head_dim).
        if self.char_hidden_size != self.char_num_attention_heads * self.head_dim:
            raise ValueError(
                f"The char hidden size ({self.char_hidden_size}) must equal "
                f"char_num_attention_heads ({self.char_num_attention_heads}) × head_dim ({self.head_dim})."
            )
        super().validate_architecture()

    @contextlib.contextmanager
    def tokenization(self):
        """Context manager that yields *self* with token-level config active.

        This is a no-op passthrough — the inherited Llama fields already hold
        the token-level values.  Provided for symmetry with ``characterization``.
        """
        yield self

    @contextlib.contextmanager
    def characterization(self):
        """Context manager that temporarily swaps token-level fields with
        char-level equivalents so that ``XYZDecoderLayer`` / ``XYZDecoderLayers``
        can be constructed or invoked with char-granularity dimensions.

        On exit, all fields are restored to their original (token-level) values.
        """
        with attr(
            self,
            hidden_size=self.char_hidden_size,
            intermediate_size=self.char_intermediate_size,
            num_hidden_layers=self.char_num_hidden_layers,
            num_attention_heads=self.char_num_attention_heads,
            num_key_value_heads=self.char_num_key_value_heads,
        ):
            yield self


class XYZRMSNorm(LlamaRMSNorm):
    pass


class XYZRotaryEmbedding(LlamaRotaryEmbedding):
    pass


class XYZMLP(LlamaMLP):
    pass


class XYZDistogram(nn.Module):
    def __init__(self, config: XYZConfig):
        super().__init__()
        self.config = config

        self.left_proj = nn.Linear(
            config.char_hidden_size, config.distogram_intermediate_size
        )
        self.right_proj = nn.Linear(
            config.char_hidden_size, config.distogram_intermediate_size
        )
        self.out_proj = nn.Linear(
            config.distogram_intermediate_size**2, config.distogram_bins_num
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        # x = torch.einsum(
        #     "... i c, ... j d -> ... i j c d", self.left_proj(x), self.right_proj(x)
        # )
        # x = x.view(*x.shape[:-2], -1)
        # return self.out_proj(self.act_fn((x + x.transpose(-2, -3)) / 2))  # symmetrize

        # Decompose the bilinear form to avoid materializing (B, L, L, D²).
        # Reshape out_proj weight: (bins, D²) -> (bins, D, D)
        d = self.config.distogram_intermediate_size
        x = torch.einsum(
            "k c d, ... i c, ... j d -> ... i j k",
            self.out_proj.weight.view(self.config.distogram_bins_num, d, d),
            self.act_fn(self.left_proj(x)),
            self.act_fn(self.right_proj(x)),
        )
        # Symmetrize on (B, L, L, bins) and add bias
        x = (x + x.transpose(-2, -3)) / 2
        if self.out_proj.bias is not None:
            x = x + self.out_proj.bias
        return x


class XYZDecoderLayer(LlamaDecoderLayer):
    pass


class XYZDecoderLayers(nn.Module):
    def __init__(self, config: XYZConfig):
        super().__init__()
        self.config = config  # FIX: AttributeError: 'XYZDecoderLayers' object has no attribute 'config'

        self.layers = nn.ModuleList(
            [XYZDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = XYZRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        causal_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        """
        Runs a stack of [`XYZDecoderLayer`] modules followed by RMS normalization.

        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch, seq_len, hidden_size)`):
                Input embeddings (output of a projection or embedding layer).
            causal_mask (`torch.Tensor`):
                Pre-computed causal attention mask.
            position_embeddings (`torch.Tensor`):
                RoPE cos/sin embeddings for the current positions.
            position_ids (`torch.LongTensor`, *optional*):
                Position indices.  Inferred from *past_key_values* when *None*.
            past_key_values (`Cache`, *optional*):
                Key-value cache for incremental decoding.
            use_cache (`bool`, *optional*):
                Whether to populate and return the KV cache.

        Returns:
            [`BaseModelOutputWithPast`] with the normalized hidden states and
            (optionally) the updated KV cache.
        """
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class XYZPreTrainedModel(LlamaPreTrainedModel):
    _skip_keys_device_placement = ["past_key_values", "char_past_key_values"]


@auto_docstring
@dataclass
class XYZModelOutputWithPast(BaseModelOutputWithPast):
    """
    Output of [`XYZModel`].

    Extends [`BaseModelOutputWithPast`] with character-granularity outputs from
    the U-Net decoder branch.

    char_last_hidden_state (`torch.Tensor` of shape `(batch, char_seq_len, char_hidden_size)`, *optional*):
        Character-level hidden states from the char decoder.
    char_past_key_values (`Tuple[Cache, Cache]`, *optional*):
        Pair of KV caches — one for the char encoder, one for the char
        decoder — used during incremental generation.
    char_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Char-decoder hidden states at every layer (returned when
        ``output_hidden_states=True``).  Each element has shape
        `(batch, char_seq_len, char_hidden_size)`.
    char_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Char-decoder self-attention weights at every layer (returned when
        ``output_attentions=True``).  Each element has shape
        `(batch, num_heads, char_seq_len, char_seq_len)`.
    """

    char_last_hidden_state: torch.FloatTensor | None = None
    char_past_key_values: Tuple[Cache, Cache] | None = None
    char_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    char_attentions: tuple[torch.FloatTensor, ...] | None = None


class XYZModel(XYZPreTrainedModel):
    def __init__(self, config: XYZConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.rotary_emb = XYZRotaryEmbedding(config=config)

        # Encoder (character granularity)
        self.to_char_encoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.char_hidden_size, bias=False)
        )
        with self.config.characterization() as config:
            self.char_encoder = XYZDecoderLayers(config)
        self.from_char_encoder = nn.Sequential(
            nn.Linear(config.char_hidden_size, config.hidden_size, bias=False),
        )

        # Trunk
        with self.config.tokenization() as config:
            self.trunk = XYZDecoderLayers(config)

        # Decoder (character granularity)
        self.to_char_decoder = nn.Sequential(
            nn.Linear(config.hidden_size, config.char_hidden_size, bias=False)
        )
        with self.config.characterization() as config:
            self.char_decoder = XYZDecoderLayers(config)

        self.gradient_checkpointing = False

    @merge_with_config_defaults
    @capture_outputs
    @auto_docstring
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        char_input_ids: torch.LongTensor | None = None,
        char_attention_mask: torch.Tensor | None = None,
        char_position_ids: torch.LongTensor | None = None,
        char_past_key_values: Tuple[Cache, Cache] | None = None,
        char_inputs_embeds: torch.FloatTensor | None = None,
        repr_char_idx: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> XYZModelOutputWithPast:
        """
        U-Net forward pass: char encoder → token trunk → char decoder.

        **Stage 1 — Char encoder.**  Token embeddings are projected to
        *char_hidden_size* and processed by the char encoder stack.

        **Stage 2 — Token trunk.**  Char hidden states are gathered at
        representative positions (``repr_char_idx``), projected back to
        *hidden_size*, and added to the token embeddings.  The result is
        processed by the trunk transformer.

        **Stage 3 — Char decoder.**  Trunk output is projected to
        *char_hidden_size* and scattered back to char positions via
        ``scatter_add`` (skip connection from the trunk).  The char decoder
        stack refines the representation with U-Net skip connections from the
        encoder.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch, token_seq_len)`, *optional*):
                BPE token indices.  Mutually exclusive with *inputs_embeds*.
            attention_mask (`torch.Tensor` of shape `(batch, token_seq_len)`, *optional*):
                Token-level attention mask (1 = attend, 0 = mask).
            position_ids (`torch.LongTensor`, *optional*):
                Token position indices.  Inferred from *past_key_values* when *None*.
            past_key_values (`Cache`, *optional*):
                Trunk KV cache for incremental decoding.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed token embeddings.  Mutually exclusive with *input_ids*.
            char_input_ids (`torch.LongTensor` of shape `(batch, char_seq_len)`, *optional*):
                Character-level token indices.  Mutually exclusive with
                *char_inputs_embeds*.
            char_attention_mask (`torch.Tensor` of shape `(batch, char_seq_len)`, *optional*):
                Character-level attention mask.
            char_position_ids (`torch.LongTensor`, *optional*):
                Character position indices.  Inferred from *char_past_key_values*
                when *None*.
            char_past_key_values (`Tuple[Cache, Cache]`, *optional*):
                Pair of KV caches for the char encoder and char decoder.
            char_inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed char embeddings.  Mutually exclusive with *char_input_ids*.
            repr_char_idx (`torch.LongTensor` of shape `(batch, token_seq_len)`, *optional*):
                Maps each BPE token to its representative character position in
                the char sequence.  Used to gather encoder output into the trunk
                and scatter trunk output back to the decoder.
            use_cache (`bool`, *optional*):
                Whether to populate and return KV caches.

        Returns:
            [`XYZModelOutputWithPast`]:
                - **last_hidden_state** — token-level output from the trunk.
                - **char_last_hidden_state** — char-level output from the char
                  decoder.
                - **past_key_values** / **char_past_key_values** — updated caches.
        """
        # character level.
        if (char_input_ids is None) ^ (char_inputs_embeds is not None):
            raise ValueError("You must specify exactly one of char_input_ids or char_inputs_embeds")

        if char_inputs_embeds is None:
            char_inputs_embeds: torch.Tensor = self.to_char_encoder(self.embed_tokens(char_input_ids))

        if use_cache and char_past_key_values is None:
            with self.config.characterization() as config:
                char_past_key_values = (
                    DynamicCache(config=config), DynamicCache(config=config)
                )

        if char_position_ids is None:
            char_past_seen_tokens = char_past_key_values[0].get_seq_length() if char_past_key_values is not None else 0
            char_position_ids = torch.arange(char_inputs_embeds.shape[1], device=char_inputs_embeds.device) + char_past_seen_tokens
            char_position_ids = char_position_ids.unsqueeze(0)

        char_position_embeddings = self.rotary_emb(char_inputs_embeds, position_ids=char_position_ids)

        # token level
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            with self.config.tokenization() as config:
                past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            if self.config.use_char_position_ids:  # gather from char position_ids
                position_ids = char_position_ids
                if char_position_ids.shape[0] != repr_char_idx.shape[0]:
                    assert char_position_ids.shape[0] == 1
                    position_ids = position_ids.expand(
                        repr_char_idx.shape[0], char_position_ids.shape[1]
                    )
                position_ids = position_ids.gather(1, repr_char_idx)
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
                position_ids = position_ids.unsqueeze(0)
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids=position_ids)

        # encode
        with self.config.characterization() as config:
           char_causal_mask = create_causal_mask(
               config=config,
               inputs_embeds=char_inputs_embeds,
               attention_mask=char_attention_mask,
               past_key_values=char_past_key_values[0] if char_past_key_values else None,
               position_ids=char_position_ids,
           )
           encoder_outputs: BaseModelOutputWithPast = self.char_encoder(
                inputs_embeds=char_inputs_embeds,
                causal_mask=char_causal_mask,
                position_embeddings=char_position_embeddings,
                position_ids=char_position_ids,
                past_key_values=char_past_key_values[0] if char_past_key_values else None,
                use_cache=use_cache,
                **kwargs,
            )

        # trunk
        inputs_embeds = inputs_embeds + self.from_char_encoder(
            encoder_outputs.last_hidden_state.gather(
                1, repr_char_idx[..., None].expand(
                    *repr_char_idx.shape, encoder_outputs.last_hidden_state.shape[2]
                )
            )
        )
        with self.config.tokenization() as config:
            causal_mask = create_causal_mask(
                config=config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=position_ids,
            )

            trunk_outputs: BaseModelOutputWithPast = self.trunk(
                inputs_embeds=inputs_embeds,
                causal_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                **kwargs,
            )

        # decode
        trunk_last_hidden_state = self.to_char_decoder(trunk_outputs.last_hidden_state)
        char_inputs_embeds = encoder_outputs.last_hidden_state.scatter_add(  # skip connection
            1, repr_char_idx[..., None].expand_as(trunk_last_hidden_state), trunk_last_hidden_state
        )
        with self.config.characterization() as config:
            decoder_outputs: BaseModelOutputWithPast = self.char_decoder(
                inputs_embeds=char_inputs_embeds,
                causal_mask=char_causal_mask,
                position_embeddings=char_position_embeddings,
                position_ids=char_position_ids,
                past_key_values=char_past_key_values[1] if char_past_key_values else None,
                use_cache=use_cache,
                **kwargs,
            )

        return XYZModelOutputWithPast(
            last_hidden_state=trunk_outputs.last_hidden_state,
            past_key_values=trunk_outputs.past_key_values,
            char_last_hidden_state=decoder_outputs.last_hidden_state,
            char_past_key_values=(
                encoder_outputs.past_key_values, decoder_outputs.past_key_values
            )
        )


@auto_docstring
@dataclass
class XYZCausalLMOutputWithPast(CausalLMOutputWithPast):
    """
    Output of [`XYZCausalLMOutputWithPast`].

    Extends [`CausalLMOutputWithPast`] with character-granularity logits and
    hidden states for the auxiliary next-character prediction head.

    char_logits (`torch.FloatTensor` of shape `(batch, char_seq_len, vocab_size)`, *optional*):
        Next-character prediction logits from the char decoder LM head.
    char_past_key_values (`Tuple[Cache, Cache]`, *optional*):
        Pair of KV caches — one for the char encoder, one for the char
        decoder — used during incremental generation.
    char_hidden_states (`tuple[torch.FloatTensor]`, *optional*):
        Char-decoder hidden states at every layer (returned when
        ``output_hidden_states=True``).
    char_attentions (`tuple[torch.FloatTensor]`, *optional*):
        Char-decoder self-attention weights at every layer (returned when
        ``output_attentions=True``).
    cle_logits (`torch.FloatTensor` of shape `(batch, char_seq_len, 26)`, *optional*):
        CLE (Cα–Local–Environment) classification logits from the
        auxiliary CLE head.
    distogram_logits (`torch.FloatTensor` of shape `(batch, char_seq_len, char_seq_len, distogram_bins_num)`, *optional*):
        Pairwise residue–residue distance-bin logits from the distogram
        head.
    """

    char_logits: torch.FloatTensor | None = None
    char_past_key_values: Tuple[Cache, Cache] | None = None
    char_hidden_states: tuple[torch.FloatTensor, ...] | None = None
    char_attentions: tuple[torch.FloatTensor, ...] | None = None
    cle_logits: torch.FloatTensor | None = None
    distogram_logits: torch.FloatTensor | None = None


class XYZForCausalLM(LlamaForCausalLM):
    def __init__(self, config: XYZConfig):
        super().__init__(config)

        if config.has_char_lm_head:
            self.char_lm_head = nn.Linear(
                config.char_hidden_size, config.vocab_size, bias=False
            )
        if config.has_cle_lm_head:
            self.cle_lm_head = nn.Linear(config.char_hidden_size, 26, bias=False)
        if config.has_distogram_lm_head:
            # self.distogram_head = nn.Sequential(
            #     nn.Linear(config.char_hidden_size, config.char_hidden_size, bias=False),
            #     nn.GELU(),
            #     nn.Linear(config.char_hidden_size, config.distogram_bins_num, bias=False),
            # )
            self.distogram_head = XYZDistogram(config)

    @contextlib.contextmanager
    def num_items_in_batch(self, kwargs, amplify: float | None):
        if "num_items_in_batch" in kwargs:
            num_items_in_batch = kwargs["num_items_in_batch"]
            if amplify is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch * amplify
            else:
                kwargs["num_items_in_batch"] = None
        yield kwargs
        if "num_items_in_batch" in kwargs:
            kwargs["num_items_in_batch"] = num_items_in_batch

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        char_input_ids: torch.LongTensor | None = None,
        char_attention_mask: torch.Tensor | None = None,
        char_position_ids: torch.LongTensor | None = None,
        char_past_key_values: Tuple[Cache, Cache] | None = None,
        char_inputs_embeds: torch.FloatTensor | None = None,
        char_labels: torch.LongTensor | None = None,
        char_logits_to_keep: int | torch.Tensor = 0,
        cle_labels: torch.LongTensor | None = None,
        distogram_labels: torch.LongTensor | None = None,
        repr_char_idx: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> XYZCausalLMOutputWithPast:
        """
        Causal language-modeling forward pass with dual prediction heads.

        Runs the U-Net backbone ([`XYZModel`]) and applies two LM heads:

        - **Next-token** — ``lm_head`` on trunk output → token logits
        - **Next-character** — ``char_lm_head`` on char-decoder output → char logits

        Losses are summed when both *labels* and *char_labels* are provided.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch, token_seq_len)`, *optional*):
                BPE token indices.
            attention_mask (`torch.Tensor`, *optional*):
                Token-level attention mask.
            position_ids (`torch.LongTensor`, *optional*):
                Token position indices.
            past_key_values (`Cache`, *optional*):
                Trunk KV cache.
            inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed token embeddings.
            labels (`torch.LongTensor` of shape `(batch, token_seq_len)`, *optional*):
                Ground-truth token ids for next-token loss.  Shifted internally.
            logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                If an int, only compute logits for the last ``logits_to_keep``
                token positions (saves memory during training).  If a tensor,
                used as an explicit slice / boolean mask.
            char_input_ids (`torch.LongTensor` of shape `(batch, char_seq_len)`, *optional*):
                Character-level token indices.
            char_attention_mask (`torch.Tensor`, *optional*):
                Character-level attention mask.
            char_position_ids (`torch.LongTensor`, *optional*):
                Character position indices.
            char_past_key_values (`Tuple[Cache, Cache]`, *optional*):
                Char encoder / decoder KV caches.
            char_inputs_embeds (`torch.FloatTensor`, *optional*):
                Pre-computed char embeddings.
            char_labels (`torch.LongTensor` of shape `(batch, char_seq_len)`, *optional*):
                Ground-truth char ids for next-character loss.  Shifted internally.
            char_logits_to_keep (`int` or `torch.Tensor`, *optional*, defaults to 0):
                Same as ``logits_to_keep`` but for the char-level heads
                (``char_lm_head``, ``cle_lm_head``).
            cle_labels (`torch.LongTensor` of shape `(batch, char_seq_len)`, *optional*):
                Ground-truth CLE class ids for the CLE classification loss.
            distogram_labels (`torch.LongTensor` of shape `(batch, char_seq_len, char_seq_len)`, *optional*):
                Ground-truth distance-bin indices for the distogram head.
                Each value is a bin index in ``[0, distogram_bins_num)``.
            repr_char_idx (`torch.LongTensor`, *optional*):
                Token → representative-char mapping.
            use_cache (`bool`, *optional*):
                Whether to populate and return KV caches.

        Returns:
            [`XYZForCausalLMOutput`]:
                - **loss** — combined next-token + next-character + aux CE loss.
                - **logits** — next-token logits (used by `generate()`).
                - **char_logits** — next-character logits.
                - **cle_logits** — CLE classification logits.
                - **distogram_logits** — pairwise distance-bin logits
                  of shape `(batch, char_seq_len, char_seq_len, distogram_bins_num)`.
        """
        outputs: XYZModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            char_input_ids=char_input_ids,
            char_attention_mask=char_attention_mask,
            char_position_ids=char_position_ids,
            char_past_key_values=char_past_key_values,
            char_inputs_embeds=char_inputs_embeds,
            repr_char_idx=repr_char_idx,
            use_cache=use_cache,
            **kwargs,
        )

        # trunk
        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(
            logits_to_keep, int
        ) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        # decoder
        char_hidden_states = outputs.char_last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-char_logits_to_keep, None) if isinstance(
            char_logits_to_keep, int
        ) else char_logits_to_keep

        aux_loss = []

        char_logits = None
        if self.config.has_char_lm_head:
            char_logits = self.char_lm_head(char_hidden_states[:, slice_indices, :])
            if char_labels is not None:
                with self.num_items_in_batch(
                    kwargs, outputs.char_last_hidden_state.size(-2) / outputs.last_hidden_state.size(-2)
                ):
                    aux_loss.append(
                        self.loss_function(
                            logits=char_logits,
                            labels=char_labels,
                            vocab_size=self.config.vocab_size,
                            **kwargs,
                        )
                    )
        cle_logits = None
        if self.config.has_cle_lm_head:
            cle_logits = self.cle_lm_head(char_hidden_states[:, slice_indices, :])
            if cle_labels is not None:
                with self.num_items_in_batch(
                    kwargs, outputs.char_last_hidden_state.size(-2) / outputs.last_hidden_state.size(-2)
                ):
                    aux_loss.append(
                        self.loss_function(
                            logits=cle_logits,
                            labels=cle_labels,
                            vocab_size=26,
                            shift_labels=cle_labels,  # FIX: Fake the `ForCausalLMLoss`
                            **kwargs,
                        )
                    )

        distogram_logits = None
        if self.config.has_distogram_lm_head:
            # pairwise outer sum: (B, L, H) -> (B, L, L, H) -> (B, L, L, num_bins)
            # h = char_hidden_states
            # pair_repr = h.unsqueeze(1) + h.unsqueeze(2)
            distogram_logits = self.distogram_head(char_hidden_states[:, slice_indices, :])
            if distogram_labels is not None:
                ignore_index = kwargs.get("ignore_index", -100)
                # shift_distogram_labels = nn.functional.pad(
                #     distogram_labels, (0, 1, 0, 1), value=ignore_index
                # )
                # shift_distogram_labels = shift_distogram_labels[..., 1:, 1:].contiguous()
                with self.num_items_in_batch(
                    kwargs, outputs.char_last_hidden_state.size(-2) / outputs.last_hidden_state.size(-2)
                ):
                    num_items_in_batch = kwargs.get("num_items_in_batch")
                if num_items_in_batch is None:
                    aux_loss.append(
                        self.loss_function(
                            logits=distogram_logits,
                            labels=distogram_labels,
                            vocab_size=self.config.distogram_bins_num,
                            shift_labels=distogram_labels,  # FIX: Fake the `ForCausalLMLoss`
                            **kwargs,
                        )
                    )
                else:
                    eps = 1e-8
                    distogram_loss = nn.functional.cross_entropy(
                        distogram_logits.reshape(-1, self.config.distogram_bins_num),
                        distogram_labels.reshape(-1),
                        ignore_index=ignore_index,
                        reduction="none",
                    ).view_as(distogram_labels)
                    distogram_loss = (
                        distogram_loss.sum(-1) / ((distogram_labels != ignore_index).sum(-1) + eps)
                    ).sum()
                    # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
                    if torch.is_tensor(num_items_in_batch):
                        num_items_in_batch = num_items_in_batch.to(distogram_loss.device)
                    aux_loss.append(distogram_loss / num_items_in_batch)
                # aux_loss.append(
                #     self.loss_function(
                #         logits=distogram_logits,
                #         labels=distogram_labels,
                #         vocab_size=self.config.distogram_bins_num,
                #         labels=distogram_labels,
                #         **kwargs,
                #     )
                # )

        if loss is not None and aux_loss:
            loss = loss + sum(aux_loss)
        elif aux_loss:
            loss = sum(aux_loss)

        return XYZCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            char_logits=char_logits,
            char_past_key_values=outputs.char_past_key_values,
            char_hidden_states=outputs.char_hidden_states,
            char_attentions=outputs.char_attentions,
            cle_logits=cle_logits,
            distogram_logits=distogram_logits,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        return None


class XYZForSequenceClassification(LlamaForSequenceClassification):
    pass


@auto_docstring
@dataclass
class XYZTokenClassifierOutput(TokenClassifierOutput):
    """
    Output of [`XYZForTokenClassification`].

    Extends [`TokenClassifierOutput`] with character-granularity logits and
    hidden states for the auxiliary next-character prediction head.

    lm_output (`XYZCausalLMOutputWithPast`, *optional*):
        Full causal-LM output from the U-Net backbone, including next-token
        logits (``logits``) and next-character logits (``char_logits``)
    cle_logits (`torch.FloatTensor` of shape `(batch, char_seq_len, 26)`, *optional*):
        CLE (Cα–Local–Environment) classification logits from the
        auxiliary CLE head.
    distogram_logits (`torch.FloatTensor` of shape `(batch, char_seq_len, char_seq_len, distogram_bins_num)`, *optional*):
        Pairwise residue–residue distance-bin logits from the distogram
        head.
    """

    lm_output: XYZCausalLMOutputWithPast | None = None
    cle_logits: torch.FloatTensor | None = None
    distogram_logits: torch.FloatTensor | None = None


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

    def __init__(
        self, tokenizer, text_column: str = "text", features: List[str] = None, **kwargs
    ) -> None:
        tokenizer.add_special_tokens({"additional_special_tokens": self.FIM_TOKENS})
        logger.info(
            f"Added FIM tokens: {self.FIM_TOKENS} to tokenizer. "
            f"(vocabu size: {len(tokenizer)})"
        )

        super().__init__(tokenizer, **kwargs)

        self.text_column = text_column
        self.features = features
        self.ignore_index = kwargs.get("ignore_index", -100)

        self.distogram_bins = torch.linspace(
            kwargs.get("distogram_bins_min",  2.3125),
            kwargs.get("distogram_bins_max", 21.6875),
            steps=kwargs.get("distogram_bins_num", 64) - 1,
        )

    def __call__(
        self,
        examples: Dict,
        *,
        bpe_dropout: Optional[float] = None,
        char_apply: bool = False,
        fim_apply: bool = False,
        fim_spm_rate: float = 0.5,
        fim_sft_style: bool = False,
        max_length: Optional[int] = None,
        **kwargs
    ) -> Dict:
        examples = self.apply_crop(examples, max_length=max_length)
        if fim_apply:
            examples = self.apply_fim(examples, spm_rate=fim_spm_rate)
        examples = self.apply_wrap(examples)

        def tokenize_with_dropout(dropout, prefix=""):
            with attr(self.tokenizer.backend_tokenizer.model, dropout=dropout):
                tokenized = self.tokenizer(
                    examples[self.text_column],
                    truncation=True,
                    return_tensors="pt",
                    padding=True,
                    return_offsets_mapping=char_apply,
                )
            tokenized["labels"] = tokenized["input_ids"].where(
                tokenized["attention_mask"] > 0, self.ignore_index
            )
            if fim_apply and fim_sft_style:
                middle_pos = (
                    tokenized["labels"] == self.tokenizer.convert_tokens_to_ids(
                        self.FIM_MIDDLE
                    )
                ).cumsum(1)
                tokenized["labels"] = tokenized["labels"].where(
                    middle_pos.cumsum(1) > 1, self.ignore_index  # the <fim_middle> is excluded
                )
            return {f"{prefix}{k}": v for k, v in tokenized.items()}

        tokenized = tokenize_with_dropout(bpe_dropout, prefix="")
        if char_apply:
            tokenized.update(tokenize_with_dropout(1.0, prefix="char_"))

            # Align characters with tokenized *text*
            tokenized["char_to_token_idx"] = tokenized["char_attention_mask"].new_zeros(
                tokenized["char_attention_mask"].size()
            )
            tokenized["repr_char_idx"] = tokenized["attention_mask"].new_zeros(
                tokenized["attention_mask"].size()
            )
            # FIX: use torch.searchsorted instead
            for k in range(len(examples[self.text_column])):
                i, j = 0, 0
                while i < len(tokenized["offset_mapping"][k]
                             ) and j < len(tokenized["char_offset_mapping"][k]):
                    if tokenized["attention_mask"][k, i] == 0:
                        i += 1
                    elif tokenized["char_attention_mask"][k, j] == 0:
                        j += 1
                    else:
                        si, ei = tokenized["offset_mapping"][k, i]
                        sj, ej = tokenized["char_offset_mapping"][k, j]
                        if si <= sj and ej <= ei:
                            tokenized["char_to_token_idx"][k, j] = i
                            if ej == ei:
                                tokenized["repr_char_idx"][k, i] = j
                            j += 1
                        elif ei <= sj:
                            i += 1
                        else:
                            assert False, (text[k], i, j)

            del tokenized["char_offset_mapping"]
            del tokenized["offset_mapping"]

        # copy features
        if self.features is not None:
            def feat_collate(feat, is_label: bool = False):
                if isinstance(feat, tuple):
                    return tuple(feat_collate(feat[t]) for t in range(len(feat)))
                max_len = max(feat[k].shape[0] for k in range(len(feat)))
                for k in range(len(feat)):
                    pad = feat[k].new_full(
                        (max_len - feat[k].shape[0], *feat[k].shape[1:]),
                        self.ignore_index if is_label else 0,
                    )
                    feat[k] = torch.cat((feat[k], pad))
                return torch.stack(feat)

            for column in self.features:
                if column in examples:
                    tokenized[column] = feat_collate(
                        examples[column], "labels" in column
                    )

            if "distogram_labels" in self.features:
                if "distogram_labels" in tokenized:
                    tokenized["distogram_labels"] = self.to_distogram(
                        *tokenized["distogram_labels"]
                    )

        return tokenized

    def apply_fim(self, examples: Dict, spm_rate: float = 0.5) -> Dict:
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
            if self.features is not None:
                def feat_fim(feat, is_label: bool = False):
                    pad = feat.new_full(
                        (1, *feat.shape[1:]), self.ignore_index if is_label else 0
                    )
                    if is_spm:
                        return torch.cat(
                            (
                                pad, feat[cut2:, ...],
                                pad, feat[:cut1, ...],
                                pad, feat[cut1:cut2, ...],
                            )
                        )
                    return torch.cat(
                        (
                            pad, feat[:cut1, ...],
                            pad, feat[cut2:, ...],
                            pad, feat[cut1:cut2, ...],
                        )
                    )

                for column in self.features:
                    if column in examples:
                        if isinstance(examples[column], tuple):
                            for t in range(len(examples[column])):
                                examples[column][t][idx] = feat_fim(
                                    examples[column][t][idx]
                                )
                        else:
                            examples[column][idx] = feat_fim(
                                examples[column][idx], "labels" in column
                            )
        return examples

    def apply_crop(self, examples: Dict, max_length: Optional[int] = None) -> Dict:
        for idx, text in enumerate(examples[self.text_column]):
            n = len(text)
            if max_length and max_length < n:
                cut = random.randint(0, n - max_length)
                text = text[cut:cut + max_length]
                examples[self.text_column][idx] = text
                if self.features is not None:
                    def feat_crop(feat):
                        return feat[cut:cut + max_length, ...]

                    for column in self.features:
                        if column in examples:
                            if isinstance(examples[column], tuple):
                                for t in range(len(examples[column])):
                                    examples[column][t][idx] = feat_crop(
                                        examples[column][t][idx]
                                    )
                            else:
                                examples[column][idx] = feat_crop(examples[column][idx])
        return examples

    def apply_wrap(self, examples: Dict) -> Dict:
        for idx, text in enumerate(examples[self.text_column]):
            examples[self.text_column][idx] = (
                f"{self.tokenizer.bos_token}{text}{self.tokenizer.eos_token}"
            )
            if self.features is not None:
                def feat_wrap(feat, is_label: bool = False):
                    pad = feat.new_full(
                        (1, *feat.shape[1:]), self.ignore_index if is_label else 0
                    )
                    return torch.cat((pad, feat, pad))

                for column in self.features:
                    if column in examples:
                        if isinstance(examples[column], tuple):
                            for t in range(len(examples[column])):
                                examples[column][t][idx] = feat_wrap(
                                    examples[column][t][idx]
                                )
                        else:
                            examples[column][idx] = feat_wrap(
                                examples[column][idx], "labels" in column
                            )
        return examples

    def to_distogram(self, pseudo_beta, pseudo_beta_mask): 
        distogram_labels = torch.cdist(pseudo_beta, pseudo_beta)
        distogram_labels = (distogram_labels[..., None] > self.distogram_bins).sum(-1)
        return distogram_labels.where(
            pseudo_beta_mask[..., :, None] * pseudo_beta_mask[..., None, :],
            self.ignore_index
        )


__all__ = [
    "XYZPreTrainedModel",
    "XYZModelOutputWithPast",
    "XYZModel",
    "XYZCausalLMOutputWithPast",
    "XYZForCausalLM",
    "XYZForSequenceClassification",
    "XYZTokenClassifierOutput",
    "XYZForTokenClassification",
    "XYZConfig",
    "XYZProcessor",
]
