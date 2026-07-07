"""U-Net style protein language model.

Three-stage architecture:
  -> AA encoder         (residue granularity)
  -> token transformmer (token granularity)
  -> AA decoder         (residue granularity)
  => predict next amino acid
"""

import random
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaDecoderLayer
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

AA_VOCAB = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_VOCAB)}
IDX_TO_AA = {i: aa for i, aa in enumerate(AA_VOCAB)}
AA_VOCAB_SIZE = 20


# ======================================================================
# Config
# ======================================================================
class ProXYZConfig(PretrainedConfig):
    model_type = "proxyz"

    def __init__(
        self,
        vocab_size: int = 30000,
        hidden_size: int = 2048,
        intermediate_size: int = 5632,
        num_attention_heads: int = 16,
        num_key_value_heads: int = 4,
        aa_vocab_size: int = AA_VOCAB_SIZE,
        aa_encoder_layers: int = 4,
        bpe_encoder_layers: int = 12,
        aa_decoder_layers: int = 4,
        max_position_embeddings: int = 4096,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 10000.0,
        use_skip_connection: bool = True,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.aa_vocab_size = aa_vocab_size
        self.aa_encoder_layers = aa_encoder_layers
        self.bpe_encoder_layers = bpe_encoder_layers
        self.aa_decoder_layers = aa_decoder_layers
        self.max_position_embeddings = max_position_embeddings
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.use_skip_connection = use_skip_connection
        super().__init__(**kwargs)


# ======================================================================
# Down / Up sampling
# ======================================================================
class _Downsample(nn.Module):
    """Pool amino-acid representations → BPE-token representations."""

    def forward(self, aa_hidden: torch.Tensor, alignments: torch.Tensor) -> torch.Tensor:
        B, M, _ = alignments.shape
        D = aa_hidden.shape[-1]
        device = aa_hidden.device

        starts = alignments[:, :, 0]  # (B, M)
        ends = alignments[:, :, 1]    # (B, M)
        max_len = aa_hidden.shape[1]

        pos = torch.arange(max_len, device=device).unsqueeze(0).unsqueeze(0)
        mask = (pos >= starts.unsqueeze(-1)) & (pos < ends.unsqueeze(-1))

        counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (B, M, 1)
        pooled = (aa_hidden.unsqueeze(1) * mask.unsqueeze(-1).float()).sum(dim=2) / counts
        return pooled


class _Upsample(nn.Module):
    """Expand BPE representations → amino-acid granularity with shift+1 for causality."""

    def forward(self, bpe_hidden: torch.Tensor, alignments: torch.Tensor, target_len: int) -> torch.Tensor:
        B, M, D = bpe_hidden.shape
        device = bpe_hidden.device

        starts = alignments[:, :, 0]  # (B, M)

        # Map each AA position → its BPE token index, then shift -1 for causality
        pos = torch.arange(target_len, device=device).unsqueeze(0).unsqueeze(0)
        bpe_idx = (pos >= starts.unsqueeze(-1)).long().sum(dim=1) - 1  # (B, N)

        gather_idx = bpe_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, D)
        result = torch.gather(bpe_hidden, 1, gather_idx)

        valid = (bpe_idx >= 0).unsqueeze(-1).float()
        return result * valid


# ======================================================================
# Model
# ======================================================================
class ProXYZForCausalLM(PreTrainedModel):
    """U-Net style pLM: AA encoder -> token transformer -> AA decoder => predict next AA."""

    config_class = ProXYZConfig
    base_model_prefix = "proxyz"
    supports_gradient_checkpointing = False

    def __init__(self, config: ProXYZConfig):
        super().__init__(config)

        # --- Embeddings ---
        self.aa_embedding = nn.Embedding(config.aa_vocab_size, config.hidden_size)
        self.bpe_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # --- Shared LlamaDecoderLayer config ---
        layer_cfg = LlamaConfig(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            tie_word_embeddings=False,
        )

        # --- AA Encoder (residue granularity, causal) ---
        self.aa_encoder = nn.ModuleList(
            [LlamaDecoderLayer(layer_cfg, layer_idx=i) for i in range(config.aa_encoder_layers)]
        )
        self.aa_encoder_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- Down / Up ---
        self.downsample = _Downsample()
        self.upsample = _Upsample()

        # --- BPE Encoder (token granularity, causal) ---
        self.bpe_encoder = nn.ModuleList(
            [LlamaDecoderLayer(layer_cfg, layer_idx=i) for i in range(config.bpe_encoder_layers)]
        )
        self.bpe_encoder_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- Skip projection (concat → hidden_size) ---
        if config.use_skip_connection:
            self.skip_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)

        # --- AA Decoder (residue granularity, causal) ---
        self.aa_decoder = nn.ModuleList(
            [LlamaDecoderLayer(layer_cfg, layer_idx=i) for i in range(config.aa_decoder_layers)]
        )
        self.aa_decoder_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # --- LM Head → 20 amino acids ---
        self.lm_head = nn.Linear(config.hidden_size, config.aa_vocab_size, bias=False)

        # Initialize weights
        self.post_init()

    # ------------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.RMSNorm):
            nn.init.ones_(module.weight)

    # ------------------------------------------------------------------
    def forward(
        self,
        aa_input_ids: torch.Tensor,
        bpe_input_ids: torch.Tensor,
        alignments: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        aa_attention_mask: Optional[torch.Tensor] = None,
        bpe_attention_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithPast:
        """
        Args:
            aa_input_ids:      (B, N)  amino-acid indices 0-19
            bpe_input_ids:     (B, M)  BPE token ids (incl. BOS/EOS)
            alignments:        (B, M, 2)  AA start/end per BPE token
            labels:            (B, N)  next-AA targets, -100 = ignore
            aa_attention_mask: (B, N)  1=real, 0=pad  (None → all real)
            bpe_attention_mask:(B, M)  1=real, 0=pad  (None → all real)
        """
        B, N = aa_input_ids.shape
        M = bpe_input_ids.shape[1]
        device = aa_input_ids.device

        # 2-D masks: LlamaDecoderLayer internally builds causal + padding mask
        if aa_attention_mask is None:
            aa_attention_mask = aa_input_ids.new_ones(B, N)
        if bpe_attention_mask is None:
            bpe_attention_mask = bpe_input_ids.new_ones(B, M)

        aa_pos = torch.arange(N, device=device).unsqueeze(0)
        bpe_pos = torch.arange(M, device=device).unsqueeze(0)

        # ====== AA Encoder ======
        h = self.aa_embedding(aa_input_ids)
        for layer in self.aa_encoder:
            h = layer(h, attention_mask=aa_attention_mask, position_ids=aa_pos)[0]
        aa_hidden = self.aa_encoder_norm(h)  # (B, N, D)

        # ====== Downsample: AA → BPE ======
        bpe_from_aa = self.downsample(aa_hidden, alignments)  # (B, M, D)

        # ====== BPE Encoder ======
        h = self.bpe_embedding(bpe_input_ids) + bpe_from_aa
        for layer in self.bpe_encoder:
            h = layer(h, attention_mask=bpe_attention_mask, position_ids=bpe_pos)[0]
        bpe_hidden = self.bpe_encoder_norm(h)  # (B, M, D)

        # ====== Upsample: BPE → AA (shift +1 for causality) ======
        upsampled = self.upsample(bpe_hidden, alignments, N)  # (B, N, D)

        # ====== Skip connection ======
        if self.config.use_skip_connection:
            h = self.skip_proj(torch.cat([aa_hidden, upsampled], dim=-1))
        else:
            h = upsampled

        # ====== AA Decoder ======
        for layer in self.aa_decoder:
            h = layer(h, attention_mask=aa_attention_mask, position_ids=aa_pos)[0]
        aa_decoded = self.aa_decoder_norm(h)  # (B, N, D)

        # ====== LM Head ======
        logits = self.lm_head(aa_decoded)  # (B, N, 20)

        # ====== Loss ======
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithPast(loss=loss, logits=logits)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        tokenizer,
        prompt_aa: str = "",
        max_length: int = 500,
        temperature: float = 1.0,
        top_k: int = 1,
    ) -> str:
        """Autoregressively generate amino acids one at a time."""
        device = next(self.parameters()).device

        aa_seq: List[int] = [AA_TO_IDX[aa] for aa in prompt_aa if aa in AA_TO_IDX]
        if not aa_seq:
            aa_seq = [random.randint(0, AA_VOCAB_SIZE - 1)]

        for _ in range(max_length - len(aa_seq)):
            N = len(aa_seq)
            aa_tensor = torch.tensor([aa_seq], dtype=torch.long, device=device)

            # Build BPE tokens from current AA string
            aa_str = "".join(IDX_TO_AA[a] for a in aa_seq)
            bpe_wrapped = f"{tokenizer.bos_token}{aa_str}{tokenizer.eos_token}"
            bpe_enc = tokenizer(bpe_wrapped, return_offsets=True)
            bpe_ids = torch.tensor([bpe_enc["input_ids"]], dtype=torch.long, device=device)

            # Build alignment (content only, skip BOS/EOS)
            offsets = bpe_enc["offsets"]
            M_bpe = len(bpe_enc["input_ids"])
            align = [[0, 0]] * M_bpe
            for j in range(1, M_bpe - 1):
                s, e = offsets[j]
                if e > 0 and s < len(aa_str):
                    align[j] = [max(s, 0), min(e, len(aa_str))]

            alignments = torch.tensor([align], dtype=torch.long, device=device)

            outputs = self.forward(aa_tensor, bpe_ids, alignments)
            next_logits = outputs.logits[0, -1, :].clone()

            if temperature != 1.0:
                next_logits = next_logits / temperature
            if top_k > 1:
                topk_vals, _ = torch.topk(next_logits, top_k)
                next_logits[next_logits < topk_vals[-1]] = float("-inf")

            probs = F.softmax(next_logits, dim=-1)
            next_aa = torch.multinomial(probs, 1).item()
            aa_seq.append(next_aa)

        return "".join(IDX_TO_AA.get(a, "") for a in aa_seq)


# ======================================================================
# Utility
# ======================================================================
def build_alignments(tokenizer, text: str) -> Tuple[torch.Tensor, int]:
    """Tokenize *text* (without BOS/EOS) and return (alignments, num_bpe_tokens)."""
    enc = tokenizer(text, return_offsets=True, add_special_tokens=False)
    offsets = enc["offsets"]
    M = len(enc["input_ids"])
    alignments = [[0, 0]] * M
    for j, (s, e) in enumerate(offsets):
        alignments[j] = [s, e]
    return torch.tensor(alignments, dtype=torch.long), M
