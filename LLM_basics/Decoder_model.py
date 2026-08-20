"""Decoder-only Transformer language model.

The model in this module deliberately stops at token tensors: tokenization and
data loading belong to the caller.  This keeps the model useful both in a
training script and in small CPU unit tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from .Modules import FeedForward, MoE, MultiHeadAttention, RMSNorm


@dataclass(slots=True)
class TransformerConfig:
    """Hyperparameters for :class:`TransformerLM`.

    ``use_moe`` is intentionally only an extension point for now.  Passing a
    ``moe_factory`` lets a future MoE implementation be plugged in without
    changing the Transformer block or checkpoint structure.
    """

    vocab_size: int
    context_length: int
    d_model: int
    num_layers: int
    num_heads: int
    d_ff: int
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    bias: bool = False
    tie_embeddings: bool = False
    use_rope: bool = True
    use_moe: bool = False

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "vocab_size",
            "context_length",
            "d_model",
            "num_layers",
            "num_heads",
            "d_ff",
        )
        for field_name in positive_integer_fields:
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer, got {value!r}")

        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


# Friendly aliases for code that uses a more model-specific config name.
DecoderConfig = TransformerConfig
ModelConfig = TransformerConfig


class TransformerBlock(nn.Module):
    """Pre-normalization decoder block with causal self-attention and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        *,
        dropout: float = 0.0,
        norm_eps: float = 1e-5,
        bias: bool = False,
        use_rope: bool = True,
        use_moe: bool = False,
        moe_factory: Callable[..., nn.Module] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, num_heads, and d_ff must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.attn_norm = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)
        self.attn = MultiHeadAttention(
            d_model,
            num_heads,
            use_rope=use_rope,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            dropout=dropout,
            bias=bias,
        )
        self.ffn_norm = RMSNorm(d_model, eps=norm_eps, device=device, dtype=dtype)

        if use_moe:
            # Keep the architectural slot without pretending that routing or
            # expert balancing has already been implemented.
            self.ffn = (
                MoE(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
                if moe_factory is None
                else moe_factory(
                    d_model=d_model,
                    d_ff=d_ff,
                    device=device,
                    dtype=dtype,
                )
            )
        else:
            self.ffn = FeedForward(
                d_model,
                d_ff,
                device=device,
                dtype=dtype,
                bias=bias,
                dropout=dropout,
            )

        # Residual dropout is distinct from attention-probability/FFN dropout.
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply one block while preserving ``(batch, sequence, d_model)``.

        ``attention_mask`` is forwarded unchanged to self-attention.  In
        particular, a ``(batch, sequence)`` mask acts as a key-padding mask;
        the attention module combines it with its causal mask.
        """

        attention_output = self.attn(
            self.attn_norm(x),
            token_positions=token_positions,
            mask=attention_mask,
        )
        # ``use_cache`` is false here, so the public attention API returns a tensor.
        if isinstance(attention_output, tuple):  # defensive for custom attention modules
            attention_output = attention_output[0]
        x = x + self.residual_dropout(attention_output)
        x = x + self.residual_dropout(self.ffn(self.ffn_norm(x)))
        return x


# Another common name used in Transformer implementations and course adapters.
DecoderBlock = TransformerBlock


class TransformerLM(nn.Module):
    """A causal decoder-only Transformer that maps token ids to next-token logits.

    The constructor accepts either a :class:`TransformerConfig` as its first
    argument, ``config=...``, or the individual hyperparameters.  Supporting
    both forms keeps experiments concise while retaining an explicit config for
    saved runs.
    """

    def __init__(
        self,
        vocab_size: int | TransformerConfig | None = None,
        context_length: int | None = None,
        d_model: int | None = None,
        num_layers: int | None = None,
        num_heads: int | None = None,
        d_ff: int | None = None,
        rope_theta: float = 10_000.0,
        *,
        config: TransformerConfig | None = None,
        norm_eps: float = 1e-5,
        dropout: float = 0.0,
        bias: bool = False,
        tie_embeddings: bool = False,
        use_rope: bool = True,
        use_moe: bool = False,
        moe_factory: Callable[..., nn.Module] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        positional_config = vocab_size if isinstance(vocab_size, TransformerConfig) else None
        if positional_config is not None:
            if config is not None:
                raise TypeError("pass the model config either positionally or by keyword, not both")
            if any(
                value is not None
                for value in (context_length, d_model, num_layers, num_heads, d_ff)
            ):
                raise TypeError(
                    "individual model arguments cannot be combined with a positional config"
                )
            config = positional_config
        elif config is None:
            missing = {
                name
                for name, value in (
                    ("vocab_size", vocab_size),
                    ("context_length", context_length),
                    ("d_model", d_model),
                    ("num_layers", num_layers),
                    ("num_heads", num_heads),
                    ("d_ff", d_ff),
                )
                if value is None
            }
            if missing:
                names = ", ".join(sorted(missing))
                raise TypeError(f"missing required model arguments: {names}")
            config = TransformerConfig(
                vocab_size=vocab_size,  # type: ignore[arg-type]
                context_length=context_length,  # type: ignore[arg-type]
                d_model=d_model,  # type: ignore[arg-type]
                num_layers=num_layers,  # type: ignore[arg-type]
                num_heads=num_heads,  # type: ignore[arg-type]
                d_ff=d_ff,  # type: ignore[arg-type]
                rope_theta=rope_theta,
                norm_eps=norm_eps,
                dropout=dropout,
                bias=bias,
                tie_embeddings=tie_embeddings,
                use_rope=use_rope,
                use_moe=use_moe,
            )
        elif vocab_size is not None or any(
            value is not None
            for value in (context_length, d_model, num_layers, num_heads, d_ff)
        ):
            raise TypeError("individual model arguments cannot be combined with config")

        self.config = config
        factory_kwargs: dict[str, Any] = {"device": device, "dtype": dtype}
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
            **factory_kwargs,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    max_seq_len=config.context_length,
                    theta=config.rope_theta,
                    dropout=config.dropout,
                    norm_eps=config.norm_eps,
                    bias=config.bias,
                    use_rope=config.use_rope,
                    use_moe=config.use_moe,
                    moe_factory=moe_factory,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = RMSNorm(
            config.d_model,
            eps=config.norm_eps,
            device=device,
            dtype=dtype,
        )
        self.lm_head = nn.Linear(
            config.d_model,
            config.vocab_size,
            bias=False,
            **factory_kwargs,
        )

        self._reset_parameters()
        if config.tie_embeddings:
            # Parameter sharing saves memory and is common in decoder-only LMs.
            self.lm_head.weight = self.token_embeddings.weight

    def _reset_parameters(self) -> None:
        """Use a small normal initialization suitable for language models."""

        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        if not self.config.tie_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    @property
    def vocab_size(self) -> int:
        return self.config.vocab_size

    @property
    def context_length(self) -> int:
        return self.config.context_length

    def forward(
        self,
        input_ids: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return logits of shape ``(batch, sequence, vocab_size)``.

        ``attention_mask`` accepts the same layouts as
        :class:`MultiHeadAttention`, including the common ``(batch, sequence)``
        key-padding mask.  Boolean/integer masks use non-zero entries for
        positions that may be attended to.
        """

        if input_ids.ndim != 2:
            raise ValueError(
                "input_ids must have shape (batch, sequence), "
                f"got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must contain integer token ids")

        sequence_length = input_ids.size(1)
        if sequence_length == 0:
            raise ValueError("input_ids must contain at least one token")
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )
        if token_positions is not None:
            if not isinstance(token_positions, torch.Tensor):
                raise TypeError("token_positions must be a torch.Tensor")
            if (
                token_positions.dtype == torch.bool
                or token_positions.is_floating_point()
                or token_positions.is_complex()
            ):
                raise TypeError("token_positions must contain integer indices")
            if token_positions.ndim not in (1, 2):
                raise ValueError(
                    "token_positions must have shape (sequence,) or "
                    "(batch, sequence)"
                )
            if token_positions.shape[-1] != sequence_length:
                raise ValueError(
                    "the last dimension of token_positions must match the sequence"
                )
            if (
                token_positions.ndim == 2
                and token_positions.shape[0] not in (1, input_ids.shape[0])
            ):
                raise ValueError(
                    "the token_positions batch dimension must be 1 or match input_ids"
                )
            if token_positions.device != input_ids.device:
                raise ValueError("token_positions and input_ids must share a device")
            if torch.any(token_positions < 0).item():
                raise ValueError("token_positions cannot contain negative indices")

        hidden_states = self.token_embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                token_positions=token_positions,
                attention_mask=attention_mask,
            )
        return self.lm_head(self.final_norm(hidden_states))

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        *,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_token_id: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend a prompt without taking ownership of tokenization.

        ``temperature=0`` performs greedy decoding.  For long generations the
        oldest tokens are cropped to the configured context window.
        """

        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("input_ids must have shape (batch, non_empty_sequence)")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")
        if temperature < 0:
            raise ValueError("temperature must be non-negative")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")
        if eos_token_id is not None and not 0 <= eos_token_id < self.config.vocab_size:
            raise ValueError("eos_token_id is outside the vocabulary")

        was_training = self.training
        self.eval()
        tokens = input_ids
        finished = torch.zeros(tokens.size(0), dtype=torch.bool, device=tokens.device)
        try:
            for _ in range(max_new_tokens):
                model_input = tokens[:, -self.config.context_length :]
                next_logits = self(model_input)[:, -1, :]

                if temperature == 0:
                    next_token = next_logits.argmax(dim=-1, keepdim=True)
                else:
                    next_logits = next_logits / temperature
                    if top_k is not None:
                        k = min(top_k, next_logits.size(-1))
                        cutoff = torch.topk(next_logits, k=k, dim=-1).values[:, -1:]
                        next_logits = next_logits.masked_fill(
                            next_logits < cutoff,
                            -torch.inf,
                        )
                    probabilities = torch.softmax(next_logits, dim=-1)
                    next_token = torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=generator,
                    )

                if eos_token_id is not None:
                    # Finished rows keep emitting EOS while other rows continue.
                    next_token = torch.where(
                        finished[:, None],
                        torch.full_like(next_token, eos_token_id),
                        next_token,
                    )
                    finished |= next_token.squeeze(-1).eq(eos_token_id)

                tokens = torch.cat((tokens, next_token), dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
        finally:
            self.train(was_training)

        return tokens


# Public names for callers that prefer a descriptive decoder model name.
DecoderOnlyTransformer = TransformerLM
DecoderModel = TransformerLM


__all__ = [
    "DecoderBlock",
    "DecoderConfig",
    "DecoderModel",
    "DecoderOnlyTransformer",
    "ModelConfig",
    "TransformerBlock",
    "TransformerConfig",
    "TransformerLM",
]
