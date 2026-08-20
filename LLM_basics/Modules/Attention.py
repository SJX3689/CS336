"""Attention primitives and a decoder-style multi-head attention layer."""

import math
from typing import TypeAlias

import torch
from torch import nn
from torch.nn import functional as F

from .rope import RoPEEmbedding


KVCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]


def stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax stably, returning zeros for a fully masked row.

    PyTorch's regular softmax returns ``nan`` for a row made entirely of
    ``-inf``. In attention, such a row represents a query with no valid keys;
    a zero probability row is the useful result.
    """

    if not logits.is_floating_point():
        raise TypeError("stable_softmax expects floating-point logits")
    if logits.numel() == 0:
        return logits.clone()

    compute_dtype = (
        torch.float32
        if logits.dtype in (torch.float16, torch.bfloat16)
        else logits.dtype
    )
    values = logits.to(compute_dtype)
    maximum = values.amax(dim=dim, keepdim=True)
    positive_infinity = torch.isposinf(values)
    infinity_count = positive_infinity.sum(dim=dim, keepdim=True)
    shifted = torch.where(
        torch.isfinite(maximum),
        values - maximum,
        torch.full_like(values, -torch.inf),
    )
    numerator = shifted.exp()
    denominator = numerator.sum(dim=dim, keepdim=True)
    finite_probabilities = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(torch.finfo(compute_dtype).tiny),
        torch.zeros_like(numerator),
    )
    # If overflow produced one or more +inf maxima, put equal mass on those
    # maxima instead of returning NaN or discarding the row.
    infinite_probabilities = positive_infinity.to(
        compute_dtype
    ) / infinity_count.clamp_min(1)
    probabilities = torch.where(
        infinity_count > 0, infinite_probabilities, finite_probabilities
    )
    return probabilities.to(logits.dtype)


def _validate_attention_inputs(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
) -> None:
    tensors = {"query": query, "key": key, "value": value}
    for name, tensor in tensors.items():
        if tensor.ndim < 2:
            raise ValueError(f"{name} must have at least two dimensions")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating-point tensor")

    if query.shape[-1] != key.shape[-1]:
        raise ValueError(
            "query and key head dimensions must match, got "
            f"{query.shape[-1]} and {key.shape[-1]}"
        )
    if query.shape[-1] == 0:
        raise ValueError("the query/key head dimension must be non-zero")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError(
            "key and value sequence lengths must match, got "
            f"{key.shape[-2]} and {value.shape[-2]}"
        )
    if key.shape[-2] == 0:
        raise ValueError("attention requires at least one key/value position")
    if len({query.device, key.device, value.device}) != 1:
        raise ValueError("query, key, and value must be on the same device")
    if len({query.dtype, key.dtype, value.dtype}) != 1:
        raise ValueError("query, key, and value must have the same dtype")

    try:
        score_prefix = torch.broadcast_shapes(query.shape[:-2], key.shape[:-2])
        torch.broadcast_shapes(score_prefix, value.shape[:-2])
    except RuntimeError as error:
        raise ValueError(
            "query, key, and value batch/head dimensions are not broadcastable"
        ) from error


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> torch.Tensor:
    """Compute scaled dot-product attention.

    Shapes follow ``(..., query_length, head_dim)``. Mask meaning is determined
    only by dtype: boolean and integer masks are allow masks (``True``/non-zero
    permits attention), while every floating-point mask is additive (normally
    ``0`` leaves a score unchanged and ``-inf`` blocks it). In particular, an
    all-zero floating mask is equivalent to no mask. When query and key lengths
    differ, a causal mask is aligned to the right, which is the desired
    convention for an autoregressive KV cache.
    """

    _validate_attention_inputs(query, key, value)
    if not 0.0 <= dropout_p < 1.0:
        raise ValueError(f"dropout_p must be in [0, 1), got {dropout_p}")
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError(f"scale must be a finite positive number, got {scale}")

    scale_factor = scale if scale is not None else query.shape[-1] ** -0.5
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale_factor

    if is_causal:
        query_length, key_length = query.shape[-2], key.shape[-2]
        diagonal = key_length - query_length
        causal_mask = torch.ones(
            query_length,
            key_length,
            dtype=torch.bool,
            device=query.device,
        ).tril(diagonal=diagonal)
        scores = scores.masked_fill(~causal_mask, -torch.inf)

    if mask is not None:
        if mask.device != scores.device:
            raise ValueError("attention mask must be on the same device as the inputs")
        try:
            torch.broadcast_shapes(scores.shape, mask.shape)
        except RuntimeError as error:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is not broadcastable to "
                f"attention scores {tuple(scores.shape)}"
            ) from error

        if mask.dtype == torch.bool or (
            not mask.is_floating_point() and not mask.is_complex()
        ):
            scores = scores.masked_fill(~mask.to(torch.bool), -torch.inf)
        elif mask.is_floating_point():
            scores = scores + mask.to(scores.dtype)
        else:
            raise TypeError(
                "attention mask must have a boolean, integer, or floating dtype"
            )

    attention_weights = stable_softmax(scores, dim=-1)
    if dropout_p:
        # This functional primitive has no train/eval state. Callers decide
        # whether to pass a non-zero probability (MultiHeadAttention does so).
        attention_weights = F.dropout(attention_weights, p=dropout_p, training=True)
    return torch.matmul(attention_weights, value)


# Keep the old private name as an identity alias for callers that reached into
# this module. The implementation itself lives in ``rope.py`` so the public
# class and MultiHeadAttention cannot drift apart.
_RotaryEmbedding = RoPEEmbedding


class MultiHeadAttention(nn.Module):
    """Decoder self-attention with optional RoPE and autoregressive KV cache.

    ``forward`` returns a tensor in the usual path. With ``use_cache=True`` it
    returns ``(output, (key_cache, value_cache))``; cached tensors have shape
    ``(batch, heads, cached_tokens, head_dim)`` and grow as tokens are appended.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_rope: bool = False,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        dropout: float = 0.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError(
                f"d_model and num_heads must be positive, got {d_model} and {num_heads}"
            )
        if d_model % num_heads:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        # d_k is retained for callers that used the original implementation.
        self.d_k = self.head_dim
        self.dropout = dropout
        self.use_rope = use_rope

        factory_kwargs = {"device": device, "dtype": dtype}
        # Keep the draft's registered module names so existing state dicts and
        # callers remain compatible; modern ``*_proj`` aliases are below.
        self.q_linear = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.v_linear = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)
        self.out_linear = nn.Linear(d_model, d_model, bias=bias, **factory_kwargs)

        self.rope: RoPEEmbedding | None = None
        if use_rope:
            self.rope = RoPEEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )

    # Conventional aliases used by many Transformer implementations.
    @property
    def q_proj(self) -> nn.Linear:
        return self.q_linear

    @property
    def k_proj(self) -> nn.Linear:
        return self.k_linear

    @property
    def v_proj(self) -> nn.Linear:
        return self.v_linear

    @property
    def out_proj(self) -> nn.Linear:
        return self.out_linear

    def _create_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
        past_len: int = 0,
    ) -> torch.Tensor:
        """Create an allow-mask shaped ``(1, 1, query, key)``."""

        if seq_len < 0 or past_len < 0:
            raise ValueError("seq_len and past_len cannot be negative")
        query_positions = torch.arange(past_len, past_len + seq_len, device=device)
        key_positions = torch.arange(past_len + seq_len, device=device)
        return (key_positions[None, :] <= query_positions[:, None])[None, None]

    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _validate_cache(
        self,
        past_key_value: KVCache | None,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, int]:
        if past_key_value is None:
            return None, None, 0
        if not isinstance(past_key_value, (tuple, list)) or len(past_key_value) != 2:
            raise ValueError("past_key_value must be a (key, value) pair")
        past_key, past_value = past_key_value
        expected_prefix = (x.shape[0], self.num_heads)
        for name, cached in (("key", past_key), ("value", past_value)):
            if cached.ndim != 4:
                raise ValueError(f"cached {name} must have four dimensions")
            if cached.shape[:2] != expected_prefix or cached.shape[-1] != self.head_dim:
                raise ValueError(
                    f"cached {name} must have shape (batch={x.shape[0]}, "
                    f"heads={self.num_heads}, tokens, head_dim={self.head_dim})"
                )
            if cached.device != x.device or cached.dtype != x.dtype:
                raise ValueError(f"cached {name} must match the input device and dtype")
        if past_key.shape[-2] != past_value.shape[-2]:
            raise ValueError("cached key and value sequence lengths must match")
        return past_key, past_value, past_key.shape[-2]

    @staticmethod
    def _normalize_mask(
        mask: torch.Tensor,
        batch_size: int,
        query_length: int,
        key_length: int,
    ) -> torch.Tensor:
        """Expand common padding/attention mask layouts for score broadcasting."""

        if mask.ndim == 1 and mask.shape[0] == key_length:
            mask = mask.view(1, 1, 1, key_length)
        elif mask.ndim == 2:
            # Prefer the common key-padding layout if batch and query lengths
            # happen to be equal; use 3-D/4-D masks to remove that ambiguity.
            if mask.shape == (batch_size, key_length):
                mask = mask.view(batch_size, 1, 1, key_length)
            elif mask.shape == (query_length, key_length):
                mask = mask.view(1, 1, query_length, key_length)
        elif mask.ndim == 3 and mask.shape in (
            (batch_size, query_length, key_length),
            (batch_size, 1, key_length),
        ):
            mask = mask.unsqueeze(1)

        target_shape = (batch_size, 1, query_length, key_length)
        try:
            torch.broadcast_shapes(target_shape, mask.shape)
        except RuntimeError as error:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} is incompatible with batch={batch_size}, "
                f"query_length={query_length}, key_length={key_length}"
            ) from error
        return mask

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        *,
        causal: bool = True,
        past_key_value: KVCache | None = None,
        use_cache: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(
                f"x must have shape (batch, sequence, {self.d_model}), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("multi-head attention expects floating-point inputs")

        batch_size, seq_len, _ = x.shape
        if seq_len == 0:
            raise ValueError("multi-head attention requires at least one input token")
        past_key, past_value, past_len = self._validate_cache(past_key_value, x)

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    past_len, past_len + seq_len, device=x.device, dtype=torch.long
                )
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        if past_key is not None:
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if mask is not None:
            mask = self._normalize_mask(mask, batch_size, seq_len, key.shape[-2])

        attention_output = scaled_dot_product_attention(
            query,
            key,
            value,
            mask=mask,
            is_causal=causal,
            dropout_p=self.dropout if self.training else 0.0,
        )
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(attention_output)
        if use_cache:
            return output, (key, value)
        return output


__all__ = [
    "KVCache",
    "MultiHeadAttention",
    "RoPEEmbedding",
    "scaled_dot_product_attention",
    "stable_softmax",
]
