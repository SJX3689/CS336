"""Rotary positional embeddings used by decoder attention."""

import torch
from torch import nn


class RoPEEmbedding(nn.Module):
    """Apply rotary position embeddings to adjacent feature pairs.

    Args:
        theta: Base used to construct the rotation frequencies.
        d_k: Size of each attention head. It must be positive and even.
        max_seq_len: Initial length of the trigonometric cache. The cache grows
            automatically when a larger token position is encountered.
        device: Optional device on which to create the non-persistent cache.

    The constructor keeps the ``(theta, d_k, max_seq_len, device)`` interface
    from the original project draft. Inputs may have shape
    ``(..., sequence_length, d_k)``. ``token_positions`` may be shared across
    the leading dimensions with shape ``(sequence_length,)`` or supplied per
    batch item with shape ``(batch, sequence_length)``.
    """

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k <= 0 or d_k % 2:
            raise ValueError(f"RoPE requires a positive, even d_k, got {d_k}")
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        # ``dim`` is retained as a descriptive alias for internal and external
        # callers that used the previous private implementation.
        self.dim = d_k
        self.register_buffer(
            "inv_freq", self._make_frequencies(device), persistent=False
        )
        self.register_buffer(
            "_cos_cache",
            torch.empty(0, d_k // 2, device=device, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "_sin_cache",
            torch.empty(0, d_k // 2, device=device, dtype=torch.float32),
            persistent=False,
        )
        self._ensure_cache(max_seq_len)

    @property
    def cached_sequence_length(self) -> int:
        """Number of positions currently stored in the trig cache."""

        return self._cos_cache.shape[0]

    def _make_frequencies(
        self, device: torch.device | str | None
    ) -> torch.Tensor:
        return 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
                / self.d_k
            )
        )

    def _ensure_cache(self, required_length: int) -> None:
        if (
            required_length <= self.cached_sequence_length
            and self._cos_cache.dtype == torch.float32
            and self._sin_cache.dtype == torch.float32
            and self.inv_freq.dtype == torch.float32
        ):
            return

        # ``module.half()`` converts floating buffers too. Rebuild these
        # numerically sensitive, non-persistent buffers in float32 on demand.
        self.inv_freq = self._make_frequencies(self.inv_freq.device)
        positions = torch.arange(
            required_length,
            device=self.inv_freq.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, self.inv_freq)
        self._cos_cache = angles.cos()
        self._sin_cache = angles.sin()

    def forward(
        self, x: torch.Tensor, token_positions: torch.Tensor
    ) -> torch.Tensor:
        """Rotate ``x`` according to integer ``token_positions``."""

        if x.ndim < 2 or x.shape[-1] != self.d_k:
            raise ValueError(
                "RoPE expected shape (..., sequence_length, "
                f"{self.d_k}), got {tuple(x.shape)}"
            )
        if not x.is_floating_point():
            raise TypeError("RoPE expects floating-point inputs")
        if (
            token_positions.dtype == torch.bool
            or token_positions.is_floating_point()
            or token_positions.is_complex()
        ):
            raise TypeError("token_positions must contain integer indices")
        if token_positions.device != x.device:
            raise ValueError("token_positions and attention inputs must share a device")
        if self.inv_freq.device != x.device:
            raise ValueError("RoPE and attention inputs must share a device")
        if (
            token_positions.ndim not in (1, 2)
            or token_positions.shape[-1] != x.shape[-2]
        ):
            raise ValueError(
                "token_positions must have shape (sequence,) or (batch, sequence)"
            )
        if token_positions.ndim == 2:
            if x.ndim < 3:
                raise ValueError(
                    "batched token_positions require an input batch dimension"
                )
            if token_positions.shape[0] not in (1, x.shape[0]):
                raise ValueError(
                    f"token_positions batch dimension must be 1 or {x.shape[0]}"
                )
        if token_positions.numel() and torch.any(token_positions < 0).item():
            raise ValueError("token_positions cannot contain negative indices")

        positions = token_positions.to(torch.long)
        required_length = int(positions.max().item()) + 1 if positions.numel() else 0
        self._ensure_cache(required_length)
        cosine = self._cos_cache[positions]
        sine = self._sin_cache[positions]

        # A per-batch position matrix needs singleton head/other-prefix axes.
        # Shared 1-D positions already broadcast over every leading dimension.
        if positions.ndim == 2:
            for _ in range(x.ndim - 3):
                cosine = cosine.unsqueeze(1)
                sine = sine.unsqueeze(1)

        compute_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )
        pairs = x.to(compute_dtype).unflatten(-1, (self.d_k // 2, 2))
        even, odd = pairs[..., 0], pairs[..., 1]
        cosine = cosine.to(compute_dtype)
        sine = sine.to(compute_dtype)
        rotated = torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)
        return rotated.to(x.dtype)


__all__ = ["RoPEEmbedding"]
