"""Root-mean-square layer normalization."""

import torch
from torch import nn


class RMSNorm(nn.Module):
    """Normalize the last dimension using its root mean square.

    RMSNorm intentionally has no bias and does not subtract the mean. For
    half-precision inputs, the reduction is carried out in float32 to avoid
    overflow and excessive rounding error.
    """

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def _rms(self, x: torch.Tensor) -> torch.Tensor:
        """Return the per-token RMS (kept for compatibility and inspection)."""

        return torch.sqrt(x.square().mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_floating_point():
            raise TypeError("RMSNorm expects a floating-point input tensor")
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            last_dim = None if x.ndim == 0 else x.shape[-1]
            raise ValueError(
                f"expected input with last dimension {self.d_model}, got {last_dim}"
            )

        input_dtype = x.dtype
        # float16/bfloat16 reductions are needlessly fragile for long vectors.
        compute_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        x_float = x.to(compute_dtype)
        output = x_float / self._rms(x_float) * self.weight.to(compute_dtype)
        return output.to(input_dtype)


__all__ = ["RMSNorm"]
