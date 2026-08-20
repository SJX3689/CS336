"""Transformer feed-forward layers."""

import torch
from torch import nn
from torch.nn import functional as F


def silu(x: torch.Tensor) -> torch.Tensor:
    """Return the SiLU activation (kept as a public compatibility helper)."""

    return F.silu(x)


class FeedForward(nn.Module):
    """A SwiGLU feed-forward network used inside a Transformer block.

    The two input projections respectively produce the values and gates. All
    elementary projections use :class:`torch.nn.Linear` directly.
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        bias: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError(
                f"d_model and d_ff must be positive, got {d_model} and {d_ff}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.d_model = d_model
        self.d_ff = d_ff
        factory_kwargs = {"device": device, "dtype": dtype}
        self.up = nn.Linear(d_model, d_ff, bias=bias, **factory_kwargs)
        self.gate = nn.Linear(d_model, d_ff, bias=bias, **factory_kwargs)
        self.down = nn.Linear(d_ff, d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            last_dim = None if x.ndim == 0 else x.shape[-1]
            raise ValueError(
                f"expected input with last dimension {self.d_model}, got {last_dim}"
            )

        # Keep the original project's projection convention: ``up`` is the
        # activated branch and ``gate`` is the multiplicative linear branch.
        hidden = F.silu(self.up(x)) * self.gate(x)
        return self.down(self.dropout(hidden))


# Common names used by different Transformer implementations.
SwiGLU = FeedForward
FFN = FeedForward

__all__ = ["FFN", "FeedForward", "SwiGLU", "silu"]
