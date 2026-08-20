"""Mixture-of-experts extension point.

Routing, expert-capacity handling, and the auxiliary load-balancing loss need
to be designed together. Returning an identity tensor in the meantime would
silently train a different model, so this placeholder fails explicitly when
it is used in a forward pass.
"""

from typing import Any

from torch import nn


class MoE(nn.Module):
    """Reserved interface for a future mixture-of-experts layer.

    Arbitrary constructor arguments are retained as configuration metadata so
    a higher-level model can already expose an MoE configuration window. The
    computational path remains intentionally unavailable.
    """

    is_implemented = False

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.args = args
        self.config = dict(kwargs)

    def forward(self, *args: Any, **kwargs: Any):
        raise NotImplementedError(
            "MoE routing is an extension point and is not implemented yet. "
            "Use FeedForward until routing, expert capacity, and the "
            "load-balancing loss are implemented."
        )


MixtureOfExperts = MoE

__all__ = ["MixtureOfExperts", "MoE"]
