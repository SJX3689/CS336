"""Linear-layer compatibility helpers.

The project used to reserve this module for a handwritten linear layer. A
handwritten implementation does not add useful behaviour here, so the public
``Linear`` name deliberately points at PyTorch's well-tested implementation.
"""

from torch import nn


# Keep the old import path without wrapping or reimplementing a fundamental
# operation.
Linear = nn.Linear

__all__ = ["Linear"]
