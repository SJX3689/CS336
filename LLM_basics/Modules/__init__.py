"""Public Transformer building blocks."""

import importlib as _importlib
import sys as _sys

from .Attention import (
    KVCache,
    MultiHeadAttention,
    scaled_dot_product_attention,
    stable_softmax,
)
from .MoE import MixtureOfExperts, MoE
from .RMSNorm import RMSNorm
from .ffn import FFN, FeedForward, SwiGLU, silu
from .rope import RoPEEmbedding


# macOS commonly uses a case-insensitive filesystem, so ``Linear.py`` and
# ``linear.py`` cannot coexist there. Register the lowercase draft path as an
# alias of the canonical module instead of keeping two files that could drift.
_linear_module = _importlib.import_module(f"{__name__}.Linear")
_sys.modules.setdefault(f"{__name__}.linear", _linear_module)
# Also expose the conventional package attribute used by
# ``import LLM_basics.Modules.linear``.
linear = _linear_module
Linear = _linear_module.Linear

__all__ = [
    "FFN",
    "FeedForward",
    "KVCache",
    "Linear",
    "MixtureOfExperts",
    "MoE",
    "MultiHeadAttention",
    "RMSNorm",
    "RoPEEmbedding",
    "SwiGLU",
    "scaled_dot_product_attention",
    "silu",
    "stable_softmax",
]
