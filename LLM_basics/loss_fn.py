"""Numerically stable losses and metrics for language-model training."""

from __future__ import annotations

from typing import Literal

import torch
from torch.nn import functional as F

Reduction = Literal["none", "mean", "sum"]


def cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute cross entropy with the vocabulary on the last axis.

    PyTorch's native function expects classes on axis 1.  Flattening the token
    axes here makes the convenient LLM layout ``(..., vocab_size)`` work for
    both batched sequences and simple two-dimensional classification logits.
    The implementation delegates log-sum-exp stabilization and mixed-precision
    details to :func:`torch.nn.functional.cross_entropy`.
    """

    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if tuple(labels.shape) != tuple(logits.shape[:-1]):
        raise ValueError(
            "labels must match every non-vocabulary logits dimension: "
            f"got logits {tuple(logits.shape)} and labels {tuple(labels.shape)}"
        )
    if labels.dtype != torch.long:
        raise TypeError("labels must have dtype torch.long")
    if reduction not in ("none", "mean", "sum"):
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")
    if not 0.0 <= label_smoothing <= 1.0:
        raise ValueError("label_smoothing must be in [0, 1]")

    vocabulary_size = logits.size(-1)
    flat_loss = F.cross_entropy(
        logits.reshape(-1, vocabulary_size),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )
    if reduction == "none":
        return flat_loss.reshape(labels.shape)
    return flat_loss


def causal_lm_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    *,
    ignore_index: int = -100,
    reduction: Reduction = "mean",
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Compute next-token loss when logits and unshifted tokens have equal length.

    Position ``t`` predicts the token at ``t + 1``.  Callers that already have
    separately shifted inputs and targets should call :func:`cross_entropy`
    directly.
    """

    if logits.ndim < 3:
        raise ValueError("causal LM logits must have shape (..., sequence, vocabulary)")
    if tuple(token_ids.shape) != tuple(logits.shape[:-1]):
        raise ValueError("token_ids must match the batch and sequence axes of logits")
    if logits.size(-2) < 2:
        raise ValueError("at least two sequence positions are required for a causal loss")
    return cross_entropy(
        logits[..., :-1, :],
        token_ids[..., 1:],
        ignore_index=ignore_index,
        reduction=reduction,
        label_smoothing=label_smoothing,
    )


def perplexity(loss: torch.Tensor | float) -> torch.Tensor:
    """Convert average token cross entropy (natural-log units) to perplexity."""

    return torch.exp(torch.as_tensor(loss))


__all__ = ["causal_lm_loss", "cross_entropy", "perplexity"]
