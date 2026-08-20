"""Reusable training entry points for the decoder-only language model.

There is intentionally no dataset construction or hard-coded training run in
this file.  Supply an existing tensor iterable/DataLoader from an experiment and
call :func:`train`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import nn

from LLM_basics.loss_fn import cross_entropy
from LLM_basics.optimizer import gradient_clip

TensorPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
Batch: TypeAlias = torch.Tensor | Sequence[torch.Tensor] | Mapping[str, torch.Tensor]


@dataclass(slots=True)
class TrainingState:
    """Small serializable summary returned by :func:`train`."""

    step: int = 0
    epoch: int = 0
    last_loss: float | None = None


def prepare_batch(batch: Batch) -> TensorPair:
    """Normalize common tensor/DataLoader batch formats to ``(inputs, labels)``.

    Accepted forms are:

    * one ``(batch, sequence)`` token tensor, shifted here for next-token loss;
    * ``(input_ids, labels)`` from a TensorDataset/DataLoader;
    * a mapping with ``input_ids`` and optional ``labels``/``targets``.

    A mapping without labels is shifted in the same way as a lone tensor.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor | None
    if isinstance(batch, torch.Tensor):
        input_ids, labels = batch, None
    elif isinstance(batch, Mapping):
        if "input_ids" not in batch:
            raise KeyError("a mapping batch must contain 'input_ids'")
        input_ids = batch["input_ids"]
        labels = batch.get("labels")
        if labels is None:
            labels = batch.get("targets")
    elif isinstance(batch, Sequence) and not isinstance(batch, (str, bytes)):
        if len(batch) != 2:
            raise ValueError("a sequence batch must contain exactly (input_ids, labels)")
        input_ids, labels = batch
    else:
        raise TypeError("unsupported batch type")

    if not isinstance(input_ids, torch.Tensor):
        raise TypeError("input_ids must be a torch.Tensor")
    if input_ids.ndim != 2:
        raise ValueError("input_ids must have shape (batch, sequence)")

    if labels is None:
        if input_ids.size(1) < 2:
            raise ValueError("an unshifted token batch needs at least two positions")
        labels = input_ids[:, 1:]
        input_ids = input_ids[:, :-1]
    elif not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor")

    if labels.shape != input_ids.shape:
        raise ValueError(
            "input_ids and labels must have the same shape after shifting, "
            f"got {tuple(input_ids.shape)} and {tuple(labels.shape)}"
        )
    return input_ids, labels


def _model_device(model: nn.Module) -> torch.device:
    """Return the first parameter/buffer device, falling back to CPU."""

    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _devices_match(requested: torch.device, actual: torch.device) -> bool:
    """Compare devices while treating an omitted accelerator index as default."""

    return requested.type == actual.type and (
        requested.index is None
        or actual.index is None
        or requested.index == actual.index
    )


def train_step(
    model: nn.Module,
    batch: Batch,
    optimizer: torch.optim.Optimizer,
    *,
    targets: torch.Tensor | None = None,
    device: torch.device | str | None = None,
    max_grad_norm: float | None = 1.0,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    scheduler: object | None = None,
) -> torch.Tensor:
    """Run one optimizer update and return a detached scalar loss tensor.

    ``targets=...`` is a convenience for callers that hold inputs and targets
    separately; DataLoader users can pass the pair as ``batch`` instead.
    """

    if targets is not None:
        if not isinstance(batch, torch.Tensor):
            raise TypeError("targets can only accompany a tensor input batch")
        batch = (batch, targets)
    input_ids, labels = prepare_batch(batch)
    model_device = _model_device(model)
    destination = torch.device(device) if device is not None else model_device
    if device is not None and not _devices_match(destination, model_device):
        raise ValueError(
            f"requested device {destination} does not match model device "
            f"{model_device}; move the model before constructing the optimizer"
        )
    input_ids = input_ids.to(device=destination, dtype=torch.long)
    labels = labels.to(device=destination, dtype=torch.long)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_ids)
    loss = cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )
    if not bool(torch.isfinite(loss.detach())):
        raise FloatingPointError(f"non-finite training loss: {loss.detach().item()}")
    loss.backward()

    if max_grad_norm is not None:
        gradient_clip(model.parameters(), max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    if scheduler is not None:
        step_method = getattr(scheduler, "step", None)
        if not callable(step_method):
            raise TypeError("scheduler must provide a callable step() method")
        step_method()
    return loss.detach()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    batches: Iterable[Batch],
    *,
    device: torch.device | str | None = None,
    max_batches: int | None = None,
    ignore_index: int = -100,
) -> float:
    """Return token-weighted mean loss while preserving the model's prior mode."""

    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive when provided")
    destination = torch.device(device) if device is not None else _model_device(model)
    was_training = model.training
    model.eval()
    total_loss = 0.0
    token_count = 0
    try:
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break
            input_ids, labels = prepare_batch(batch)
            input_ids = input_ids.to(device=destination, dtype=torch.long)
            labels = labels.to(device=destination, dtype=torch.long)
            logits = model(input_ids)
            batch_loss = cross_entropy(
                logits,
                labels,
                ignore_index=ignore_index,
                reduction="sum",
            )
            valid_tokens = labels.ne(ignore_index).sum().item()
            total_loss += float(batch_loss.item())
            token_count += int(valid_tokens)
    finally:
        model.train(was_training)

    if token_count == 0:
        raise ValueError("evaluation received no non-ignored target tokens")
    return total_loss / token_count


def train(
    model: nn.Module,
    train_batches: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    *,
    epochs: int = 1,
    max_steps: int | None = None,
    device: torch.device | str | None = None,
    max_grad_norm: float | None = 1.0,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    scheduler: object | None = None,
    on_step: Callable[[TrainingState], None] | None = None,
) -> TrainingState:
    """Train on caller-supplied batches and return progress state.

    For ``epochs > 1``, ``train_batches`` should be re-iterable (as DataLoaders
    are).  ``on_step`` can implement logging or call ``save_checkpoint`` without
    coupling those policies to this basic loop.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if max_steps is not None and max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    state = TrainingState()
    if max_steps == 0:
        return state

    for epoch_index in range(epochs):
        saw_batch = False
        for batch in train_batches:
            saw_batch = True
            # Do not report an epoch until it actually yielded a batch.  This
            # matters for empty inputs and exhausted one-shot iterators.
            state.epoch = epoch_index + 1
            loss = train_step(
                model,
                batch,
                optimizer,
                device=device,
                max_grad_norm=max_grad_norm,
                ignore_index=ignore_index,
                label_smoothing=label_smoothing,
                scheduler=scheduler,
            )
            state.step += 1
            state.last_loss = float(loss.item())
            if on_step is not None:
                on_step(state)
            if max_steps is not None and state.step >= max_steps:
                return state

        if not saw_batch:
            # This also gives a clear result for a one-shot iterator reused
            # across epochs, instead of silently reporting phantom progress.
            break
    return state


__all__ = [
    "Batch",
    "TrainingState",
    "evaluate",
    "prepare_batch",
    "train",
    "train_step",
]
