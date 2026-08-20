"""Optimization, learning-rate, clipping, and checkpoint utilities."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from os import PathLike
from typing import Any, BinaryIO, Optional

import torch


def cosine_annealing_lr(
    t: int,
    alpha_max: float,
    alpha_min: float,
    Tw: int,
    Tc: int,
) -> float:
    """
    Cosine annealing learning-rate schedule with linear warmup.

    Args:
        t:
            Current training step.
        alpha_max:
            Maximum learning rate after warmup.
        alpha_min:
            Minimum learning rate after cosine decay.
        Tw:
            Number of warmup steps.
        Tc:
            Step at which cosine decay finishes.

    Returns:
        Learning rate at step t.
    """

    if not isinstance(t, int) or isinstance(t, bool) or t < 0:
        raise ValueError("t must be a non-negative integer")
    if not isinstance(Tw, int) or isinstance(Tw, bool) or Tw < 0:
        raise ValueError("Tw must be a non-negative integer")
    if not isinstance(Tc, int) or isinstance(Tc, bool) or Tc <= Tw:
        raise ValueError("Tc must be an integer greater than Tw")
    if alpha_min < 0 or alpha_max < 0:
        raise ValueError("learning rates must be non-negative")
    if alpha_min > alpha_max:
        raise ValueError("alpha_min cannot be larger than alpha_max")

    # Linear warmup:
    #
    # t = 0   -> lr = 0
    # t = Tw  -> lr = alpha_max
    if t < Tw:
        return alpha_max * t / Tw

    # Cosine decay:
    #
    # t = Tw  -> lr = alpha_max
    # t = Tc  -> lr = alpha_min
    if t <= Tc:
        progress = (t - Tw) / (Tc - Tw)

        return alpha_min + 0.5 * (
            1.0 + math.cos(math.pi * progress)
        ) * (alpha_max - alpha_min)

    # After cosine decay, keep learning rate fixed at alpha_min.
    return alpha_min


class AdamW(torch.optim.Optimizer):
    """A compact, fully functional implementation of decoupled AdamW.

    It intentionally follows :class:`torch.optim.AdamW`'s default update rule
    and state layout closely, making it useful for learning and for checkpoint
    round trips.  Sparse gradients are not supported by AdamW.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0:
            raise ValueError(f"lr must be non-negative, got {lr}")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"eps must be finite and positive, got {eps}")
        if weight_decay < 0:
            raise ValueError(
                f"weight_decay must be non-negative, got {weight_decay}"
            )
        if len(betas) != 2 or not all(0.0 <= beta < 1.0 for beta in betas):
            raise ValueError("betas must contain two values in [0, 1)")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

        # This compact implementation stores a real-valued second moment and
        # uses ``grad * grad``.  Silently applying that rule to complex tensors
        # would not implement complex AdamW's required magnitude semantics.
        if any(
            parameter.is_complex()
            for group in self.param_groups
            for parameter in group["params"]
        ):
            raise TypeError("AdamW does not support complex parameters")

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable[[], torch.Tensor]] = None,
    ) -> torch.Tensor | None:
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                # Also guard parameter groups appended after construction.
                if p.is_complex():
                    raise TypeError("AdamW does not support complex parameters")
                if p.grad is None:
                    continue

                grad = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                # Each parameter has its own optimizer state.
                state = self.state[p]

                # Initialize Adam state on the first update.
                if len(state) == 0:
                    state["step"] = 0

                    # First moment:
                    # exponential moving average of gradients.
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                    # Second moment:
                    # exponential moving average of squared gradients.
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                state["step"] += 1
                t = state["step"]

                # -----------------------------------------------------
                # 1. Update first moment
                #
                # m_t = beta1 * m_{t-1}
                #       + (1 - beta1) * g_t
                # -----------------------------------------------------
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1.0 - beta1,
                )

                # -----------------------------------------------------
                # 2. Update second moment
                #
                # v_t = beta2 * v_{t-1}
                #       + (1 - beta2) * g_t^2
                # -----------------------------------------------------
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )

                # -----------------------------------------------------
                # 3. Bias correction
                #
                # m_hat = m_t / (1 - beta1^t)
                # v_hat = v_t / (1 - beta2^t)
                # -----------------------------------------------------
                bias_correction1 = 1.0 - beta1**t
                bias_correction2 = 1.0 - beta2**t

                # We fold m's bias correction into the step size.
                step_size = lr / bias_correction1

                # sqrt(v_hat) + eps
                denom = (
                    exp_avg_sq / bias_correction2
                ).sqrt().add_(eps)

                # -----------------------------------------------------
                # 4. Decoupled weight decay
                #
                # theta <- (1 - lr * weight_decay) * theta
                # -----------------------------------------------------
                p.mul_(1.0 - lr * weight_decay)

                # -----------------------------------------------------
                # 5. Adam parameter update
                #
                # theta <- theta
                #          - lr * m_hat / (sqrt(v_hat) + eps)
                # -----------------------------------------------------
                p.addcdiv_(
                    exp_avg,
                    denom,
                    value=-step_size,
                )

        return loss


@torch.no_grad()
def gradient_clip(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
    *,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """
    Clip the global L2 norm of all gradients and return its pre-clip value.

    If the total gradient norm is larger than max_l2_norm,
    all gradients are scaled by the same factor.
    """

    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")
    if eps < 0:
        raise ValueError("eps must be non-negative")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)

    # Accumulate norms in float32 to avoid overflowing fp16 training gradients.
    norm_device = gradients[0].device
    individual_norms = []
    for gradient in gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        individual_norms.append(values.detach().float().norm(2).to(norm_device))
    total_norm = torch.stack(individual_norms).norm(2)

    if error_if_nonfinite and not bool(torch.isfinite(total_norm)):
        raise RuntimeError(f"the total gradient norm is non-finite: {total_norm.item()}")

    # A single coefficient preserves the direction of the complete gradient.
    clip_coefficient = (max_l2_norm / (total_norm + eps)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm


# Compatibility with the naming used by the CS336 assignment adapters.
gradient_clipping = gradient_clip


class WarmupCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    """Linearly warm up, cosine-decay, then hold each parameter group's LR.

    The optimizer's learning rates at construction are treated as the maximum
    rates.  Like PyTorch schedulers, this scheduler initializes step 0 when it is
    constructed, so the first optimizer update uses the step-0 warmup rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float | Sequence[float] = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps")

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lrs = [float(group["lr"]) for group in optimizer.param_groups]
        if isinstance(min_lr, Sequence) and not isinstance(min_lr, (str, bytes)):
            self.min_lrs = [float(value) for value in min_lr]
            if len(self.min_lrs) != len(self.max_lrs):
                raise ValueError("min_lr must have one value per parameter group")
        else:
            self.min_lrs = [float(min_lr)] * len(self.max_lrs)
        if any(value < 0 for value in self.min_lrs):
            raise ValueError("min_lr values must be non-negative")
        if any(minimum > maximum for minimum, maximum in zip(self.min_lrs, self.max_lrs)):
            raise ValueError("a min_lr cannot exceed its parameter group's learning rate")
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self) -> list[float]:
        """Calculate the rates for ``last_epoch``, interpreted as a step."""

        return [
            cosine_annealing_lr(
                self.last_epoch,
                alpha_max=maximum,
                alpha_min=minimum,
                Tw=self.warmup_steps,
                Tc=self.total_steps,
            )
            for maximum, minimum in zip(self.max_lrs, self.min_lrs)
        ]


# A longer descriptive alias familiar to users of torch.optim schedulers.
WarmupCosineAnnealingLR = WarmupCosineScheduler


def set_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    """Set every optimizer parameter group to the same learning rate."""

    if learning_rate < 0:
        raise ValueError("learning_rate must be non-negative")
    for group in optimizer.param_groups:
        group["lr"] = learning_rate


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    iteration: int,
    out: str | PathLike[str] | BinaryIO,
    *,
    scheduler: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save model/training state without making assumptions about data loading."""

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "iteration": int(iteration),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if extra is not None:
        checkpoint["extra"] = extra
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | PathLike[str] | BinaryIO,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    *,
    scheduler: Any | None = None,
    map_location: str | torch.device | Callable | None = "cpu",
    strict: bool = True,
) -> int:
    """Restore a checkpoint and return the saved training iteration."""

    checkpoint = torch.load(src, map_location=map_location, weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("checkpoint does not contain model_state_dict")
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    if optimizer is not None:
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if optimizer_state is None:
            raise ValueError("checkpoint does not contain optimizer_state_dict")
        optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler_state_dict")
        if scheduler_state is None:
            raise ValueError("checkpoint does not contain scheduler_state_dict")
        scheduler.load_state_dict(scheduler_state)
    return int(checkpoint.get("iteration", 0))


__all__ = [
    "AdamW",
    "WarmupCosineAnnealingLR",
    "WarmupCosineScheduler",
    "cosine_annealing_lr",
    "gradient_clip",
    "gradient_clipping",
    "load_checkpoint",
    "save_checkpoint",
    "set_learning_rate",
]
