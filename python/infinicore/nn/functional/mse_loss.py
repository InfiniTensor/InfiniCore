import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mse_loss(input: Tensor, target: Tensor, reduction: str = "mean") -> Tensor:
    r"""Compute mean squared error loss between input and target."""

    assert reduction in (
        "none",
        "mean",
        "sum",
    ), "`reduction` must be one of 'none', 'mean', or 'sum'."

    assert input.shape == target.shape, "`input` and `target` must have the same shape."

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.mse_loss(
            input,
            target,
            reduction=reduction,
        )

    return Tensor(
        _infinicore.mse_loss(
            input._underlying,
            target._underlying,
            reduction,
        )
    )