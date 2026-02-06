import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def binary_cross_entropy(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    *,
    out=None,
) -> Tensor:
    r"""Apply the binary_cross_entropy function."""

    if size_average is not None or reduce is not None:
        if reduce is False:
            reduction = "none"
        elif size_average is True or size_average is None:
            reduction = "mean"
        else:
            reduction = "sum"

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.binary_cross_entropy(
            input, target, weight=weight, reduction=reduction, out=out
        )

    weight_underlying = weight._underlying if weight is not None else None

    if out is None:
        return Tensor(
            _infinicore.binary_cross_entropy(
                input._underlying, target._underlying, weight_underlying, reduction
            )
        )

    _infinicore.binary_cross_entropy_(
        input._underlying,
        target._underlying,
        weight_underlying,
        out._underlying,
        reduction,
    )
    return out
