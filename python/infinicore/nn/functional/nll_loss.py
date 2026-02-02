import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def nll_loss(
    input: Tensor,
    target: Tensor,
    weight: Tensor | None = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    *,
    out=None,
) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.nll_loss(
            input, target, weight=weight, ignore_index=ignore_index, reduction=reduction
        )

    weight_underlying = weight._underlying if weight is not None else None

    if out is None:
        return Tensor(
            _infinicore.nll_loss(
                input._underlying, target._underlying, weight_underlying, ignore_index
            )
        )

    _infinicore.nll_loss_(
        input._underlying,
        target._underlying,
        weight_underlying,
        out._underlying,
        ignore_index,
    )
    return out
