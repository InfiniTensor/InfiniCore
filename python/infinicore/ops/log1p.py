import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log1p(input: Tensor, *, out=None) -> Tensor:
    r"""Compute the ln(x + 1)."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.log1p(input, out=out)

    if out is None:
        return Tensor(_infinicore.log1p(input._underlying))

    _infinicore.log1p_(out._underlying, input._underlying)
    return out
