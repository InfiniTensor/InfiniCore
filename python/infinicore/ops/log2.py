import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log2(input: Tensor, *, out=None) -> Tensor:
    r"""Computes the base-2 logarithm of the input tensor element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.log2(input, out=out)

    if out is None:
        return Tensor(
            _infinicore.log2(input._underlying)
        )

    _infinicore.log2_(
        out._underlying, input._underlying
    )

    return out
