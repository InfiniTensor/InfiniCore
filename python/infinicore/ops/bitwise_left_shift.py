import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def bitwise_left_shift(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes the left arithmetic shift of input by other bits. The input tensor must be of integral type."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.bitwise_left_shift(input, other, out=out)

    if out is None:
        return Tensor(
            _infinicore.bitwise_left_shift(input._underlying, other._underlying)
        )

    _infinicore.bitwise_left_shift_(
        out._underlying, input._underlying, other._underlying
    )

    return out
