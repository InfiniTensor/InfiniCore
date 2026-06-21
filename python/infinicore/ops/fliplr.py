import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def fliplr(input: Tensor, *, out=None) -> Tensor:
    r"""Flip tensor in the left/right direction.

    Equivalent to torch.fliplr(input), i.e. torch.flip(input, dims=(1,)).
    """

    assert input.ndim >= 2, "`fliplr` requires input with ndim >= 2."

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.fliplr(input)

    if out is None:
        return Tensor(_infinicore.fliplr(input._underlying))

    _infinicore.fliplr_(out._underlying, input._underlying)

    return out