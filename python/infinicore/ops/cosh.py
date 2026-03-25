import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cosh(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.cosh(input)

    if out is None:
        return Tensor(_infinicore.cosh(input._underlying))

    _infinicore.cosh_(out._underlying, input._underlying)

    return out
