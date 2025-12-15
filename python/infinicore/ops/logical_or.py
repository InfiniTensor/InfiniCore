import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_or(input, other, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.logical_or(input, other, out = out)

    if out is None:
        return Tensor(_infinicore.logical_or(input._underlying, other._underlying))

    _infinicore.logical_or_(out._underlying, input._underlying, other._underlying)

    return out

