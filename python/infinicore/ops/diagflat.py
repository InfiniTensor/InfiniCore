from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def diagflat(input, *, offset=0, out=None):
    if out is None:
        return Tensor(_infinicore.diagflat(input._underlying, offset))
    _infinicore.diagflat_(out._underlying, input._underlying, offset)
    return out


