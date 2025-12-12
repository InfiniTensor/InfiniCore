from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erfinv(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.erfinv(input._underlying))

    _infinicore.erfinv_(out._underlying, input._underlying)

    return out