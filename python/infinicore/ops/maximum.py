from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def maximum(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.maximum(input._underlying, other._underlying))

    _infinicore.maximum_(out._underlying, input._underlying, other._underlying)

    return out
