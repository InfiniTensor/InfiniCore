from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_or(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.logical_or(input._underlying, other._underlying))

    _infinicore.logical_or_(out._underlying, input._underlying, other._underlying)

    return out

