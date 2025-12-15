from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logical_xor(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.logical_xor(input._underlying, other._underlying))

    _infinicore.logical_xor_(out._underlying, input._underlying, other._underlying)

    return out

