
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erf(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.erf(input._underlying))

    _infinicore.erf_(out._underlying, input._underlying)

    return out