from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erfc(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.erfc(input._underlying))

    _infinicore.erfc_(out._underlying, input._underlying)

    return out