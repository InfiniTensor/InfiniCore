from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def asum(x: Tensor, *, out=None):
    if out is None:
        return Tensor(_infinicore.asum(x._underlying))

    _infinicore.asum_(x._underlying, out._underlying)

    return out
