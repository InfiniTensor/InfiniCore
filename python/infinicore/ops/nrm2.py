from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def nrm2(x: Tensor, *, out=None):
    if out is None:
        return Tensor(_infinicore.nrm2(x._underlying))

    _infinicore.nrm2_(x._underlying, out._underlying)

    return out
