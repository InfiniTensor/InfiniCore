from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_amax(x: Tensor, *, out=None):
    if out is None:
        return Tensor(_infinicore.blas_amax(x._underlying))

    _infinicore.blas_amax_(out._underlying, x._underlying)
    return out
