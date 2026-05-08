from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_amin(x: Tensor, *, out=None):
    if out is None:
        return Tensor(_infinicore.blas_amin(x._underlying))

    _infinicore.blas_amin_(x._underlying, out._underlying)

    return out
