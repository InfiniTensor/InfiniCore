from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_dot(x: Tensor, y: Tensor, *, out=None):
    if out is None:
        return Tensor(_infinicore.blas_dot(x._underlying, y._underlying))

    _infinicore.blas_dot_(x._underlying, y._underlying, out._underlying)

    return out
