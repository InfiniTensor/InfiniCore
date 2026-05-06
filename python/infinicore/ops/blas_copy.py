from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def blas_copy(x: Tensor, y: Tensor):
    _infinicore.blas_copy_(x._underlying, y._underlying)
    return y
