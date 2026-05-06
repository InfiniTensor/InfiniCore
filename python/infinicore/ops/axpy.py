from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def axpy(alpha: Tensor, x: Tensor, y: Tensor):
    _infinicore.axpy_(alpha._underlying, x._underlying, y._underlying)
    return y
