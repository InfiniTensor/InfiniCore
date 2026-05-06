from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def swap(x: Tensor, y: Tensor):
    _infinicore.swap_(x._underlying, y._underlying)
    return x, y
