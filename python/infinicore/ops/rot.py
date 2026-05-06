from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rot(x: Tensor, y: Tensor, c: Tensor, s: Tensor):
    _infinicore.rot_(x._underlying, y._underlying, c._underlying, s._underlying)
    return x, y
