from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotg(x: Tensor, y: Tensor, c: Tensor, s: Tensor):
    _infinicore.rotg_(x._underlying, y._underlying, c._underlying, s._underlying)
    return x, y, c, s
