from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def rotm(x: Tensor, y: Tensor, param: Tensor):
    _infinicore.rotm_(x._underlying, y._underlying, param._underlying)
    return x, y
