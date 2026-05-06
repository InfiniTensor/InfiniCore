from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scal(x: Tensor, alpha: Tensor):
    _infinicore.scal_(x._underlying, alpha._underlying)
    return x
