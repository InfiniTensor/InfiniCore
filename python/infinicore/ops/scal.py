from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def scal(x: Tensor, alpha: Tensor):
    _infinicore.scal_(alpha._underlying, x._underlying)

    return x
