from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def sqrt(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.sqrt(input._underlying))
    _infinicore.sqrt_(out._underlying, input._underlying)
    return out