from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def cat(tensors, dim:int=0, out=None):
    if out is None:
        return Tensor(_infinicore.cat([tensor._underlying for tensor in tensors], dim))

    _infinicore.cat_([tensor._underlying for tensor in tensors], dim, out._underlying)

    return out
