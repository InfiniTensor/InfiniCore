from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def topk(input,  k,  dim=None,  largest=True,  sorted=True, out=None):
    if out is None:
        return Tensor(_infinicore.sum(input._underlying, k,  dim,  largest,  sorted))

    _infinicore.sum_(out._underlying, input._underlying,  k,  dim,  largest,  sorted)

    return out
