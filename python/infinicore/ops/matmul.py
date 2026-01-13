from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def matmul(input, other, *, alpha=1.0, out=None):
    if out is None:
        return Tensor(_infinicore.matmul(input._underlying, other._underlying, alpha))

    print("In matmul python")
    _infinicore.matmul_(out._underlying, input._underlying, other._underlying, alpha)

    return out
