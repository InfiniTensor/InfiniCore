import _infinicore

from infinicore.tensor import Tensor


def matmul(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.matmul(input._underlying, other._underlying))

    _infinicore.matmul_(out._underlying, input._underlying, other._underlying)
