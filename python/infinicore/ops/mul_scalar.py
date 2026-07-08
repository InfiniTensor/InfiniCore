from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mul_scalar(input, alpha, *, out=None):
    alpha = float(alpha)
    if out is None:
        return Tensor(_infinicore.mul_scalar(input._underlying, alpha))

    _infinicore.mul_scalar_(out._underlying, input._underlying, alpha)

    return out
