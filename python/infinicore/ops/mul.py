import numbers

from infinicore.lib import _infinicore
from infinicore.ops.mul_scalar import mul_scalar
from infinicore.tensor import Tensor


def mul(input, other, *, out=None):
    if isinstance(other, numbers.Real):
        return mul_scalar(input, other, out=out)

    if out is None:
        return Tensor(_infinicore.mul(input._underlying, other._underlying))

    _infinicore.mul_(out._underlying, input._underlying, other._underlying)

    return out
