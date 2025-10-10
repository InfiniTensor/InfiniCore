from infinicore.tensor import Tensor

from .. import _infinicore


def rearrange(input, other, *, out=None):
    if out is None:
        return Tensor(_infinicore.rearrange(input._underlying))

    _infinicore.rearrange_(out._underlying, input._underlying)
