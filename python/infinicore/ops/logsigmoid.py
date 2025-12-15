from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logsigmoid(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.logsigmoid(input._underlying))

    _infinicore.logsigmoid_(out._underlying, input._underlying)

    return out

