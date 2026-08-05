from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def tanh(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.tanh(input._underlying))

    _infinicore.tanh_(out._underlying, input._underlying)
    return out
