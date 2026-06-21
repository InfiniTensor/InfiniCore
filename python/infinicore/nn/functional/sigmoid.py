from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def sigmoid(input: Tensor, *, out: Tensor | None = None) -> Tensor:
    if out is None:
        return Tensor(_infinicore.sigmoid(input._underlying))

    _infinicore.sigmoid_(out._underlying, input._underlying)
    return out
