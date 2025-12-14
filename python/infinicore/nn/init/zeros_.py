import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def zeros_(input: Tensor) -> Tensor:
    r"""Fill the input tensor with the scalar value 0."""
    _infinicore.zeros_(input._underlying)
    return input
