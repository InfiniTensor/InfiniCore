import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def mish(
    input: Tensor, inplace: bool = False
) -> Tensor:
    r"""Applies the Mish activation function element-wise:  mish(x) = x * tanh(softplus(x))."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.mish(input, inplace)

    if inplace:
        _infinicore.mish(input._underlying, inplace)
        return input
    else:
        return Tensor(
            _infinicore.mish(input._underlying, inplace)
        )