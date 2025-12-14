import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def dot(input: Tensor, tensor: Tensor, *, out=None) -> Tensor:
    r"""Compute the dot product of two 1-D tensors."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.dot(input, tensor, out=out)

    if out is None:
        return Tensor(_infinicore.dot(input._underlying, tensor._underlying))

    _infinicore.dot_(out._underlying, input._underlying, tensor._underlying)
    return out
