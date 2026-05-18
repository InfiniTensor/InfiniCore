import infinicore
from infinicore.tensor import Tensor


def copysign(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes element-wise copysign: magnitude of input with sign of other."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.copysign(input, other, out=out)
