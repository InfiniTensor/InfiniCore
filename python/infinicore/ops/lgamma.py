import infinicore
from infinicore.tensor import Tensor


def lgamma(input: Tensor, *, out=None) -> Tensor:
    r"""Computes element-wise natural logarithm of the absolute value of the gamma function."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.lgamma(input, out=out)
