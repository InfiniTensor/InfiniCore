import infinicore
from infinicore.tensor import Tensor


def lcm(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Computes element-wise least common multiple. Integer dtypes only."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.lcm(input, other, out=out)
