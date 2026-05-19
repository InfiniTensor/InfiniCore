import infinicore
from infinicore.tensor import Tensor


def rad2deg(input: Tensor, *, out=None) -> Tensor:
    r"""Converts angles in radians to degrees, element-wise."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.rad2deg(input, out=out)
