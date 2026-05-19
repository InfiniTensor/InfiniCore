import infinicore
from infinicore.tensor import Tensor


def nextafter(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    r"""Returns the next representable float value of input toward other, element-wise."""
    assert infinicore.use_ntops
    return infinicore.ntops.torch.nextafter(input, other, out=out)
