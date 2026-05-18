import infinicore
from infinicore.tensor import Tensor


def rad2deg(input: Tensor, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.rad2deg(input, out=out)

    raise NotImplementedError("rad2deg is only implemented through the ntops GPU path")
