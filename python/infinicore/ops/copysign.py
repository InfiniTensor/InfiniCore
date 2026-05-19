import infinicore
from infinicore.tensor import Tensor


def copysign(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.copysign(input, other, out=out)

    raise NotImplementedError("copysign is only implemented through the ntops GPU path")
