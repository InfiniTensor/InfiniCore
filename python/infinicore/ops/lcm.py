import infinicore
from infinicore.tensor import Tensor


def lcm(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.lcm(input, other, out=out)

    raise NotImplementedError("lcm is only implemented through the ntops GPU path")
