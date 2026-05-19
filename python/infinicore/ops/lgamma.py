import infinicore
from infinicore.tensor import Tensor


def lgamma(input: Tensor, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.lgamma(input, out=out)

    raise NotImplementedError("lgamma is only implemented through the ntops GPU path")
