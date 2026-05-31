import infinicore
from infinicore.tensor import Tensor


def nextafter(input: Tensor, other: Tensor, *, out=None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.nextafter(input, other, out=out)

    raise NotImplementedError("nextafter is only implemented through the ntops GPU path")
