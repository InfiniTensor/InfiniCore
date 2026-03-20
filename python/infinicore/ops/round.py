import infinicore
from infinicore.tensor import Tensor


def round(input: Tensor, decimals=0, *, out=None) -> Tensor:
    r"""Round elements to the nearest integer, with banker's rounding."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.round(input, decimals=decimals)

    if out is None:
        return infinicore.ntops.torch.round(input, decimals=decimals)

    result = infinicore.ntops.torch.round(input, decimals=decimals)
    out.copy_(result)
    return out