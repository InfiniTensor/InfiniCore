import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def atan2(
    input: Tensor,
    other: Tensor,
    *,
    out=None,
) -> Tensor:
    r"""Apply the atan2 function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.atan2(input, other, out=out)

    if out is None:
        return Tensor(_infinicore.atan2(input._underlying, other._underlying))

    _infinicore.atan2_(input._underlying, other._underlying, out._underlying)
    return out
