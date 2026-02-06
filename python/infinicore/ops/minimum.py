import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def minimum(
    input: Tensor,
    other: Tensor,
    *,
    out=None,
) -> Tensor:
    r"""Apply the minimum function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.minimum(input, other, out=out)

    if out is None:
        return Tensor(_infinicore.minimum(input._underlying, other._underlying))

    _infinicore.minimum_(input._underlying, other._underlying, out._underlying)
    return out
