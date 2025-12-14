import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def histc(input: Tensor, bins: int = 100, min: float | None = None, max: float | None = None) -> Tensor:
    r"""Apply the logsumexp function."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        is_moore = input._underlying.device.type == _infinicore.Device.Type.MOORE
        return infinicore.ntops.torch.histc(input, bins, min, max, is_moore)

    return Tensor(_infinicore.histc(input._underlying, bins, min, max))
