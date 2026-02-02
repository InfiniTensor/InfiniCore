import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def gt(input: Tensor, other: Tensor | float, *, out: Tensor | None = None) -> Tensor:
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.gt(input, other, out=out)

    if isinstance(other, (int, float)):
        other = Tensor.full(input.shape, other, dtype=input.dtype, device=input.device)

    if out is None:
        return Tensor(_infinicore.gt(input._underlying, other._underlying))

    _infinicore.gt_(input._underlying, other._underlying, out._underlying)
    return out
