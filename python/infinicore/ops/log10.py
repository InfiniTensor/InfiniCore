import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def log10(input: Tensor, *, out=None) -> Tensor:
    r"""Apply the Sigmoid Linear Unit (SiLU) function, element-wise."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.log10(input, out=out)

    if out is None:
        return Tensor(_infinicore.log10(input._underlying))

    _infinicore.log10_(out._underlying, input._underlying)
    return out
