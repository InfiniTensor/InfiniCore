import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def round(input, *, decimals=0, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.round(input, decimals=decimals)

    if out is None:
        return Tensor(_infinicore.round(input._underlying, decimals))

    _infinicore.round_(out._underlying, input._underlying, decimals)

    return out
