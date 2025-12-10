from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
import infinicore


def sqrt(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.sqrt(input, out=out)
    
    if out is None:
        return Tensor(_infinicore.sqrt(input._underlying))

    _infinicore.sqrt_(out._underlying, input._underlying)
    return out