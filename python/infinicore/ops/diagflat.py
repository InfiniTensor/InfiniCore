from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
import infinicore


def diagflat(input, *, offset=0):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.diagflat(input, offset=offset)

    
    return Tensor(_infinicore.diagflat(input._underlying, offset))

    _infinicore.diagflat_(out._underlying, input._underlying, offset)
    return out


