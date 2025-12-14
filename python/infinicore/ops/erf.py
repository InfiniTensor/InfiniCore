
import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erf(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        if out is None:
            return infinicore.ntops.torch.erf(input)
        else:
            result = infinicore.ntops.torch.erf(input)
            out.copy_(result)
            return out

    if out is None:
        return Tensor(_infinicore.erf(input._underlying))

    _infinicore.erf_(out._underlying, input._underlying)

    return out