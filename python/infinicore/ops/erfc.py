import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erfc(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        if out is None:
            return infinicore.ntops.torch.erfc(input)
        else:
            result = infinicore.ntops.torch.erfc(input)
            out.copy_(result)
            return out

    if out is None:
        return Tensor(_infinicore.erfc(input._underlying))

    _infinicore.erfc_(out._underlying, input._underlying)

    return out