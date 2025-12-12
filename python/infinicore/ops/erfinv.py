import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def erfinv(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        if out is None:
            return infinicore.ntops.torch.erfinv(input)
        else:
            result = infinicore.ntops.torch.erfinv(input)
            out.copy_(result)
            return out

        return infinicore.ntops.torch.erfinv(input)

    if out is None:
        return Tensor(_infinicore.erfinv(input._underlying))

    _infinicore.erfinv_(out._underlying, input._underlying)

    return out