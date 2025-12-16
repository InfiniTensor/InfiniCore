import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def logsigmoid(input, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.logsigmoid(input, out=out)

    if out is None:
        return Tensor(_infinicore.logsigmoid(input._underlying))

    _infinicore.logsigmoid_(out._underlying, input._underlying)

    return out
