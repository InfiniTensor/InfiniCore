from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
import infinicore


def matmul(input, other, *, out=None):
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.matmul(input, other)
    
    if out is None:
        return Tensor(_infinicore.matmul(input._underlying, other._underlying))

    _infinicore.matmul_(out._underlying, input._underlying, other._underlying)

    return out
