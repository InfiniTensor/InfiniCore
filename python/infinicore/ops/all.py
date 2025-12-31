from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
#inputs=[in_0: tensor(4, 5, 6), strides=(60, 12, 2), uint8], kwargs={dim=2; keepdim=True; out=tensor(4, 5, 1), strides=(12, 4, 1), bool})
def all(input, dim=None, keepdim=False, out=None):
    if out is None:
        return Tensor(_infinicore.all(input._underlying, dim, keepdim))

    _infinicore.all_(out._underlying, input._underlying, dim, keepdim)

    return out
