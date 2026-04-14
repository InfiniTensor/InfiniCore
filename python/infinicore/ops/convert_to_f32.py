from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def convert_to_f32(input, *, out=None):
    if out is None:
        return Tensor(_infinicore.convert_to_f32(input._underlying))

    _infinicore.convert_to_f32_(out._underlying, input._underlying)

    return out
