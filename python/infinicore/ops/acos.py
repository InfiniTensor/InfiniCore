from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

def acos(input, *, out=None):
    # 1. 如果没有指定 out，创建新 Tensor 返回
    if out is None:
        return Tensor(_infinicore.acos(input._underlying))
    _infinicore.acos_(out._underlying, input._underlying)

    return out