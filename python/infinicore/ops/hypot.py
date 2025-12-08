from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def hypot(input, other, *, out=None):
    # 如果没有提供输出 Tensor，调用返回新 Tensor 的版本
    if out is None:
        return Tensor(_infinicore.hypot(input._underlying, other._underlying))

    # 如果提供了输出 Tensor，调用指定输出的版本 (hypot_)
    _infinicore.hypot_(out._underlying, input._underlying, other._underlying)

    return out