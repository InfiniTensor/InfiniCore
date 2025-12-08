from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

# 注意：增加了 out 参数，以及 beta 和 alpha 的默认值
def addbmm(input, batch1, batch2, *, beta=1.0, alpha=1.0, out=None):
    # 1. Out-of-place 模式 (如果没有指定 out)
    if out is None:
        # 错误修复：去掉 .ops，直接调用 _infinicore.addbmm
        # 记得要取 ._underlying 传给 C++
        return Tensor(_infinicore.addbmm(
            input._underlying, 
            batch1._underlying, 
            batch2._underlying, 
            beta, 
            alpha
        ))

    # 2. In-place 模式 (指定了 out)
    # 假设 C++ 端有 addbmm_ (带下划线) 的原地版本，或者重载了支持 out 的版本
    # 这里通常是把 out 作为第一个参数或者通过特定绑定调用
    _infinicore.addbmm_(
        out._underlying, 
        input._underlying, 
        batch1._underlying, 
        batch2._underlying, 
        beta, 
        alpha
    )

    return out