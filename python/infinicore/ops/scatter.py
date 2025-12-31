from typing import Optional
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor

# Scatter 算子常用的 reduction 模式
_SCATTER_REDUCTION_MODES = {
    "none": 0,      # 直接赋值/覆盖
    "add": 1,       # 累加
    "multiply": 2,  # 累乘
}

# -----------------------------------------------------------------------------
# 修改点 1: 调整函数签名
# 将 dim 移动到所有 Tensor 参数 (input, index, src) 之后
# 这样 func(*[t1, t2, t3], dim=1) 才能正确解析
# -----------------------------------------------------------------------------
def scatter(
    input: Tensor, 
    index: Tensor,    # <--- index 移到这里
    src: Tensor,      # <--- src 移到这里
    dim: int,         # <--- dim 移到后面
    reduction: str = "none", 
    *, 
    out: Optional[Tensor] = None
) -> Tensor:
    r"""Writes all values from the tensor src into input at the indices specified in the index tensor.
    """

    if not input.is_contiguous():
        input = input.contiguous()
    if not index.is_contiguous():
        index = index.contiguous()
    if not src.is_contiguous():
        src = src.contiguous()
    
    # 解析 reduction 参数
    if reduction not in _SCATTER_REDUCTION_MODES:
        raise ValueError(f"{reduction} is not a valid value for reduction")
    reduction_val = _SCATTER_REDUCTION_MODES[reduction]

    # -------------------------------------------------------------------------
    # 修改点 2: 调整底层 C++ 调用顺序
    # 既然您之前已经修改了 C++ bind_scatter 为 (input, index, src, dim, reduction)
    # 这里必须严格匹配那个顺序
    # -------------------------------------------------------------------------
    
    # In-place 分支 (scatter_)
    if out is not None:
        _infinicore.scatter_(
            out._underlying,
            input._underlying,
            index._underlying,  # index (第3个)
            src._underlying,    # src (第4个)
            dim,                # dim (第5个)
            reduction_val
        )
        return out

    # Out-of-place 分支 (scatter)
    return Tensor(
        _infinicore.scatter(
            input._underlying,
            index._underlying,  # index (第2个)
            src._underlying,    # src (第3个)
            dim,                # dim (第4个)
            reduction_val
        )
    )