from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def silu_and_mul(input: Tensor, out=None) -> Tensor:
    r"""Apply the SiLU and Mul (SwiGLU) function.
    
    Formula: output = SiLU(input_gate) * input_up
    Input shape: [..., 2*d], Output shape: [..., d]
    """

    if out is None:
        # 调用 C++ 非原地接口，内部处理输出 Tensor 的创建
        return Tensor(_infinicore.silu_and_mul(input._underlying))

    # 调用 C++ 原地/指定输出接口
    _infinicore.silu_and_mul_(out._underlying, input._underlying)

    return out
