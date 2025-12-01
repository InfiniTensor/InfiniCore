import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def elu(input: Tensor, alpha: float = 1.0, inplace: bool = False, *, out=None) -> Tensor:
    r"""Apply the Exponential Linear Unit (ELU) function, element-wise.
    
    ELU(x) = x if x >= 0 else alpha * (exp(x) - 1)
    
    Args:
        input: Input tensor
        alpha: ELU parameter (default: 1.0)
        inplace: If True, performs the operation in-place (default: False)
        out: Optional output tensor for in-place operation
    
    Returns:
        Output tensor with ELU applied element-wise.
    """
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.elu(input, alpha=alpha, inplace=inplace)

    if inplace:
        _infinicore.elu_(input._underlying, input._underlying, alpha)
        return input

    if out is None:
        return Tensor(_infinicore.elu(input._underlying, alpha))

    _infinicore.elu_(out._underlying, input._underlying, alpha)
    return out

