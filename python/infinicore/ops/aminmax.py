from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def aminmax(input, dim=None, keepdim=False, *, out=None):
    if out is None:
        min_tensor, max_tensor = _infinicore.aminmax(input._underlying, dim, keepdim)
        return (Tensor(min_tensor), Tensor(max_tensor))
    
    # if not isinstance(out, tuple) or len(out) != 2:
    #     raise ValueError("out must be a tuple of (min_tensor, max_tensor)")

     # 接受元组或列表
    if not isinstance(out, (tuple, list)) or len(out) != 2:
        raise ValueError("out must be a tuple or list of (min_tensor, max_tensor)")
    
    
    min_out, max_out = out
    _infinicore.aminmax_(min_out._underlying, max_out._underlying, input._underlying, dim, keepdim)
    return out

