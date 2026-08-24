# python/infinicore/ops/softshrink/__init__.py
import torch
from .._C import softshrink as _softshrink
def softshrink(input, lambda=0.5):
    if input.is_cuda and input.dtype == torch.float16:
        return _softshrink(input, lambda)
    return torch.nn.functional.softshrink(input.float(), lambda).to(input.dtype)