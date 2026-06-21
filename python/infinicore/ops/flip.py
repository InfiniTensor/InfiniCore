import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
import torch
def _normalize_dims(dims, ndim):
    if isinstance(dims, int):
        dims = (dims,)
    dims = tuple(dim if dim >= 0 else dim + ndim for dim in dims)
    assert all(0 <= dim < ndim for dim in dims), "`dims` out of range."
    result = []
    for dim in dims:
        if dim not in result:
            result.append(dim)
    return tuple(result)
def _is_dims(x):
    return isinstance(x, int) or (
        isinstance(x, (tuple, list)) and all(isinstance(i, int) for i in x)
    )
def flip(*args) -> Tensor:
    r"""Reverse the order of an n-D tensor along given dims."""
    assert len(args) >= 2, "`flip` requires input and dims."
    dims = args[-1]
    assert _is_dims(dims), "`dims` must be int, tuple[int], or list[int]."
    input = None
    for arg in args[:-1]:
        if isinstance(arg, Tensor) or hasattr(arg, "_underlying"):
            input = arg
            break
        if isinstance(arg, torch.Tensor):
            return torch.flip(arg, dims)
    assert input is not None, "`flip` requires a Tensor input."
    dims = _normalize_dims(dims, input.ndim)
    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.flip(input, dims)
    return Tensor(
        _infinicore.flip(
            input._underlying,
            dims,
        )
    )