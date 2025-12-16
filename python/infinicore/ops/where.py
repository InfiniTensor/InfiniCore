from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, from_torch
import infinicore

import torch


def where(*args, out=None):
    """Elementwise where(cond, x, y) selection.

    Two supported call patterns:
      - where(cond, x, y) -> Tensor
      - where(cond, x, y, out=...) -> out

    The condition-only variant where(cond) returning indices is implemented
    by delegating to the underlying Torch tensor stored in cond._torch_ref.
    """
    # condition-only mode: where(cond) -> indices tuple
    if len(args) == 1:
        cond = args[0]

        # Use native infiniop implementation
        idx_tensors = _infinicore.where_indices(cond._underlying)
        # Convert C++ Tensor objects to Python Tensor objects
        return tuple(Tensor(t) for t in idx_tensors)

    if len(args) != 3:
        raise TypeError("infinicore.where expects (cond, x, y)")

    cond, x, y = args

    if infinicore.use_ntops and x.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.where(cond, x, y, out=out)

    if out is None:
        return Tensor(_infinicore.where(cond._underlying, x._underlying, y._underlying))

    _infinicore.where_(out._underlying, cond._underlying, x._underlying, y._underlying)
    return out
