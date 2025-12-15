from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, from_torch

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

        # Prefer using the original Torch tensor reference when available
        cond_torch = getattr(cond, "_torch_ref", None)
        if cond_torch is None:
            # Fallback: create a Torch tensor, then copy data from infinicore tensor.
            # Tests use CPU bool tensors for condition-only where.
            cond_torch = torch.zeros(
                cond.shape,
                dtype=torch.bool,
                device="cpu",
            )
            # Share storage between Torch tensor and an infinicore view, then copy.
            ic_view = from_torch(cond_torch)
            ic_view.copy_(cond)

        idx_tensors = torch.where(cond_torch)
        # torch.where(cond) returns a tuple of index tensors; mirror that with
        # infinicore tensors sharing the same underlying storage.
        return tuple(from_torch(t) for t in idx_tensors)

    if len(args) != 3:
        raise TypeError("infinicore.where expects (cond, x, y)")

    cond, x, y = args

    if out is None:
        return Tensor(_infinicore.where(cond._underlying, x._underlying, y._underlying))

    _infinicore.where_(out._underlying, cond._underlying, x._underlying, y._underlying)
    return out



