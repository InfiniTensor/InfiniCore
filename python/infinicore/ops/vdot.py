from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def vdot(a: Tensor, b: Tensor) -> Tensor:
    """
    InfiniCore vdot: 1D vector dot product, aligned with torch.vdot
    for real-valued tensors (no complex conjugation).
    """
    return Tensor(_infinicore.vdot(a._underlying, b._underlying))


