"""Identity matrix creation operations."""

from infinicore.context import get_device
from infinicore.dtype import float32
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def eye(n, m=None, *, dtype=None, device=None):
    """Create an identity matrix of shape (n, m).

    Args:
        n: Number of rows.
        m: Number of columns. If not provided, defaults to n (square matrix).
        dtype: Data type of the tensor. Defaults to float32.
        device: Device to create the tensor on. Defaults to current device.

    Returns:
        A 2D tensor with ones on the diagonal and zeros elsewhere.
    """
    if dtype is None:
        dtype = float32
    if device is None:
        device = get_device()

    device_arg = getattr(device, "_underlying", device)
    return Tensor(
        _infinicore.eye(n, m, dtype._underlying, device_arg)
    )

