from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def add_rms_norm(a, b, weight, epsilon=1e-5, *, out=None):
    """
    Fused Add and RMS Normalization.

    Args:
        a: First input tensor
        b: Second input tensor
        weight: Scale weights
        epsilon: Small constant for numerical stability, default is 1e-5
        out: Optional output tuple (y, residual_out) for in-place operation

    Returns:
        Tuple of (normalized_result, add_result): (RMSNorm(a + b) * weight, a + b)
        The add_result can be used as residual for subsequent layers.
    """
    if out is None:
        result = _infinicore.add_rms_norm(
            a._underlying, b._underlying, weight._underlying, epsilon
        )
        return (Tensor(result[0]), Tensor(result[1]))

    y, residual_out = out
    _infinicore.add_rms_norm_(
        y._underlying,
        residual_out._underlying,
        a._underlying,
        b._underlying,
        weight._underlying,
        epsilon,
    )
    return (y, residual_out)


def add_rms_norm_(y, residual_out, a, b, weight, epsilon=1e-5):
    """In-place Fused Add and RMS Normalization."""
    _infinicore.add_rms_norm_(
        y._underlying,
        residual_out._underlying,
        a._underlying,
        b._underlying,
        weight._underlying,
        epsilon,
    )
