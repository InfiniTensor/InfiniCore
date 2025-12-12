from infinicore.lib import _infinicore
from infinicore.tensor import Tensor
import numpy as np
import ml_dtypes
import infinicore


def matrix_power(input, n, *, out=None):
    """
    Compute the n-th power of a square matrix.
    
    Args:
        input: Input square matrix tensor
        n: Power to raise the matrix to
        out: Optional output tensor
    
    Returns:
        Tensor: The result of matrix^n
    """
    shape = input.shape
    if len(shape) < 2 or shape[-2] != shape[-1]:
        raise ValueError("matrix_power: input must be a square matrix")
    
    # Handle n=0 (identity matrix) in Python layer
    if n == 0:
        if out is None:
            result = _create_identity_matrix(shape, input.dtype, input.device)
        else:
            identity = _create_identity_matrix(shape, input.dtype, input.device)
            out.copy_(identity)
            result = out
        return result
    
    if out is None:
        return Tensor(_infinicore.matrix_power(input._underlying, n))
    
    _infinicore.matrix_power_(out._underlying, input._underlying, n)
    return out


def _create_identity_matrix(shape, dtype, device):
    """Helper function to create identity matrix"""
    size = shape[-1]
    
    # Create identity matrix using numpy
    identity_np = np.eye(size, dtype=np.float32)
    
    # Handle batch dimensions if any
    if len(shape) > 2:
        # Expand to batch shape
        batch_shape = shape[:-2]
        identity_np = np.broadcast_to(identity_np, (*batch_shape, size, size))
    
    # Convert numpy dtype to infinicore dtype
    if dtype == infinicore.float16:
        identity_np = identity_np.astype(np.float16)
    elif dtype == infinicore.bfloat16:
        identity_np = identity_np.astype(ml_dtypes.bfloat16)
    elif dtype == infinicore.float32:
        identity_np = identity_np.astype(np.float32)
    elif dtype == infinicore.float64:
        identity_np = identity_np.astype(np.float64)
    else:
        identity_np = identity_np.astype(np.float32)
    
    return infinicore.from_numpy(identity_np, dtype=dtype, device=device)

