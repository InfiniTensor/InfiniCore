import torch
import infinicore
from ..datatypes import to_infinicore_dtype, to_torch_dtype

# =================================================================
#  Device & Synchronization
# =================================================================


def synchronize_device(torch_device):
    """Device synchronization"""
    if torch_device == "cuda":
        torch.cuda.synchronize()
    elif torch_device == "npu":
        torch.npu.synchronize()
    elif torch_device == "mlu":
        torch.mlu.synchronize()
    elif torch_device == "musa":
        torch.musa.synchronize()


# =================================================================
#  Tensor Operations & Conversions
# =================================================================


def clone_torch_tensor(torch_tensor):
    cloned = torch_tensor.clone().detach()
    if not torch_tensor.is_contiguous():
        cloned = rearrange_tensor(cloned, torch_tensor.stride())
    return cloned


def infinicore_tensor_from_torch(torch_tensor):
    infini_device = infinicore.device(torch_tensor.device.type, 0)
    if torch_tensor.is_contiguous():
        return infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infini_device,
        )
    else:
        return infinicore.strided_from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            list(torch_tensor.stride()),
            dtype=to_infinicore_dtype(torch_tensor.dtype),
            device=infini_device,
        )


def convert_infinicore_to_torch(infini_result):
    """
    Convert infinicore tensor to PyTorch tensor for comparison

    Args:
        infini_result: infinicore tensor result

    Returns:
        torch.Tensor: PyTorch tensor with infinicore data
    """
    torch_result_from_infini = torch.zeros(
        infini_result.shape,
        dtype=to_torch_dtype(infini_result.dtype),
        device=infini_result.device.type,
    )
    if not infini_result.is_contiguous():
        torch_result_from_infini = rearrange_tensor(
            torch_result_from_infini, infini_result.stride()
        )
    temp_tensor = infinicore_tensor_from_torch(torch_result_from_infini)
    temp_tensor.copy_(infini_result)
    return torch_result_from_infini


def rearrange_tensor(tensor, new_strides):
    """
    Given a PyTorch tensor and a list of new strides, return a new PyTorch tensor with the given strides.
    """
    import torch

    shape = tensor.shape

    new_size = [0] * len(shape)
    left = 0
    right = 0
    for i in range(len(shape)):
        if new_strides[i] >= 0:
            new_size[i] = (shape[i] - 1) * new_strides[i] + 1
            right += new_strides[i] * (shape[i] - 1)
        else:  # TODO: Support negative strides in the future
            # new_size[i] = (shape[i] - 1) * (-new_strides[i]) + 1
            # left += new_strides[i] * (shape[i] - 1)
            raise ValueError("Negative strides are not supported yet")

    # Create a new tensor with zeros
    new_tensor = torch.zeros(
        (right - left + 1,), dtype=tensor.dtype, device=tensor.device
    )

    # Generate indices for original tensor based on original strides
    indices = [torch.arange(s) for s in shape]
    mesh = torch.meshgrid(*indices, indexing="ij")

    # Flatten indices for linear indexing
    linear_indices = [m.flatten() for m in mesh]

    # Calculate new positions based on new strides
    new_positions = sum(
        linear_indices[i] * new_strides[i] for i in range(len(shape))
    ).to(tensor.device)
    offset = -left
    new_positions += offset

    # Copy the original data to the new tensor
    new_tensor.reshape(-1).index_add_(0, new_positions, tensor.reshape(-1))
    new_tensor.set_(new_tensor.untyped_storage(), offset, shape, tuple(new_strides))

    return new_tensor


def is_broadcast(strides):
    """
    Check if strides indicate a broadcasted tensor

    Args:
        strides: Tensor strides or None

    Returns:
        bool: True if the tensor is broadcasted (has zero strides)
    """
    if strides is None:
        return False
    return any(s == 0 for s in strides)


# =================================================================
#  Type Checks (Moved here to avoid circular imports in check.py)
# =================================================================


def is_integer_dtype(dtype):
    """Check if dtype is integer type"""
    return dtype in [
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
        torch.bool,
    ]


def is_complex_dtype(dtype):
    """Check if dtype is complex type"""
    return dtype in [torch.complex64, torch.complex128]


def is_floating_dtype(dtype):
    """Check if dtype is floating-point type"""
    return dtype in [
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ]
