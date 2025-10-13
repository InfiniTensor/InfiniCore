import torch
import time
import infinicore
from .datatypes import to_infinicore_dtype, to_torch_dtype


def synchronize_device(torch_device):
    """Device synchronization"""
    if torch_device == "cuda":
        torch.cuda.synchronize()
    elif torch_device == "npu":
        torch.npu.synchronize()
    elif torch_device == "mlu":
        torch.mlu.synchronize()


def timed_op(func, num_iterations, device):
    """Timed operation"""
    synchronize_device(device)
    start = time.time()
    for _ in range(num_iterations):
        func()
    synchronize_device(device)
    return (time.time() - start) / num_iterations


def profile_operation(desc, func, torch_device, num_prerun, num_iterations):
    """
    Performance profiling workflow
    """
    # Warm-up runs
    for _ in range(num_prerun):
        func()

    # Timed execution
    elapsed = timed_op(lambda: func(), num_iterations, torch_device)
    print(f" {desc} time: {elapsed * 1000 :6f} ms")


def debug(actual, desired, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    Debug function to compare two tensors and print differences
    """
    if actual.dtype == torch.bfloat16 or desired.dtype == torch.bfloat16:
        actual = actual.to(torch.float32)
        desired = desired.to(torch.float32)

    print_discrepancy(actual, desired, atol, rtol, equal_nan, verbose)

    import numpy as np

    np.testing.assert_allclose(
        actual.cpu(), desired.cpu(), rtol, atol, equal_nan, verbose=True
    )


def print_discrepancy(
    actual, expected, atol=0, rtol=1e-3, equal_nan=True, verbose=True
):
    """Print detailed tensor differences"""
    if actual.shape != expected.shape:
        raise ValueError("Tensors must have the same shape to compare.")

    import torch
    import sys

    is_terminal = sys.stdout.isatty()

    actual_isnan = torch.isnan(actual)
    expected_isnan = torch.isnan(expected)

    # Calculate difference mask
    nan_mismatch = (
        actual_isnan ^ expected_isnan if equal_nan else actual_isnan | expected_isnan
    )
    diff_mask = nan_mismatch | (
        torch.abs(actual - expected) > (atol + rtol * torch.abs(expected))
    )
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    delta = actual - expected

    # Display formatting
    col_width = [18, 20, 20, 20]
    decimal_places = [0, 12, 12, 12]
    total_width = sum(col_width) + sum(decimal_places)

    def add_color(text, color_code):
        if is_terminal:
            return f"\033[{color_code}m{text}\033[0m"
        else:
            return text

    if verbose:
        for idx in diff_indices:
            index_tuple = tuple(idx.tolist())
            actual_str = f"{actual[index_tuple]:<{col_width[1]}.{decimal_places[1]}f}"
            expected_str = (
                f"{expected[index_tuple]:<{col_width[2]}.{decimal_places[2]}f}"
            )
            delta_str = f"{delta[index_tuple]:<{col_width[3]}.{decimal_places[3]}f}"
            print(
                f" > Index: {str(index_tuple):<{col_width[0]}}"
                f"actual: {add_color(actual_str, 31)}"
                f"expect: {add_color(expected_str, 32)}"
                f"delta: {add_color(delta_str, 33)}"
            )

        print(add_color(" INFO:", 35))
        print(f"  - Actual dtype: {actual.dtype}")
        print(f"  - Desired dtype: {expected.dtype}")
        print(f"  - Atol: {atol}")
        print(f"  - Rtol: {rtol}")
        print(
            f"  - Mismatched elements: {len(diff_indices)} / {actual.numel()} ({len(diff_indices) / actual.numel() * 100}%)"
        )
        print(
            f"  - Min(actual) : {torch.min(actual):<{col_width[1]}} | Max(actual) : {torch.max(actual):<{col_width[2]}}"
        )
        print(
            f"  - Min(desired): {torch.min(expected):<{col_width[1]}} | Max(desired): {torch.max(expected):<{col_width[2]}}"
        )
        print(
            f"  - Min(delta)  : {torch.min(delta):<{col_width[1]}} | Max(delta)  : {torch.max(delta):<{col_width[2]}}"
        )
        print("-" * total_width + "\n")

    return diff_indices


def get_tolerance(tolerance_map, tensor_dtype, default_atol=0, default_rtol=1e-3):
    """
    Get tolerance settings based on data type
    """
    tolerance = tolerance_map.get(
        tensor_dtype, {"atol": default_atol, "rtol": default_rtol}
    )
    return tolerance["atol"], tolerance["rtol"]


def create_infinicore_tensor(torch_tensor, device_str):
    """Create infinicore tensor from PyTorch tensor"""
    infini_device = infinicore.device(device_str, 0)

    return infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=to_infinicore_dtype(torch_tensor.dtype),
        device=infini_device,
    )


def create_strided_infinicore_tensor(torch_tensor, device_str):
    """Create strided infinicore tensor from PyTorch tensor"""
    infini_device = infinicore.device(device_str, 0)

    return infinicore.strided_from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        list(torch_tensor.stride()),
        dtype=to_infinicore_dtype(torch_tensor.dtype),
        device=infini_device,
    )


def is_tensor_contiguous(tensor):
    """Check if a tensor (PyTorch or infinicore) is contiguous"""
    if hasattr(tensor, "is_contiguous"):
        return tensor.is_contiguous()
    elif hasattr(tensor, "stride"):
        # For PyTorch tensors
        expected_stride = torch._C._compute_contiguous_strides(tensor.shape)
        return list(tensor.stride()) == expected_stride
    else:
        # Assume contiguous by default
        return True


def convert_infinicore_to_torch(infini_result, torch_reference):
    """
    Convert infinicore tensor to PyTorch tensor for comparison

    Args:
        infini_result: infinicore tensor result
        torch_reference: PyTorch tensor reference (for shape and device)
        dtype: infinicore data type
        device_str: torch device string

    Returns:
        torch.Tensor: PyTorch tensor with infinicore data
    """
    torch_result_from_infini = torch.zeros(
        torch_reference.shape,
        dtype=to_torch_dtype(infini_result.dtype),
        device=infini_result.device.type,
    )
    temp_tensor = create_infinicore_tensor(
        torch_result_from_infini, infini_result.device.type
    )
    temp_tensor.copy_(infini_result)
    return torch_result_from_infini


def compare_results(
    infini_result, torch_result, atol=1e-5, rtol=1e-5, debug_mode=False
):
    """
    Generic function to compare infinicore result with PyTorch reference result

    Args:
        infini_result: infinicore tensor result
        torch_result: PyTorch tensor reference result
        atol: absolute tolerance
        rtol: relative tolerance
        debug_mode: whether to enable debug output

    Returns:
        bool: True if results match within tolerance
    """
    # Convert infinicore result to PyTorch tensor for comparison
    torch_result_from_infini = convert_infinicore_to_torch(infini_result, torch_result)

    # Debug mode: detailed comparison
    if debug_mode:
        debug(torch_result_from_infini, torch_result, atol=atol, rtol=rtol)

    # Check if results match within tolerance
    return torch.allclose(torch_result_from_infini, torch_result, atol=atol, rtol=rtol)


def create_test_comparator(config, dtype, tolerance_map=None):
    """
    Create a test-specific comparison function that handles test configuration

    Args:
        config: test configuration
        dtype: infinicore data type
        tolerance_map: optional tolerance map (defaults to config's tolerance_map)

    Returns:
        callable: function that takes (infini_result, torch_result) and returns bool
    """
    if tolerance_map is None:
        tolerance_map = config.tolerance_map

    atol, rtol = get_tolerance(tolerance_map, dtype)

    def compare_test_results(infini_result, torch_result):
        return compare_results(
            infini_result, torch_result, atol=atol, rtol=rtol, debug_mode=config.debug
        )

    return compare_test_results


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
        if new_strides[i] > 0:
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
    new_tensor.view(-1).index_add_(0, new_positions, tensor.view(-1))
    new_tensor.set_(new_tensor.untyped_storage(), offset, shape, tuple(new_strides))

    return new_tensor
