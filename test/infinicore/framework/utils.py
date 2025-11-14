import torch
import time
import infinicore
import numpy as np
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
    return time.time() - start


def profile_operation(
    desc, func, torch_device, num_prerun, num_iterations, total=False
):
    """
    Performance profiling workflow
    """
    # Warm-up runs
    for _ in range(num_prerun):
        func()

    # Timed execution
    elapsed = timed_op(lambda: func(), num_iterations, torch_device)
    print(f"    {desc} time: {elapsed / num_iterations * 1000 :6f} ms")
    if total:
        return elapsed
    else:
        return elapsed / num_iterations


def debug(actual, desired, atol=0, rtol=1e-2, equal_nan=False, verbose=True):
    """
    Debug function to compare two tensors and print differences
    """
    # Handle complex types by converting to real representation for comparison
    if actual.is_complex() or desired.is_complex():
        actual = torch.view_as_real(actual)
        desired = torch.view_as_real(desired)
    elif actual.dtype == torch.bfloat16 or desired.dtype == torch.bfloat16:
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
        print("-" * total_width)

    return diff_indices


def get_tolerance(tolerance_map, tensor_dtype, default_atol=0, default_rtol=1e-3):
    """
    Get tolerance settings based on data type
    """
    tolerance = tolerance_map.get(
        tensor_dtype, {"atol": default_atol, "rtol": default_rtol}
    )
    return tolerance["atol"], tolerance["rtol"]


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
    temp_tensor = infinicore_tensor_from_torch(torch_result_from_infini)
    temp_tensor.copy_(infini_result)
    return torch_result_from_infini


def compare_results(
    infini_result, torch_result, atol=1e-5, rtol=1e-5, debug_mode=False
):
    """
    Generic function to compare infinicore result with PyTorch reference result
    Supports both single and multiple outputs

    Args:
        infini_result: infinicore tensor result (single or tuple)
        torch_result: PyTorch tensor reference result (single or tuple)
        atol: absolute tolerance (for floating-point only)
        rtol: relative tolerance (for floating-point only)
        debug_mode: whether to enable debug output

    Returns:
        bool: True if all results match within tolerance
    """
    # Handle multiple outputs
    if isinstance(infini_result, (tuple, list)) and isinstance(
        torch_result, (tuple, list)
    ):
        if len(infini_result) != len(torch_result):
            return False

        all_match = True
        for i, (infini_out, torch_out) in enumerate(zip(infini_result, torch_result)):
            match = compare_results(infini_out, torch_out, atol, rtol, debug_mode)
            all_match = all_match and match

        return all_match

    # Handle scalar and bool comparisons
    if not isinstance(torch_result, torch.Tensor):
        is_infini_int = isinstance(infini_result, (int, np.integer))
        is_torch_int = isinstance(torch_result, (int, np.integer))
        if isinstance(infini_result, bool) and isinstance(torch_result, bool):
            # Bool comparison
            result_equal = infini_result == torch_result
            if debug_mode:
                status = "match" if result_equal else "mismatch"
                print(
                    f"Boolean values {status}: {infini_result} {'==' if result_equal else '!='} {torch_result}"
                )
            return result_equal
        elif is_infini_int and is_torch_int:
            # Exact integer scalar comparison
            result_equal = infini_result == torch_result
            if debug_mode:
                status = "match" if result_equal else "mismatch"
                print(
                    f"Integer scalar {status}: {infini_result} {'==' if result_equal else '!='} {torch_result}"
                )
            return result_equal
        else:
            # Floating-point scalar comparison with tolerance
            result_equal = abs(infini_result - torch_result) <= atol + rtol * abs(
                torch_result
            )
            if debug_mode:
                status = "match" if result_equal else "mismatch"
                print(
                    f"Floating-point scalar {status}: {infini_result} {'~=' if result_equal else '!~='} {torch_result} (tolerance: {atol + rtol * abs(torch_result)})"
                )
            return result_equal

    # Convert infinicore result to PyTorch tensor for comparison
    torch_result_from_infini = convert_infinicore_to_torch(infini_result)

    # Debug mode: detailed comparison
    if debug_mode:
        debug(torch_result_from_infini, torch_result, atol=atol, rtol=rtol)

    # Choose comparison method based on data type
    if is_integer_dtype(torch_result_from_infini.dtype) or is_integer_dtype(
        torch_result.dtype
    ):
        # Exact equality for integer types
        result_equal = torch.equal(torch_result_from_infini, torch_result)
        if debug_mode and not result_equal:
            print("Integer tensor comparison failed - requiring exact equality")
        return result_equal
    elif is_complex_dtype(torch_result_from_infini.dtype) or is_complex_dtype(
        torch_result.dtype
    ):
        # Complex number comparison - compare real and imaginary parts separately
        real_close = torch.allclose(
            torch_result_from_infini.real, torch_result.real, atol=atol, rtol=rtol
        )
        imag_close = torch.allclose(
            torch_result_from_infini.imag, torch_result.imag, atol=atol, rtol=rtol
        )
        result_equal = real_close and imag_close
        if debug_mode and not result_equal:
            print("Complex tensor comparison failed")
            if not real_close:
                print("  Real parts don't match")
            if not imag_close:
                print("  Imaginary parts don't match")
        return result_equal
    else:
        # Tolerance-based comparison for floating-point types
        return torch.allclose(
            torch_result_from_infini, torch_result, atol=atol, rtol=rtol
        )


def create_test_comparator(config, atol, rtol, mode_name=""):
    """
    Create a test-specific comparison function

    Args:
        config: test configuration
        atol: absolute tolerance (for floating-point only)
        rtol: relative tolerance (for floating-point only)
        mode_name: operation mode name for debug output

    Returns:
        callable: function that takes (infini_result, torch_result) and returns bool
    """

    def compare_test_results(infini_result, torch_result):
        if config.debug and mode_name:
            print(f"\033[94mDEBUG INFO - {mode_name}:\033[0m")

        # For integer types, override tolerance to require exact equality
        actual_atol = atol
        actual_rtol = rtol

        # Check if we're dealing with integer types
        try:
            # Try to get dtype from infinicore tensor
            if hasattr(infini_result, "dtype"):
                infini_dtype = infini_result.dtype
                torch_dtype = to_torch_dtype(infini_dtype)
                if is_integer_dtype(torch_dtype):
                    actual_atol = 0
                    actual_rtol = 0
        except:
            pass

        return compare_results(
            infini_result,
            torch_result,
            atol=actual_atol,
            rtol=actual_rtol,
            debug_mode=config.debug,
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
