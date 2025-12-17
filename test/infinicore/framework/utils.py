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
    elif torch_device == "musa":
        torch.musa.synchronize()


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
    # Note: bool tensors are handled inside print_discrepancy

    print_discrepancy(actual, desired, atol, rtol, equal_nan, verbose)

    import numpy as np

    # For bool tensors, use assert_equal instead of assert_allclose
    if actual.dtype == torch.bool or desired.dtype == torch.bool:
        np.testing.assert_equal(
            actual.cpu().numpy(), desired.cpu().numpy()
        )
    else:
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
    
    # Handle bool tensors specially - PyTorch doesn't support subtraction for bool
    is_bool = actual.dtype == torch.bool or expected.dtype == torch.bool
    
    if is_bool:
        # For bool tensors, convert to int8 for comparison operations
        actual_for_calc = actual.to(torch.int8) if actual.dtype == torch.bool else actual
        expected_for_calc = expected.to(torch.int8) if expected.dtype == torch.bool else expected
    else:
        actual_for_calc = actual
        expected_for_calc = expected

    actual_isnan = torch.isnan(actual_for_calc)
    expected_isnan = torch.isnan(expected_for_calc)

    # Calculate difference mask
    if is_bool:
        # For bool tensors, just check equality
        diff_mask = actual != expected
    else:
        nan_mismatch = (
            actual_isnan ^ expected_isnan if equal_nan else actual_isnan | expected_isnan
        )
        diff_mask = nan_mismatch | (
            torch.abs(actual_for_calc - expected_for_calc) > (atol + rtol * torch.abs(expected_for_calc))
        )
    
    diff_indices = torch.nonzero(diff_mask, as_tuple=False)
    
    # Calculate delta (difference) - convert bool to int if needed
    if is_bool:
        delta = (actual.to(torch.int8) - expected.to(torch.int8))
    else:
        delta = actual_for_calc - expected_for_calc

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
            if is_bool:
                # For bool, display as True/False
                actual_val = actual[index_tuple].item()
                expected_val = expected[index_tuple].item()
                actual_str = f"{str(actual_val):<{col_width[1]}}"
                expected_str = f"{str(expected_val):<{col_width[2]}}"
                delta_val = delta[index_tuple].item()
                delta_str = f"{delta_val:<{col_width[3]}}"
            else:
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
        print(f"  - Equal NaN: {equal_nan}")
        print(
            f"  - Mismatched elements: {len(diff_indices)} / {actual.numel()} ({len(diff_indices) / actual.numel() * 100}%)"
        )
        
        if not is_bool:
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


def compare_results(
    infini_result, torch_result, atol=1e-5, rtol=1e-5, equal_nan=False, debug_mode=False
):
    """
    Generic function to compare infinicore result with PyTorch reference result
    Supports both single and multiple outputs

    Args:
        infini_result: infinicore tensor result (single or tuple)
        torch_result: PyTorch tensor reference result (single or tuple)
        atol: absolute tolerance (for floating-point only)
        rtol: relative tolerance (for floating-point only)
        equal_nan: whether to treat NaN as equal
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
            match = compare_results(
                infini_out, torch_out, atol, rtol, equal_nan, debug_mode
            )
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
    if isinstance(infini_result, torch.Tensor):
        torch_result_from_infini = infini_result
    else:
        torch_result_from_infini = convert_infinicore_to_torch(infini_result)

    # Debug mode: detailed comparison
    if debug_mode:
        debug(
            torch_result_from_infini,
            torch_result,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )

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
            torch_result_from_infini.real,
            torch_result.real,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )
        imag_close = torch.allclose(
            torch_result_from_infini.imag,
            torch_result.imag,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
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
            torch_result_from_infini,
            torch_result,
            atol=atol,
            rtol=rtol,
            equal_nan=equal_nan,
        )


def create_test_comparator(config, atol, rtol, mode_name="", equal_nan=False):
    """
    Create a test-specific comparison function

    Args:
        config: test configuration
        atol: absolute tolerance (for floating-point only)
        rtol: relative tolerance (for floating-point only)
        mode_name: operation mode name for debug output
        equal_nan: whether to treat NaN as equal

    Returns:
        callable: function that takes (infini_result, torch_result) and returns bool
    """

    def compare_test_results(infini_result, torch_result):
        if config.debug and mode_name:
            print(f"\033[94mDEBUG INFO - {mode_name}:\033[0m")
            print(
                f"\033[94m  Equal NaN: {'enabled' if equal_nan else 'disabled'}\033[0m"
            )

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
            equal_nan=equal_nan,
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
