import torch
import time
import sys


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
