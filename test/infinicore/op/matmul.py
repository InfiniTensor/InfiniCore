import torch
import infinicore
import sys
import os

# Framework path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    TestConfig,
    TestRunner,
    TestCase,
    create_infinicore_tensor,
    create_test_comparator,
    get_args,
    get_test_devices,
    profile_operation,
    to_torch_dtype,
    InfiniDeviceNames,
    torch_device_map,
)

# ==============================================================================
# Test Setup
# ==============================================================================

# Test cases
_TEST_CASES = [
    # (a_shape, b_shape, result_shape, a_stride, b_stride, c_stride)
    TestCase((2, 3), (3, 4), (2, 4), None, None, None),
    TestCase((128, 256), (256, 64), (128, 64), None, None, None),
    TestCase((2, 4, 2048), (2, 2048, 2048), (2, 4, 2048), None, None, None),
    TestCase((1, 2048), (2048, 2048), (1, 2048), (4096, 1), (4096, 1), (4096, 1)),
    TestCase((6, 2048), (2048, 2560), (6, 2560), (2048, 1), (1, 2048), (2560, 1)),
    TestCase((4, 8 * 6, 64), (4, 64, 6), (4, 8 * 6, 6), None, None, None),
]

# Data types - now using infinicore native types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# ==============================================================================
# Test Method
# ==============================================================================


def test_matmul(device, test_case, dtype, config):
    """
    Test matmul operation

    Args:
        device: device enum
        test_case: test case
        dtype: infinicore data type
        config: test config
    """
    a_shape, b_shape, result_shape, a_stride, b_stride, c_stride = test_case.args

    print(
        f"Testing Matmul on {InfiniDeviceNames[device]} with "
        f"a_shape:{a_shape}, b_shape:{b_shape}, result_shape:{result_shape}, "
        f"a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}, "
        f"dtype:{dtype}"
    )

    # Create PyTorch tensors
    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    torch_a = torch.rand(a_shape, dtype=torch_dtype, device=device_str)
    torch_b = torch.rand(b_shape, dtype=torch_dtype, device=device_str)

    # Calculate PyTorch reference result
    def torch_matmul():
        return torch.matmul(torch_a, torch_b)

    torch_result = torch_matmul()

    # Create infinicore tensors
    infini_a = create_infinicore_tensor(torch_a, device_str)
    infini_b = create_infinicore_tensor(torch_b, device_str)

    # Out-of-place matmul
    def infini_matmul():
        return infinicore.matmul(infini_a, infini_b)

    infini_result = infini_matmul()

    # Create test-specific comparator
    compare_fn = create_test_comparator(config, dtype)

    # Validate results using the test-specific comparator
    is_valid = compare_fn(infini_result, torch_result)
    assert is_valid, "Matmul test failed"

    # Performance test
    if config.bench:
        profile_operation(
            "PyTorch",
            torch_matmul,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )
        profile_operation(
            "Infinicore",
            infini_matmul,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )


def test_matmul_inplace(device, test_case, dtype, config):
    """
    Test in-place matmul operation

    Args:
        device: device enum
        test_case: test case
        dtype: infinicore data type
        config: test config
    """
    a_shape, b_shape, result_shape, a_stride, b_stride, c_stride = test_case.args

    print(
        f"Testing In-place Matmul on {InfiniDeviceNames[device]} with "
        f"a_shape:{a_shape}, b_shape:{b_shape}, result_shape:{result_shape}, "
        f"dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # Create PyTorch tensors
    torch_a = torch.rand(a_shape, dtype=torch_dtype, device=device_str)
    torch_b = torch.rand(b_shape, dtype=torch_dtype, device=device_str)

    # Create pre-allocated result tensor
    torch_preallocated = torch.zeros(result_shape, dtype=torch_dtype, device=device_str)

    # Calculate PyTorch reference result using in-place operation
    def torch_matmul_inplace():
        torch.matmul(torch_a, torch_b, out=torch_preallocated)

    # Execute in-place operation
    torch_matmul_inplace()

    # Create infinicore tensors
    infini_a = create_infinicore_tensor(torch_a, device_str)
    infini_b = create_infinicore_tensor(torch_b, device_str)
    infini_c = infinicore.empty(
        result_shape, dtype=dtype, device=infinicore.device(device_str, 0)
    )

    # Test in-place matmul
    def infini_matmul_inplace():
        infinicore.matmul(infini_a, infini_b, out=infini_c)

    # Execute in-place operation
    infini_matmul_inplace()

    # Create test-specific comparator
    compare_fn = create_test_comparator(config, dtype)

    # Validate results using the test-specific comparator
    is_valid = compare_fn(infini_c, torch_preallocated)
    assert is_valid, "In-place matmul test failed"

    # Performance test
    if config.bench:
        profile_operation(
            "PyTorch In-place",
            torch_matmul_inplace,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )
        profile_operation(
            "Infinicore In-place",
            infini_matmul_inplace,
            device_str,
            config.num_prerun,
            config.num_iterations,
        )


# ==============================================================================
# Main Execution Function
# ==============================================================================


def main():
    args = get_args()

    # Create test configuration
    config = TestConfig(
        tensor_dtypes=_TENSOR_DTYPES,
        tolerance_map=_TOLERANCE_MAP,
        debug=args.debug,
        bench=args.bench,
        num_prerun=args.num_prerun,
        num_iterations=args.num_iterations,
    )

    # Create test runner
    runner = TestRunner(_TEST_CASES, config)

    # Get test devices
    devices = get_test_devices(args)

    print("Starting matmul tests...")

    all_passed = True

    # Run out-of-place tests
    print("\n--- Testing Out-of-place Matmul ---")
    out_of_place_passed = runner.run_tests(devices, test_matmul)
    all_passed = all_passed and out_of_place_passed

    # Run in-place tests
    print("\n--- Testing In-place Matmul ---")
    in_place_passed = runner.run_tests(devices, test_matmul_inplace)
    all_passed = all_passed and in_place_passed

    runner.print_summary()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
