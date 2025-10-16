import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    TestWorkspace,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, x_stride, y_stride
    ((3, 3), None, None),
    ((32, 512), None, None),
    ((32, 512), (1024, 1), (1024, 1)),
    ((32, 5, 5), None, None),
    ((32, 20, 512), None, None),
    ((32, 20, 512), (20480, 512, 1), None),
    ((28, 15, 15), None, None),
    ((1, 1000), None, None),
    ((16, 50257), None, None),
    ((4, 8, 256), None, None),
    ((2, 16, 1024), None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

# Mixed precision test cases - support y_dtype == x_dtype or y_dtype == F32
_MIXED_PRECISION_CASES = [
    (InfiniDtype.F16, InfiniDtype.F32),
    (InfiniDtype.BF16, InfiniDtype.F32),
    (InfiniDtype.F16, InfiniDtype.F16),
    (InfiniDtype.BF16, InfiniDtype.BF16),
    (InfiniDtype.F32, InfiniDtype.F32),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.INPLACE_X,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def logsoftmax(x):
    """PyTorch reference implementation of log_softmax"""
    return torch.nn.functional.log_softmax(x.to(torch.float32), dim=-1)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing LogSoftmax on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    x = TestTensor(shape, x_stride, dtype, device)
    ans = logsoftmax(x.actual_tensor())

    # Convert answer to match input dtype for default behavior
    if dtype == InfiniDtype.F16:
        ans = ans.to(torch.float16)
    elif dtype == InfiniDtype.BF16:
        ans = ans.to(torch.bfloat16)
    elif dtype == InfiniDtype.F32:
        ans = ans.to(torch.float32)

    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(shape, y_stride, dtype, device)  # Default: same dtype as input

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    status = LIBINFINIOP.infiniopCreateLogSoftmaxDescriptor(
        handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
    )
    check_error(status)

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    status = LIBINFINIOP.infiniopGetLogSoftmaxWorkspaceSize(
        descriptor, ctypes.byref(workspace_size)
    )
    check_error(status)
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_logsoftmax():
        check_error(
            LIBINFINIOP.infiniopLogSoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_logsoftmax()

    if sync is not None:
        sync()

    # Use tolerance based on input dtype for numerical stability
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    # Always print debug info for failed cases
    actual = y.actual_tensor()
    max_diff = torch.max(torch.abs(actual - ans))
    is_close = torch.allclose(actual, ans, atol=atol, rtol=rtol)

    if DEBUG or not is_close:
        print(f"\n=== Debug Info ===")
        print(f"Shape: {shape}, Stride: {x_stride}, Dtype: {dtype}")
        print(f"Input tensor: {x.torch_tensor()}")
        print(f"Expected output: {ans}")
        print(f"Actual output: {actual}")
        print(f"Max diff: {max_diff}")
        print(f"Tolerance: atol={atol}, rtol={rtol}")
        print(f"Is close: {is_close}")
        print(f"First few values - Actual: {actual.flatten()[:5]}")
        print(f"First few values - Expected: {ans.flatten()[:5]}")
        if DEBUG:
            debug(actual, ans, atol=atol, rtol=rtol)

    assert is_close

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: logsoftmax(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_logsoftmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyLogSoftmaxDescriptor(descriptor))


def test_mixed_precision(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    x_dtype=InfiniDtype.F16,
    y_dtype=InfiniDtype.F32,
    sync=None,
):
    print(
        f"Testing LogSoftmax (Mixed) on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} x_dtype:{InfiniDtypeNames[x_dtype]} y_dtype:{InfiniDtypeNames[y_dtype]} inplace:{inplace}"
    )

    x = TestTensor(shape, x_stride, x_dtype, device)
    ans = logsoftmax(x.actual_tensor())

    # Convert answer to target dtype for comparison
    if y_dtype == InfiniDtype.F16:
        ans = ans.to(torch.float16)
    elif y_dtype == InfiniDtype.BF16:
        ans = ans.to(torch.bfloat16)
    elif y_dtype == InfiniDtype.F32:
        ans = ans.to(torch.float32)

    if inplace == Inplace.INPLACE_X:
        # For inplace operations, input and output must have the same dtype
        if x_dtype != y_dtype:
            print(
                f"Skipping inplace test: x_dtype ({InfiniDtypeNames[x_dtype]}) != y_dtype ({InfiniDtypeNames[y_dtype]})"
            )
            return
        y = x
    else:
        y = TestTensor(shape, y_stride, y_dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateLogSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLogSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_logsoftmax():
        check_error(
            LIBINFINIOP.infiniopLogSoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_logsoftmax()

    if sync is not None:
        sync()

    # Use tolerance based on output dtype for mixed precision cases
    atol, rtol = get_tolerance(_TOLERANCE_MAP, y_dtype)

    # Ensure both tensors have the same dtype for comparison
    y_tensor = y.actual_tensor()
    if y_tensor.dtype != ans.dtype:
        y_tensor = y_tensor.to(ans.dtype)

    if DEBUG:
        debug(y_tensor, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y_tensor, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: logsoftmax(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_logsoftmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyLogSoftmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        # Test standard cases (fp32 output)
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

        # Test mixed precision cases
        from libinfiniop import create_handle, destroy_handle, get_sync_func

        handle = create_handle()
        sync = get_sync_func(device)
        try:
            for x_dtype, y_dtype in _MIXED_PRECISION_CASES:
                for shape, x_stride, y_stride, inplace in _TEST_CASES[
                    :5
                ]:  # Test subset for mixed precision
                    test_mixed_precision(
                        handle,
                        device,
                        shape,
                        x_stride,
                        y_stride,
                        inplace,
                        x_dtype,
                        y_dtype,
                        sync,
                    )
        finally:
            destroy_handle(handle)

    print("\033[92mTest passed!\033[0m")
