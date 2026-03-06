import ctypes
from ctypes import c_uint64

import torch

from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES = [
    # input_shape, x_stride, y_stride, kernel_size, stride, padding
    ((2, 3, 16), None, None, 3, None, 0),
    ((1, 4, 15), (60, 15, 1), (60, 15, 1), 5, 1, 2),
    ((2, 1, 32), None, (32, 16, 1), 2, 2, 0),
    ((3, 2, 7), (14, 7, 1), (9, 3, 1), 3, None, 1),
    ((4, 6, 31), None, None, 4, 2, 1),
    ((2, 8, 9), (72, 9, 1), (56, 7, 1), 3, 1, 0),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def _effective_stride(stride, kernel_size):
    if stride in (None, 0):
        return kernel_size
    return stride


def _compute_output_shape(input_shape, kernel_size, stride, padding):
    stride = _effective_stride(stride, kernel_size)
    width = input_shape[2]
    out_width = (width + 2 * padding - kernel_size) // stride + 1
    return (input_shape[0], input_shape[1], out_width)


def avg_pool1d_ref(x, kernel_size, stride, padding):
    stride = _effective_stride(stride, kernel_size)
    out = torch.nn.functional.avg_pool1d(
        x.to(torch.float32), kernel_size=kernel_size, stride=stride, padding=padding
    )
    return out.to(x.dtype)


def test(
    handle,
    device,
    input_shape,
    x_stride,
    y_stride,
    kernel_size,
    stride,
    padding,
    dtype=InfiniDtype.F16,
    sync=None,
):
    stride_value = _effective_stride(stride, kernel_size)
    out_shape = _compute_output_shape(
        input_shape, kernel_size, stride_value, padding
    )
    print(
        f"Testing AvgPool1d on {InfiniDeviceNames[device]} with input_shape:{input_shape}, "
        f"output_shape:{out_shape}, kernel_size:{kernel_size}, stride:{stride_value}, "
        f"padding:{padding}, dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(input_shape, x_stride, dtype, device)
    y = TestTensor(out_shape, y_stride, dtype, device, mode="zeros")

    ans = avg_pool1d_ref(x.torch_tensor(), kernel_size, stride_value, padding)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAvgPool1dDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            kernel_size,
            stride_value,
            padding,
        )
    )

    # Invalidate descriptors in tensors after creation to make sure kernels read from arguments
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAvgPool1dWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_avg_pool1d():
        check_error(
            LIBINFINIOP.infiniopAvgPool1d(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_avg_pool1d()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation(
            "PyTorch",
            lambda: avg_pool1d_ref(x.torch_tensor(), kernel_size, stride_value, padding),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_avg_pool1d(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyAvgPool1dDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")

