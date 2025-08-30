import torch
import ctypes
from ctypes import c_uint64, c_bool
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
import math
from torch.nn import functional as F
from typing import Tuple

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

_TEST_CASES = [
    # ============ 1D Average Pooling Tests (converted to MaxPool format) ============
    # Basic cases
    ((4, 8, 128), None, (3,), (1,), (0,), False),  # kernel=3, stride=1, pad=0
    ((2, 16, 256), None, (5,), (2,), (2,), False),  # kernel=5, stride=2, pad=2
    ((8, 4, 64), None, (7,), (3,), (1,), False),  # kernel=7, stride=3, pad=1
    # ceil_mode variations
    ((1, 3, 99), None, (4,), (3,), (1,), True),  # kernel=4, stride=3, pad=1
    ((3, 2, 77), None, (6,), (4,), (0,), True),  # kernel=6, stride=4, pad=0
    # ============ 2D Average Pooling Tests ============
    # Basic cases with square kernels
    ((2, 3, 64, 64), None, (3, 3), (1, 1), (1, 1), False),
    ((4, 16, 128, 128), None, (5, 5), (2, 2), (2, 2), False),
    ((1, 8, 96, 96), None, (7, 7), (3, 3), (0, 0), False),
    # Rectangular kernels
    ((2, 4, 80, 120), None, (3, 5), (1, 2), (1, 2), False),
    ((1, 6, 72, 48), None, (7, 3), (2, 1), (3, 1), False),
    ((3, 2, 56, 84), None, (2, 4), (2, 3), (0, 2), False),
    # ceil_mode variations
    ((1, 1, 33, 33), None, (4, 4), (3, 3), (1, 1), True),
    ((2, 5, 77, 89), None, (5, 3), (4, 2), (2, 1), True),
    # ============ 3D Average Pooling Tests ============
    # Basic cubic kernels
    ((1, 2, 32, 32, 32), None, (3, 3, 3), (1, 1, 1), (1, 1, 1), False),
    ((2, 4, 48, 48, 48), None, (5, 5, 5), (2, 2, 2), (2, 2, 2), False),
    ((1, 1, 64, 64, 64), None, (7, 7, 7), (3, 3, 3), (0, 0, 0), False),
    # Non-cubic kernels
    ((1, 3, 24, 36, 48), None, (2, 3, 4), (1, 2, 2), (0, 1, 2), False),
    ((2, 2, 40, 32, 56), None, (5, 3, 7), (2, 1, 3), (2, 1, 3), False),
    ((1, 1, 28, 44, 36), None, (3, 5, 2), (2, 3, 1), (1, 2, 1), False),
    # ceil_mode variations
    ((1, 1, 27, 27, 27), None, (4, 4, 4), (3, 3, 3), (1, 1, 1), True),
    ((2, 2, 33, 45, 39), None, (5, 3, 4), (3, 2, 3), (2, 1, 1), True),
]

_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.F16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}


def averagepool_backward(
    input_tensor,
    grad_output_tensor,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    grad_input_tensor,
):
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    ndim = len(input_tensor.shape) - 2
    if ndim == 1:
        output = F.avg_pool1d(
            input_tensor, kernel_size[0], stride[0], padding[0], ceil_mode=ceil_mode
        )
    elif ndim == 2:
        output = F.avg_pool2d(
            input_tensor, kernel_size, stride, padding, ceil_mode=ceil_mode
        )
    elif ndim == 3:
        output = F.avg_pool3d(
            input_tensor, kernel_size, stride, padding, ceil_mode=ceil_mode
        )
    else:
        raise ValueError("Unsupported dimension")
    output.backward(grad_output_tensor)
    grad_input_tensor.copy_(input_tensor.grad)


def infer_output_shape(input_shape, kernel_size, stride, padding, ceil_mode):
    def calc_output_size(input_size, k, s, p, ceil_mode):
        if ceil_mode:
            return math.ceil((input_size + 2 * p - k) / s + 1)
        else:
            return math.floor((input_size + 2 * p - k) / s + 1)

    return (input_shape[0], input_shape[1]) + tuple(
        calc_output_size(
            input_shape[i + 2], kernel_size[i], stride[i], padding[i], ceil_mode
        )
        for i in range(len(kernel_size))
    )


def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_uint64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    handle,
    device,
    input_shape,
    input_stride,
    kernel_size,
    stride,
    padding,
    ceil_mode,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    input_tensor = TestTensor(
        input_shape, input_stride, dt=tensor_dtype, device=device, scale=1.0
    )
    output_shape = infer_output_shape(
        input_shape, kernel_size, stride, padding, ceil_mode
    )
    grad_output_tensor = TestTensor(
        output_shape, None, dt=tensor_dtype, device=device, scale=1.0
    )
    grad_input_tensor = TestTensor(
        input_shape, input_stride, dt=tensor_dtype, device=device
    )

    print(
        f"Testing AvgPoolBackward on {InfiniDeviceNames[device]} with input: {input_shape}, kernel: {kernel_size}, stride: {stride}, pad: {padding}, ceil_mode: {ceil_mode}"
    )
    print(
        f"Input Tensor: {input_tensor.shape}, Grad Output Tensor: {grad_output_tensor.shape}, Grad Input Tensor: {grad_input_tensor.shape}"
    )

    averagepool_backward(
        input_tensor.torch_tensor(),
        grad_output_tensor.torch_tensor(),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        grad_input_tensor.torch_tensor(),
    )

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAvgPoolBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            grad_input_tensor.descriptor,
            grad_output_tensor.descriptor,
            input_tensor.descriptor,
            tuple_to_void_p(kernel_size),
            tuple_to_void_p(stride),
            tuple_to_void_p(padding),
            c_bool(ceil_mode),
        )
    )

    for tensor in [input_tensor, grad_output_tensor, grad_input_tensor]:
        if tensor:
            tensor.destroy_desc()

    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAvgPoolBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_averagepool_backward():
        check_error(
            LIBINFINIOP.infiniopAvgPoolBackward(
                descriptor,
                workspace.data(),
                workspace_size.value,
                grad_input_tensor.data(),
                grad_output_tensor.data(),
                input_tensor.data(),
                None,
            )
        )

    lib_averagepool_backward()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(
            grad_input_tensor.actual_tensor(),
            grad_input_tensor.torch_tensor(),
            atol,
            rtol,
        )
    assert torch.allclose(
        grad_input_tensor.actual_tensor(),
        grad_input_tensor.torch_tensor(),
        atol=atol,
        rtol=rtol,
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: averagepool_backward(
                input_tensor.torch_tensor(),
                grad_output_tensor.torch_tensor(),
                kernel_size,
                stride,
                padding,
                ceil_mode,
                grad_input_tensor.torch_tensor(),
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "lib", lib_averagepool_backward, device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyAvgPoolBackwardDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    print("\033[92mAvgPoolBackward Test Passed!\033[0m")
