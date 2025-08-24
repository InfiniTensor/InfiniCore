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
from enum import Enum, auto
from typing import List, Tuple
import math
from torch.nn import functional as F

# Configuration for profiling
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

# Test cases: (input_shape, input_stride, kernel_size, stride, padding, ceil_mode)
_TEST_CASES = [
    # 1D max pooling cases
    ((1, 3, 8), None, (2,), (2,), (0,), False),
    ((2, 4, 16), None, (3,), (2,), (1,), False),
    # 2D max pooling cases
    ((1, 1, 4, 4), None, (2, 2), (2, 2), (0, 0), False),
    ((2, 3, 8, 8), None, (3, 3), (2, 2), (1, 1), False),
    ((1, 64, 32, 32), None, (2, 2), (2, 2), (0, 0), False),
    ((4, 128, 16, 16), None, (3, 3), (1, 1), (1, 1), False),
    # 3D max pooling cases
    ((1, 1, 4, 4, 4), None, (2, 2, 2), (2, 2, 2), (0, 0, 0), False),
    ((2, 2, 8, 8, 8), None, (2, 3, 3), (2, 2, 2), (0, 1, 1), False),
    # Cases with ceil_mode=True
    ((1, 1, 7, 7), None, (3, 3), (2, 2), (1, 1), True),
    ((1, 2, 5), None, (3,), (2,), (0,), True),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}


def max_pool(input_tensor, kernel_size, stride, padding, ceil_mode, output_tensor):
    """
    Perform max pooling using PyTorch as reference
    """
    ndim = len(input_tensor.shape) - 2  # Spatial dimensions

    if ndim == 1:
        result = F.max_pool1d(
            input_tensor,
            kernel_size=kernel_size[0],
            stride=stride[0],
            padding=padding[0],
            ceil_mode=ceil_mode,
        )
    elif ndim == 2:
        result = F.max_pool2d(
            input_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )
    elif ndim == 3:
        result = F.max_pool3d(
            input_tensor,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
        )
    else:
        raise ValueError(f"Unsupported spatial dimensions: {ndim}")

    output_tensor.copy_(result)


def infer_output_shape(input_shape, kernel_size, stride, padding, ceil_mode):
    """
    Infer the output shape for max pooling operation
    """

    def calc_output_size(input_size, kernel_size, stride, padding, ceil_mode):
        if ceil_mode:
            return math.ceil((input_size + 2 * padding - kernel_size) / stride + 1)
        else:
            return math.floor((input_size + 2 * padding - kernel_size) / stride + 1)

    batch_size = input_shape[0]
    channels = input_shape[1]
    spatial_dims = input_shape[2:]

    output_spatial = []
    for i, dim_size in enumerate(spatial_dims):
        output_size = calc_output_size(
            dim_size, kernel_size[i], stride[i], padding[i], ceil_mode
        )
        output_spatial.append(output_size)

    return (batch_size, channels) + tuple(output_spatial)


def tuple_to_void_p(py_tuple: Tuple):
    """Convert a python tuple to a ctype void pointer"""
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
    # Create input tensor
    input_tensor = TestTensor(
        input_shape, input_stride, dt=tensor_dtype, device=device, scale=1.0
    )

    # Infer output shape
    output_shape = infer_output_shape(
        input_shape, kernel_size, stride, padding, ceil_mode
    )
    output_tensor = TestTensor(output_shape, None, dt=tensor_dtype, device=device)

    print(
        f"Testing MaxPool on {InfiniDeviceNames[device]} with "
        f"input_shape: {input_shape}, kernel_size: {kernel_size}, "
        f"stride: {stride}, padding: {padding}, ceil_mode: {ceil_mode}, "
        f"dtype: {InfiniDtypeNames[tensor_dtype]}"
    )

    # Compute reference result using PyTorch
    max_pool(
        input_tensor.torch_tensor(),
        kernel_size,
        stride,
        padding,
        ceil_mode,
        output_tensor.torch_tensor(),
    )

    if sync is not None:
        sync()

    # Create descriptor for our max pool operator
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateMaxPoolDescriptor(
            handle,
            ctypes.byref(descriptor),
            output_tensor.descriptor,
            input_tensor.descriptor,
            tuple_to_void_p(kernel_size),
            tuple_to_void_p(stride),
            tuple_to_void_p(padding),
            c_bool(ceil_mode),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [input_tensor, output_tensor]:
        if tensor is not None:
            tensor.destroy_desc()

    # Get workspace size
    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetMaxPoolWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output_tensor.device)

    def lib_max_pool():
        check_error(
            LIBINFINIOP.infiniopMaxPool(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output_tensor.data(),
                input_tensor.data(),
                None,
            )
        )

    # Execute the operation
    lib_max_pool()

    # Check results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(
            output_tensor.actual_tensor(),
            output_tensor.torch_tensor(),
            atol=atol,
            rtol=rtol,
        )

    assert torch.allclose(
        output_tensor.actual_tensor(),
        output_tensor.torch_tensor(),
        atol=atol,
        rtol=rtol,
    ), f"Results don't match for input_shape {input_shape}, kernel_size {kernel_size}"

    # Profiling workflow
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: max_pool(
                input_tensor.torch_tensor(),
                kernel_size,
                stride,
                padding,
                ceil_mode,
                output_tensor.torch_tensor(),
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_max_pool(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyMaxPoolDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        if InfiniDeviceNames[device] == "Iluvatar":
            _TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16]
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
