from ctypes import POINTER, Structure, c_int32, c_uint64, c_void_p
import ctypes
import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from libinfiniop import (
    infiniopHandle_t,
    infiniopTensorDescriptor_t,
    open_lib,
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    create_workspace,
)

import torch
import math
import ctypes
from torch.nn import functional as F
from typing import List, Tuple

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
_TEST_CASES = [
    # x_shape, w_shape, pads, strides, dilations, x_strides
    (
        (32, 3, 4),
        (32, 3, 5),
        (1,),
        (1,),
        (1,),
        None,
    ),
    (
        (1, 3, 4, 4),
        (2, 3, 3, 3),
        (1, 1),
        (1, 2),
        (2, 1),
        None,
    ),
    (
        (32, 3, 128, 128),
        (64, 3, 5, 5),
        (2, 2),
        (2, 2),
        (1, 1),
        None,
    ),
    (
        (1, 1, 4, 4, 4),
        (1, 1, 5, 5, 5),
        (1, 1, 1),
        (1, 1, 1),
        (1, 1, 1),
        None,
    ),
    (
        (32, 3, 32, 32, 32),
        (64, 3, 5, 5, 5),
        (3, 2, 2),
        (4, 3, 3),
        (2, 2, 1),
        None,
    ),
]


class ConvDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopConvDescriptor_t = POINTER(ConvDescriptor)

# Data types used for testing
_TENSOR_DTYPES = [torch.float16, torch.float32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


class ConvDescriptor(Structure):
    _fields_ = [("device", c_int32)]


infiniopConvDescriptor_t = POINTER(ConvDescriptor)


def conv(x, w, stride, padding, dilation, bias=None):
    match len(x.shape) - 2:
        case 1:
            return F.conv1d(
                x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
            )
        case 2:
            return F.conv2d(
                x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
            )
        case 3:
            return F.conv3d(
                x, w, bias=bias, stride=stride, padding=padding, dilation=dilation
            )
        case _:
            print("Error: Pytorch -> Unsupported tensor dimension")
            return None


# infer the shape of the output given the inputs for a N-ary convolution
def inferShape(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[int, ...]:
    assert (
        len(x_shape)
        == len(w_shape)
        == len(pads) + 2
        == len(dilations) + 2
        == len(strides) + 2
    ), "x and w should have the same length; pads, strides, and dilatinos should have the same length; the length of pads should be that of x - 2"
    output_dims = [
        math.floor(
            (x_shape[i + 2] + 2 * pads[i] - dilations[i] * (w_shape[i + 2] - 1) - 1)
            / strides[i]
            + 1
        )
        for i in range(len(pads))
    ]
    return (x_shape[0], w_shape[0]) + tuple(output_dims)


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    lib,
    handle,
    torch_device,
    x_shape,
    w_shape,
    pads,
    strides,
    dilations,
    tensor_stride=None,
    tensor_dtype=torch.float16,
    sync=None,
):
    assert len(pads) == len(strides) == len(dilations)
    print(
        f"Testing Conv on {torch_device} with x_shape: {x_shape}, w_shape: {w_shape}, b_shape: {w_shape[0]}, pads: {pads}, strides: {strides}, dilations: {dilations}, x_stride: {tensor_stride} dtype:{tensor_dtype}"
    )
    x = torch.rand(x_shape, dtype=tensor_dtype).to(torch_device)
    w = torch.rand(w_shape, dtype=tensor_dtype).to(torch_device)
    y = torch.zeros(
        inferShape(x.shape, w.shape, pads, strides, dilations), dtype=tensor_dtype
    ).to(torch_device)
    bias = (
        torch.rand(w.shape[0], dtype=tensor_dtype).to(torch_device)
        if w.shape[0] > 1
        else None
    )

    for i in range(NUM_PRERUN if PROFILE else 1):
        ans = conv(x, w, strides, pads, dilations, bias)
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            _ = conv(x, w, strides, pads, dilations, bias)
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"pytorch time: {elapsed :6f}")

    x_tensor = to_tensor(x, lib)
    w_tensor = to_tensor(w, lib)
    y_tensor = to_tensor(y, lib)
    b_tensor = to_tensor(bias, lib) if bias is not None else None

    if sync is not None:
        sync()

    descriptor = infiniopConvDescriptor_t()
    check_error(
        lib.infiniopCreateConvDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w_tensor.descriptor,
            b_tensor.descriptor if b_tensor is not None else None,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            len(pads),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor, w_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetConvWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, y.device)

    def lib_conv():
        check_error(
            lib.infiniopConv(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size,
                y_tensor.data,
                x_tensor.data,
                w_tensor.data,
                b_tensor.data if b_tensor is not None else None,
                None,
            )
        )

    for i in range(NUM_PRERUN if PROFILE else 1):
        lib_conv()
    if PROFILE:
        start_time = time.time()
        for i in range(NUM_ITERATIONS):
            lib_conv()
        elapsed = (time.time() - start_time) / NUM_ITERATIONS
        print(f"    lib time: {elapsed :6f}")
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)
    check_error(lib.infiniopDestroyConvDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    lib = open_lib()
    lib.infiniopCreateConvDescriptor.restype = c_int32
    lib.infiniopCreateConvDescriptor.argtypes = [
        infiniopHandle_t,
        POINTER(infiniopConvDescriptor_t),
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        infiniopTensorDescriptor_t,
        c_void_p,
        c_void_p,
        c_void_p,
        c_uint64,
    ]
    lib.infiniopConv.restype = c_int32
    lib.infiniopConv.argtypes = [
        infiniopConvDescriptor_t,
        c_void_p,
        c_uint64,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
        c_void_p,
    ]
    lib.infiniopDestroyConvDescriptor.restype = c_int32
    lib.infiniopDestroyConvDescriptor.argtypes = [
        infiniopConvDescriptor_t,
    ]
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
