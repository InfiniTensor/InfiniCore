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
from typing import List, Tuple
import math
from torch.nn import functional as F

# constant for control whether profile the pytorch and lib functions
# NOTE: need to manually add synchronization function to the lib function,
#       e.g., cudaDeviceSynchronize() for CUDA
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
_TEST_CASES = [
    # x_shape, x_stride, w_shape, w_stride, pads, strides, dilations
    (
        (2, 16, 8),     # [B, L, Cin]
        (128, 8, 1),    # stride for [B, L, Cin]
        (24, 8, 3),     # [2*Cout, Cin, K] for gated conv1d
        (24, 3, 1),     # stride for weight
        (1,),           # padding
        (1,),           # stride
        (1,),           # dilation
    ),
    (
        (1, 32, 16),    # [B, L, Cin]
        (512, 16, 1),   # stride for [B, L, Cin]
        (64, 16, 5),    # [2*Cout, Cin, K] for gated conv1d
        (80, 5, 1),     # stride for weight
        (2,),           # padding
        (1,),           # stride
        (1,),           # dilation
    ),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def conv1d_gated_reference(x, w, stride, padding, dilation, y_tensor, bias=None):
    """Reference implementation using PyTorch conv1d + gated activation"""
    # x: [B, L, Cin] -> [B, Cin, L]
    xb = x.permute(0, 2, 1)
    # Apply conv1d and slice to original length
    y_full = F.conv1d(xb, w, bias=bias, stride=stride, padding=padding, dilation=dilation)
    y_sliced = y_full[:, :, :x.shape[1]]  # slice to original sequence length
    
    # Split into two halves for gating
    Cout2 = y_sliced.shape[1]
    assert Cout2 % 2 == 0, "weight out_channels must be 2*Cout"
    Cout = Cout2 // 2
    A, B = y_sliced[:, :Cout, :], y_sliced[:, Cout:, :]
    
    # Apply gated activation: silu(A) * B
    z = F.silu(A) * B
    # Convert back to [B, L, Cout]
    z = z.permute(0, 2, 1)
    y_tensor.copy_(z)


# infer the shape of the output given the inputs for conv1d
def inferShapeStride(
    x_shape: List[int],
    w_shape: List[int],
    pads: List[int],
    strides: List[int],
    dilations: List[int],
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    # x_shape: [B, L, Cin], w_shape: [2*Cout, Cin, K]
    # output: [B, L, Cout] where Cout = w_shape[0] // 2
    B, L, Cin = x_shape
    Cout2, _, K = w_shape
    assert Cout2 % 2 == 0, "weight out_channels must be even for gated conv1d"
    Cout = Cout2 // 2
    
    # For conv1d, we maintain the sequence length L
    output_shape = (B, L, Cout)
    output_strides = [1]
    for s in reversed(output_shape[1:]):
        output_strides.insert(0, output_strides[0] * s)
    output_strides = tuple(output_strides)
    return output_shape, output_strides


# convert a python tuple to a ctype void pointer
def tuple_to_void_p(py_tuple: Tuple):
    array = ctypes.c_int64 * len(py_tuple)
    data_array = array(*py_tuple)
    return ctypes.cast(data_array, ctypes.c_void_p)


def test(
    handle,
    device,
    x_shape,
    x_stride,
    w_shape,
    w_stride,
    pads,
    strides,
    dilations,
    tensor_dtype=InfiniDtype.F16,
    sync=None,
):
    assert len(pads) == len(strides) == len(dilations)
    x = TestTensor(x_shape, x_stride, dt=tensor_dtype, device=device, scale=0.01)
    w = TestTensor(w_shape, w_stride, dt=tensor_dtype, device=device, scale=0.01)
    y_shape, y_stride = inferShapeStride(x_shape, w_shape, pads, strides, dilations)
    y = TestTensor(y_shape, y_stride, dt=tensor_dtype, device=device)

    b = (
        TestTensor((w.shape[0],), (1,), dt=tensor_dtype, device=device, scale=0.01)
        if w.shape[0] > 1
        else None
    )
    print(
        f"Testing Conv1d on {InfiniDeviceNames[device]} with x_shape: {x_shape}, w_shape: {w_shape}, b_shape: {w_shape[0] if b else None}, pads: {pads}, strides: {strides}, dilations: {dilations}, x_stride: {x_stride} dtype:{InfiniDtypeNames[tensor_dtype]}"
    )
    conv1d_gated_reference(
        x.torch_tensor(),
        w.torch_tensor(),
        strides,
        pads,
        dilations,
        y.torch_tensor(),
        b.torch_tensor() if b is not None else None,
    )

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateConv1dDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            w.descriptor,
            b.descriptor if b is not None else None,
            tuple_to_void_p(pads),
            tuple_to_void_p(strides),
            tuple_to_void_p(dilations),
            len(pads),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y, w, b]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = ctypes.c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetConv1dWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_conv1d():
        check_error(
            LIBINFINIOP.infiniopConv1d(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                w.data(),
                b.data() if b is not None else None,
                None,
            )
        )

    lib_conv1d()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, tensor_dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: conv1d_gated_reference(x.torch_tensor(), w.torch_tensor(), strides, pads, dilations, y.torch_tensor(), b.torch_tensor() if b is not None else None), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_conv1d(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyConv1dDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")