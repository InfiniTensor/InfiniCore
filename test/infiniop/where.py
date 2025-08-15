#!/usr/bin/env python3

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
    torch_device_map,
)
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # shape, condition_stride, a_stride, b_stride, c_stride
    ((4,), None, None, None, None),
    ((2, 3), None, None, None, None),
    ((2, 3, 4), None, None, None, None),
    ((13, 4), None, None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1), (10, 1)),
    ((13, 4, 4), None, None, None, None),
    ((16, 32), None, None, None, None),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_A = auto()
    INPLACE_B = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_A,
    Inplace.INPLACE_B,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.F64,
    InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64,
    InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
    InfiniDtype.BF16
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},
    # Integer types use exact comparison
    InfiniDtype.I8: {"atol": 0, "rtol": 0},
    InfiniDtype.I16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
    InfiniDtype.U8: {"atol": 0, "rtol": 0},
    InfiniDtype.U16: {"atol": 0, "rtol": 0},
    InfiniDtype.U32: {"atol": 0, "rtol": 0},
    InfiniDtype.U64: {"atol": 0, "rtol": 0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def where(output, condition, a, b):
    """Reference implementation using torch.where"""
    torch.where(condition, a, b, out=output)


def test(
    handle,
    device,
    shape,
    condition_stride=None,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F32,
    sync=None,
):
    # Create condition tensor (always bool) - use manual creation for bool type
    condition_data = torch.randint(0, 2, shape, dtype=torch.bool, device=torch_device_map[device])
    condition = TestTensor.from_torch(condition_data, InfiniDtype.BOOL, device)
    
    # Create input tensors with specified dtype
    if dtype in [InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64,
                 InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64]:
        # For integer types, use a smaller range to avoid overflow
        a = TestTensor(shape, a_stride, dtype, device, mode="random", scale=10, bias=0)
        b = TestTensor(shape, b_stride, dtype, device, mode="random", scale=10, bias=0)
    else:
        # For floating point types
        a = TestTensor(shape, a_stride, dtype, device, mode="random")
        b = TestTensor(shape, b_stride, dtype, device, mode="random")
    
    # Handle inplace operations
    if inplace == Inplace.INPLACE_A:
        if a_stride != c_stride:
            return
        c = a
    elif inplace == Inplace.INPLACE_B:
        if b_stride != c_stride:
            return
        c = b
    else:
        c = TestTensor(shape, c_stride, dtype, device, mode="zeros")

    if c.is_broadcast():
        return

    print(
        f"Testing Where on {InfiniDeviceNames[device]} with shape:{shape} "
        f"condition_stride:{condition_stride} a_stride:{a_stride} b_stride:{b_stride} "
        f"c_stride:{c_stride} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    # Compute reference result
    where(c.torch_tensor(), condition.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateWhereDescriptor(
            handle,
            ctypes.byref(descriptor),
            condition.descriptor,
            a.descriptor,
            b.descriptor,
            c.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [condition, a, b, c]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetWhereWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_where():
        check_error(
            LIBINFINIOP.infiniopWhere(
                descriptor,
                workspace.data(),
                workspace.size(),
                condition.data(),
                a.data(),
                b.data(),
                c.data(),
                None,
            )
        )

    lib_where()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: where(c.torch_tensor(), condition.torch_tensor(), a.torch_tensor(), b.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_where(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyWhereDescriptor(descriptor))


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