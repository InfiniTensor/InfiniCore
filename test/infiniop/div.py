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

_TEST_CASES_ = [
    # shape, a_stride, b_stride, c_stride
    ((13, 4), None, None, None),
    ((13, 4), (1, 13), (1, 13), (1, 13)),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (4, 1, 52), (16, 4, 1), (1, 13, 52)),
    ((16, 5632), None, None, None),
    ((16, 5632), (1, 16), (1, 16), (1, 16)),
    ((4, 4, 5632), None, None, None),

]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]


# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_div(c, a, b):
    torch.div(a, b, out=c)
    

def test(
    handle,
    device,
    input_shape,
    a_stride,
    b_stride,
    c_stride,    
    inplace,
    dtype,
    sync=None,
):
    print(
        f"Testing div on {InfiniDeviceNames[device]} with input_shape:{input_shape}"
        f" a_stride:{a_stride} b_stride:{b_stride} c_stride:{c_stride} inplace:{inplace},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    a = TestTensor(
        input_shape,
        a_stride,
        dtype,
        device,
    )
    
    # avoid zero
    torch_b = torch.rand(input_shape).type(a.torch_tensor().dtype) - 0.5
    torch_b = torch.where(torch_b == 0., 1, torch_b)
    b = TestTensor(
        input_shape,
        torch_b.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_b
    )
    if inplace == Inplace.INPLACE:
        if a_stride != c_stride:
            return
        c = a
    else:
        c = TestTensor(
            input_shape,
            c_stride,
            dtype,
            device,
        )    

    torch_div(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDivDescriptor(
            handle,
            ctypes.byref(descriptor),
            c.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [c, a, b]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetDivWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_div():
        check_error(
            LIBINFINIOP.infiniopDiv(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),                
                None,
            )
        )

    lib_div()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_div(
            c.torch_tensor(), a.torch_tensor(), b.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_div(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyDivDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my div passed!\033[0m")