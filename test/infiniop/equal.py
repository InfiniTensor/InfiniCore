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
    # shape, a_stride, b_stride
    ((13, 4), None, None),
    ((13, 4), (13, 1), (13, 1)),
    ((13, 4, 4), (16, 4, 1), (16, 4, 1),),
    ((16, 5632), None, None),
]

class Identical(Enum):
    EQUAL = auto()
    NOT_EQUAL = auto()


_IDENTICAL = [
    Identical.EQUAL, # -> result=true
    Identical.NOT_EQUAL, # -> result=false
]

_TEST_CASES = [
    test_case + (identical_item,)
    for test_case in _TEST_CASES_
    for identical_item in _IDENTICAL
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.I32, InfiniDtype.I64]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 0},
    InfiniDtype.F32: {"atol": 0, "rtol": 0},
    InfiniDtype.BF16: {"atol": 0, "rtol": 0},
    InfiniDtype.I32: {"atol": 0, "rtol": 0},
    InfiniDtype.I64: {"atol": 0, "rtol": 0},
}


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_equal(c, a, b):
    return torch.tensor(torch.equal(input=a, other=b), dtype=torch.bool)
    

def test(
    handle,
    device,
    input_shape,
    a_strides,
    b_strides,
    identical,
    dtype,
    sync=None,
):
    torch_dtype = {
        InfiniDtype.F16: torch.half,
        InfiniDtype.F32: torch.float,
        InfiniDtype.BF16: torch.bfloat16,
        InfiniDtype.I32: torch.int32,
        InfiniDtype.I64: torch.int64
    }[dtype]

    print(
        f"Testing equal on {InfiniDeviceNames[device]} with input_shape:{input_shape},"
        f"a_stride:{a_strides} b_stride:{b_strides} identical:{identical},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )
    torch_c = torch.tensor([False], dtype=torch.bool)
    c = TestTensor(
        [1],
        torch_c.stride(),
        InfiniDtype.BOOL,
        device,
        "manual",
        set_tensor=torch_c
    )

    torch_a = (torch.rand(input_shape) * 100 - 50).type(torch_dtype)
    if a_strides is not None:
        torch_a.as_strided_(input_shape, a_strides)
    a = TestTensor(
        input_shape,
        torch_a.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_a        
    )
    if identical == Identical.EQUAL:
        torch_b = torch_a.clone()
    else:
        torch_b = (torch.rand(input_shape) * 100 - 50).type(torch_dtype)
    if b_strides is not None:
        torch_b.as_strided_(input_shape, b_strides)   
 
    b = TestTensor(
        input_shape,
        torch_b.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_b
    )


    c._torch_tensor = torch_equal(c.torch_tensor(), a.torch_tensor(), b.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateEqualDescriptor(
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
        LIBINFINIOP.infiniopGetEqualWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, c.device)

    def lib_equal():
        check_error(
            LIBINFINIOP.infiniopEqual(
                descriptor,
                workspace.data(),
                workspace.size(),
                c.data(),
                a.data(),
                b.data(),                
                None,
            )
        )

    lib_equal()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(c.actual_tensor().to(torch.uint8), c.torch_tensor().to(torch.uint8), atol=atol, rtol=rtol)
    assert torch.allclose(c.actual_tensor(), c.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_equal(
            c.torch_tensor(), a.torch_tensor(), b.torch_tensor()
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_equal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyEqualDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my equal passed!\033[0m")
