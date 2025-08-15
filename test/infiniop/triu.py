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
    ((5, 6), 0),
    ((4, 5), -1),
    ((5, 2), 3),
    ((89, 80), -20),
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
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16, InfiniDtype.I64, InfiniDtype.I32]

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


def torch_triu(output, input, diagonal):
    torch.triu(input, diagonal, out=output)
    

def test(
    handle,
    device,
    input_shape, diagonal,
    inplace,
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
        f"Testing triu on {InfiniDeviceNames[device]} with shape:{input_shape}, diagonal:{diagonal}, "
        f"inplace:{inplace},"
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_input = (torch.rand(size=input_shape) * 100 - 50).type(torch_dtype)
    input = TestTensor(
        input_shape,
        torch_input.stride(),
        dtype,
        device,
        "manual",
        set_tensor=torch_input
    )
    if inplace == Inplace.INPLACE:
        output = input
    else:
        output = TestTensor(
            input_shape,
            None,
            dtype,
            device,
            "zeros"
        )

    torch_triu(output.torch_tensor(), input.torch_tensor(), diagonal)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateTriuDescriptor(
            handle,
            ctypes.byref(descriptor),
			output.descriptor,
			input.descriptor,
			diagonal,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [output, input]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetTriuWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, output.device)

    def lib_triu():
        check_error(
            LIBINFINIOP.infiniopTriu(
                descriptor,
                workspace.data(),
                workspace.size(),
				output.data(),
				input.data(),                
                None,
            )
        )

    lib_triu()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor(), output.torch_tensor(), atol=atol, rtol=rtol)         

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_triu(
            output, input, diagonal
        ), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_triu(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyTriuDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest my triu passed!\033[0m")
