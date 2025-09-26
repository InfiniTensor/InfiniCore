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
    ((13, 4), None, None),
    ((13, 4), (10, 1), (10, 1)),
    ((13, 4), (0, 1), (0, 1)),
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1)),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    Inplace.OUT_OF_PLACE,
    Inplace.INPLACE_X,
]

# Form the test cases by appending each element of _INPLACE to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [
                  InfiniDtype.BYTE,  # 1
                  InfiniDtype.BOOL,  # 2
                  InfiniDtype.I8,  # 3
                  InfiniDtype.I16,  # 4
                  InfiniDtype.I32,  # 5
                  InfiniDtype.I64,  # 6
                  InfiniDtype.U8,  # 7
                #   InfiniDtype.U16,  # 8
                #   InfiniDtype.U32,  # 9
                #   InfiniDtype.U64,  # 10
                #   InfiniDtype.F8,  # 11
                  InfiniDtype.F16,  # 12
                  InfiniDtype.F32,  # 13
                  InfiniDtype.F64,  # 14
                  InfiniDtype.BF16,  # 19
                  ]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.BYTE: {"atol": 1e-3, "rtol": 1e-3},  # 1
    InfiniDtype.BOOL: {"atol": 1e-3, "rtol": 1e-3},  # 2
    InfiniDtype.I8: {"atol": 1e-3, "rtol": 1e-3},  # 3
    InfiniDtype.I16: {"atol": 1e-3, "rtol": 1e-3},  # 4
    InfiniDtype.I32: {"atol": 1e-3, "rtol": 1e-3},  # 5
    InfiniDtype.I64: {"atol": 1e-3, "rtol": 1e-3},  # 6
    InfiniDtype.U8: {"atol": 1e-3, "rtol": 1e-3},  # 7
    InfiniDtype.U16: {"atol": 1e-3, "rtol": 1e-3},  # 8
    InfiniDtype.U32: {"atol": 1e-3, "rtol": 1e-3},  # 9
    InfiniDtype.U64: {"atol": 1e-3, "rtol": 1e-3},  # 10
    InfiniDtype.F8: {"atol": 1e-3, "rtol": 1e-3},  # 11
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},  # 12
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},  # 13
    InfiniDtype.F64: {"atol": 1e-3, "rtol": 1e-3},  # 14
    InfiniDtype.BF16: {"atol": 1e-3, "rtol": 1e-3},  # 19
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_zeros(y, x):
    # y[...] = 0
    y.copy_(torch.zeros_like(y))


def test(
        handle,
        device,
        shape,
        x_stride=None,
        y_stride=None,
        inplace=Inplace.OUT_OF_PLACE,
        dtype=None,
        sync=None,
):
    if dtype in [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32, InfiniDtype.F64]:
        x = TestTensor(shape, x_stride, dtype, device)
    elif dtype in [InfiniDtype.BYTE, InfiniDtype.U8, InfiniDtype.U16, InfiniDtype.U32, InfiniDtype.U64,
                   InfiniDtype.I8, InfiniDtype.I16, InfiniDtype.I32, InfiniDtype.I64]:
        x = TestTensor(shape, x_stride, dtype, device, mode="randint", randint_low=0, randint_high=16)
    elif dtype in [InfiniDtype.F8]:
        x = TestTensor(shape, x_stride, dtype, device, mode="float8_e4m3fn")
    elif dtype in [InfiniDtype.BOOL]:
        x = TestTensor(shape, x_stride, dtype, device, mode="randint", randint_low=0, randint_high=2)
    else:
        raise ValueError("Unsupported dtype")

    if inplace == Inplace.INPLACE_X:
        if x_stride != y_stride:
            return
        y = x
    else:
        y = TestTensor(shape, y_stride, dtype, device, mode="ones")

    if y.is_broadcast():
        return

    print(
        f"Testing Zeros on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    torch_zeros(y.torch_tensor(), x.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateZerosDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [y, x]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetZerosWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )

    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_zeros():
        check_error(
            LIBINFINIOP.infiniopZeros(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_zeros()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)


    assert torch.allclose(y.actual_tensor().to(dtype=torch.float32), y.torch_tensor().to(dtype=torch.float32), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_zeros(y.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_zeros(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyZerosDescriptor(descriptor))


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
