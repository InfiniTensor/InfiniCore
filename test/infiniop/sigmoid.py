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
    ((13, 4, 4), (4, 0, 1), (4, 0, 1)),
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), (13312, 1)),
    ((4, 4, 5632), None, None),
    ((4, 4, 5632), (45056, 5632, 1), (45056, 5632, 1)),
    ((4, 4, 56320), None, None),
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
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_sigmoid(y, x):
    torch.sigmoid(x, out=y)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    y_stride=None,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)
    if inplace == Inplace.INPLACE_X:
        if x_stride != y_stride:
            return
        y = x
    else:
        y = TestTensor(shape, y_stride, dtype, device, mode="ones")

    if y.is_broadcast():
        return

    print(
        f"Testing Sigmoid on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} y_stride:{y_stride} "
        f"dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    torch_sigmoid(y.torch_tensor(), x.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSigmoidDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSigmoidWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_sigmoid():
        check_error(
            LIBINFINIOP.infiniopSigmoid(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_sigmoid()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_sigmoid(y.torch_tensor(), x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_sigmoid(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroySigmoidDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")
