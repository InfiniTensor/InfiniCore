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
    # shape, axis
    ((4, 4), 0),
    ((12, 16, 512, 512), 0),
    ((12, 16, 512, 512), 1),
    ((12, 16, 512, 512), 2),
    ((12, 16, 512, 512), 3),
    ((1, 16, 512, 512), 0),
    ((1, 16, 512, 512), 1),
    ((1, 16, 512, 512), 2),
    ((1, 16, 512, 512), 3),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 1e-5},
}


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE_X = auto()


_INPLACE = [
    Inplace.INPLACE_X,
    Inplace.OUT_OF_PLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def softmax(x, axis):
    return torch.softmax(x, axis)


def test(
    handle,
    device,
    shape,
    axis,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Softmax on {InfiniDeviceNames[device]} with shape:{shape}, axis:{axis} dtype:{InfiniDtypeNames[dtype]} inplace:{inplace}"
    )

    x = TestTensor(shape, None, dtype, device)
    ans = softmax(x.torch_tensor(), axis)

    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(shape, None, dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateSoftmaxDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor, axis
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetSoftmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_softmax():
        check_error(
            LIBINFINIOP.infiniopSoftmax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_softmax()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: softmax(x.torch_tensor(), axis), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_softmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroySoftmaxDescriptor(descriptor))


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
