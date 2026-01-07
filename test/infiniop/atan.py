import ctypes
from ctypes import c_uint64
from enum import Enum, auto

import torch
from libinfiniop import (
    LIBINFINIOP,
    InfiniDeviceNames,
    InfiniDtype,
    InfiniDtypeNames,
    TestTensor,
    TestWorkspace,
    check_error,
    debug,
    get_args,
    get_test_devices,
    get_tolerance,
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # tensor_shape, inplace
    ((1, 3),),
    ((3, 3, 13, 9, 17),),
    ((32, 20, 512),),
    ((33, 333, 333),),
    ((32, 256, 112, 112),),
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

# Data types used for testing (matching old operators library: only F16 and F32)
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def atan_op(x):
    return torch.atan(x).to(x.dtype)


def test(
    handle, device, shape, inplace=Inplace.OUT_OF_PLACE, dtype=InfiniDtype.F16, sync=None
):
    # Generate test tensors with values in range [-200, -100) for atan operation
    # atan domain is (-∞, +∞), so we use range [-200, -100)
    x_torch_tensor = torch.rand(shape) * 100 - 200

    x = TestTensor(
        shape,
        x_torch_tensor.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=x_torch_tensor,
    )

    if inplace == Inplace.INPLACE_X:
        y = x
    else:
        y = TestTensor(shape, None, dtype, device)

    if y.is_broadcast():
        return

    print(
        f"Testing Atan on {InfiniDeviceNames[device]} with shape:{shape} dtype:{InfiniDtypeNames[dtype]} inplace: {inplace}"
    )

    ans = atan_op(x.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAtanDescriptor(
            handle, ctypes.byref(descriptor), y.descriptor, x.descriptor
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAtanWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_atan():
        check_error(
            LIBINFINIOP.infiniopAtan(
                descriptor, workspace.data(), workspace_size.value, y.data(), x.data(), None
            )
        )

    lib_atan()
    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol, equal_nan=True)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: atan_op(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_atan(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyAtanDescriptor(descriptor))


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
