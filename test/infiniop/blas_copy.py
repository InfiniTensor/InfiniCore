import ctypes
from ctypes import c_uint64

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
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # n, x_stride, y_stride
    (3, None, None),
    (8, (2,), (3,)),
    (32, None, (2,)),
    (257, (3,), None),
    (65535, None, None),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    # InfiniDtype.BF16,
]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F64: {"atol": 1e-9, "rtol": 1e-9},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_copy(x, y):
    y.copy_(x)


def test(
    handle,
    device,
    n,
    x_stride=None,
    y_stride=None,
    dtype=torch.float16,
    sync=None,
):
    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)

    if x.is_broadcast():
        return

    print(
        f"Testing BlasCopy on {InfiniDeviceNames[device]} with n:{n} x_stride:{x_stride} "
        f"y_stride:{y_stride} dtype:{InfiniDtypeNames[dtype]}"
    )

    torch_copy(x.torch_tensor(), y.torch_tensor())

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBlasCopyDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor,
        )
    )

    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBlasCopyWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_copy():
        check_error(
            LIBINFINIOP.infiniopBlasCopy(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                y.data(),
                None,
            )
        )

    lib_copy()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch_copy(x.torch_tensor(), y.torch_tensor()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_copy(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyBlasCopyDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
