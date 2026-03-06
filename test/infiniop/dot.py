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

# ==============================================================================
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # n, a_stride, b_stride
    (3, None, None),
    (8, (2,), (2,)),
    (32, None, None),
    (257, (3,), (3,)),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def test(
    handle,
    device,
    n,
    a_stride=None,
    b_stride=None,
    dtype=torch.float16,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    a = TestTensor((n,), a_stride, dtype, device)
    b = TestTensor((n,), b_stride, dtype, device)
    y = TestTensor((1,), None, dtype, device, mode="zeros")

    print(
        f"Testing dot on {InfiniDeviceNames[device]} with n:{n} a_stride:{a_stride} b_stride:{b_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    y_ref = torch.dot(a.torch_tensor().reshape(-1), b.torch_tensor().reshape(-1)).reshape(1)
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateDotDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            a.descriptor,
            b.descriptor,
        )
    )

    for tensor in [a, b, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetDotWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_dot():
        check_error(
            LIBINFINIOP.infiniopDot(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                a.data(),
                b.data(),
                None,
            )
        )

    lib_dot()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch.dot(a.torch_tensor().reshape(-1), b.torch_tensor().reshape(-1)), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_dot(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyDotDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")

