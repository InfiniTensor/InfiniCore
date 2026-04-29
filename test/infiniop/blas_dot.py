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
    InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    InfiniDtype.BF16,
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


def test(
    handle,
    device,
    n,
    x_stride=None,
    y_stride=None,
    dtype=torch.float16,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    x = TestTensor((n,), x_stride, dtype, device)
    y = TestTensor((n,), y_stride, dtype, device)
    result = TestTensor(tuple(), None, dtype, device, mode="zeros")

    print(
        f"Testing blas_dot on {InfiniDeviceNames[device]} with n:{n} x_stride:{x_stride} y_stride:{y_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    def torch_blas_dot():
        if dtype in (InfiniDtype.F16, InfiniDtype.BF16):
            return torch.dot(x.torch_tensor().float(), y.torch_tensor().float()).to(
                x.torch_tensor().dtype
            )
        return torch.dot(x.torch_tensor(), y.torch_tensor())

    result_ref = torch_blas_dot()
    result.update_torch_tensor(result_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBlasDotDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            y.descriptor,
            result.descriptor,
        )
    )

    for tensor in [x, result]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBlasDotWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, result.device)

    def lib_blas_dot():
        check_error(
            LIBINFINIOP.infiniopBlasDot(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                y.data(),
                result.data(),
                None,
            )
        )

    lib_blas_dot()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(result.actual_tensor(), result.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(
        result.actual_tensor(), result.torch_tensor(), atol=atol, rtol=rtol
    )

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_blas_dot(), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_blas_dot(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyBlasDotDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")
