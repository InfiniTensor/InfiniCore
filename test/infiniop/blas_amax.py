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
    infiniopOperatorDescriptor_t,
    profile_operation,
    test_operator,
)

# ==============================================================================
#  Configuration
# ==============================================================================

_TEST_CASES = [
    # n, x_stride
    (3, None),
    (8, (2,)),
    (32, None),
    (257, (3,)),
    (65535, None),
]

_TENSOR_DTYPES = [
    # InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    # InfiniDtype.BF16,
]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def test(
    handle,
    device,
    n,
    x_stride=None,
    dtype=torch.float16,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    x = TestTensor((n,), x_stride, dtype, device)
    result = TestTensor(tuple(), None, InfiniDtype.I32, device, mode="zeros")

    print(
        f"Testing blas_amax on {InfiniDeviceNames[device]} with n:{n} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    result_ref = torch.argmax(x.torch_tensor().abs()).to(torch.int32) + 1
    result.update_torch_tensor(result_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateBlasAmaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            x.descriptor,
            result.descriptor,
        )
    )

    for tensor in [x, result]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetBlasAmaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, result.device)

    def lib_blas_amax():
        check_error(
            LIBINFINIOP.infiniopBlasAmax(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                result.data(),
                None,
            )
        )

    lib_blas_amax()

    if DEBUG:
        debug(result.actual_tensor(), result.torch_tensor())
    assert torch.equal(result.actual_tensor(), result.torch_tensor())

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch.argmax(x.torch_tensor().abs()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_blas_amax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyBlasAmaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")
