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
# Configuration
# ==============================================================================
# Format: (shape, x_stride)
_TEST_CASES = [
    ((13,), None),
    ((13,), (10,)),
    ((5632,), None),
    ((5632,), (5,)),
    ((16,), (4,)),
    ((5632,), (32,)),
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
    InfiniDtype.F64: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_nrm2(x):
    return torch.norm(x, p=2)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    dtype=InfiniDtype.F32,
    sync=None,
):
    x = TestTensor(shape, x_stride, dtype, device)

    result = TestTensor(tuple(), None, dtype, device, mode="zeros")

    print(
        f"Testing Nrm2 on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    result_ref = torch_nrm2(x.torch_tensor())
    result.update_torch_tensor(result_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateNrm2Descriptor(
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
        LIBINFINIOP.infiniopGetNrm2WorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_nrm2():
        check_error(
            LIBINFINIOP.infiniopNrm2(
                descriptor,
                workspace.data(),
                workspace.size(),
                x.data(),
                result.data(),
                None,
            )
        )

    lib_nrm2()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

    if DEBUG:
        debug(result.actual_tensor(), result.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(
        result.actual_tensor(), result.torch_tensor(), atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: torch_nrm2(x.torch_tensor()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib", lambda: lib_nrm2(), device, NUM_PRERUN, NUM_ITERATIONS
        )

    check_error(LIBINFINIOP.infiniopDestroyNrm2Descriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
