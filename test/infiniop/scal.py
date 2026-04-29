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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
# Format: (shape, x_stride, alpha)
_TEST_CASES = [
    ((13,), None, 2.5),
    ((13,), (10,), 2.5),
    ((5632,), None, 2.5),
    ((5632,), (5,), 2.5),
    ((16,), (4,), 2.5),
    ((5632,), (32,), 2.5),
]

# Data types used for testing
_TENSOR_DTYPES = [
    InfiniDtype.F16,
    InfiniDtype.F32,
    # InfiniDtype.F64,
    InfiniDtype.BF16,
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.F32: {"atol": 1e-7, "rtol": 1e-7},
    InfiniDtype.F64: {"atol": 1e-15, "rtol": 1e-15},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def scal(x, alpha):
    x.mul_(alpha)


def test(
    handle,
    device,
    shape,
    x_stride=None,
    alpha_value=2.5,
    dtype=torch.float16,
    sync=None,
):
    alpha_torch = torch.tensor([alpha_value])
    alpha = TestTensor(
        alpha_torch.shape,
        alpha_torch.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=alpha_torch,
    )
    x = TestTensor(shape, x_stride, dtype, device)

    if x.is_broadcast():
        return

    print(
        f"Testing Scal on {InfiniDeviceNames[device]} with shape:{shape} x_stride:{x_stride} "
        f"dtype:{InfiniDtypeNames[dtype]}"
    )

    # Compute PyTorch reference
    scal(x.torch_tensor(), alpha.torch_tensor())

    if sync is not None:
        sync()

    # Create Descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateScalDescriptor(
            handle,
            ctypes.byref(descriptor),
            alpha.descriptor,
            x.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    alpha.destroy_desc()
    x.destroy_desc()

    # Allocate Workspace
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetScalWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    # Execute C library op
    def lib_scal():
        check_error(
            LIBINFINIOP.infiniopScal(
                descriptor,
                workspace.data(),
                workspace.size(),
                alpha.data(),
                x.data(),
                None,
            )
        )

    lib_scal()

    # Compare results
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)

    assert torch.allclose(x.actual_tensor(), x.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: scal(x.torch_tensor(), alpha.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_scal(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyScalDescriptor(descriptor))


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
