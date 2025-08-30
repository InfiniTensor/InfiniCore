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
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not maxt to be imported from other modules
_TEST_CASES_ = [
    # y_shape, x_shape, y_stride, x_stride, dim
    ((), (), None, None, 0),
    ((1,), (32,), None, None, 0),
    ((1, 4), (1, 4), None, None, 0),
    ((1, 1), (1, 4), None, None, 1),
    ((16, 1), (16, 2048), None, None, 1),
    ((1, 16), (2048, 16), None, None, 0),
    ((16, 1), (16, 2048), (4096, 1), (4096, 1), 1),
    ((1, 2048), (16, 2048), (4096, 1), (4096, 1), 0),
    ((4, 4, 1), (4, 4, 2048), None, None, 2),
    ((1, 4, 4), (2048, 4, 4), None, None, 0),
    ((4, 1, 4), (4, 2048, 4), (45056, 5632, 1), (32768, 8, 1), 1),
]

# x types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TEST_CASES = _TEST_CASES_

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def reduce_max(x, dim):
    return x.max(dim=dim, keepdim=True)[0]


def test(
    handle,
    device,
    y_shape,
    x_shape,
    y_stride,
    x_stride,
    dim,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing Reduce_Max on {InfiniDeviceNames[device]} with y_shape:{y_shape} x_shape:{x_shape}"
        f" y_stride:{y_stride} x_stride:{x_stride} dim:{dim} dtype:{InfiniDtypeNames[dtype]}"
    )

    x = TestTensor(x_shape, x_stride, dtype, device)
    ans = reduce_max(x.torch_tensor(), dim)

    y = TestTensor(y_shape, y_stride, dtype, device)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateReduceMaxDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            ctypes.c_size_t(dim),
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetReduceMaxWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

    def lib_reduce_max():
        check_error(
            LIBINFINIOP.infiniopReduceMax(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                x.data(),
                None,
            )
        )

    lib_reduce_max()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: causal_softmax(x.torch_tensor()), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_causal_softmax(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyReduceMaxDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
