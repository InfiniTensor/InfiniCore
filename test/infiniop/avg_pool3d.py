import torch
import ctypes
from ctypes import c_uint64, c_size_t, c_void_p
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

# Test cases format: (x_shape, x_stride_or_None, kernel_size, stride_or_None, padding)
_TEST_CASES = [
    ((1, 2, 8, 8, 8), None, (2, 2, 2), None, (0, 0, 0)),
    ((2, 3, 7, 9, 5), None, (3, 3, 3), (2, 2, 1), (1, 1, 0)),
    ((2, 1, 9, 11, 7), (693, 77, 77, 7, 1), (3, 2, 3), None, (1, 0, 1)),
]

_TENSOR_DTYPES = [InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-4},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_avg_pool3d(x, kernel_size, stride, padding):
    kwargs = {"kernel_size": kernel_size, "padding": padding}
    if stride is not None:
        kwargs["stride"] = stride
    return torch.nn.functional.avg_pool3d(x, **kwargs)


def test(
    handle,
    device,
    x_shape,
    x_stride,
    kernel_size,
    stride,
    padding,
    dtype=torch.float32,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    x = TestTensor(x_shape, x_stride, dtype, device)
    # For arbitrary (possibly overlapping) strides, the operator reads memory
    # according to (shape, strides) from the backing storage. Use actual_tensor
    # (the strided view) as the reference input to match that behavior.
    x_ref = x.actual_tensor() if x_stride is not None else x.torch_tensor()
    y_ref = torch_avg_pool3d(x_ref, kernel_size, stride, padding)
    y = TestTensor(tuple(y_ref.shape), None, dtype, device, mode="ones")
    y.update_torch_tensor(y_ref)

    print(
        f"Testing AvgPool3d on {InfiniDeviceNames[device]} with x_shape:{x_shape} x_stride:{x_stride} "
        f"kernel_size:{kernel_size} stride:{stride} padding:{padding} dtype:{InfiniDtypeNames[dtype]}"
    )

    if sync is not None:
        sync()

    ks_arr = (c_size_t * 3)(*kernel_size)
    stride_ptr = None
    if stride is not None:
        s_arr = (c_size_t * 3)(*stride)
        stride_ptr = ctypes.cast(s_arr, c_void_p)
    pad_arr = (c_size_t * 3)(*padding)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAvgPool3dDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            ctypes.cast(ks_arr, c_void_p),
            stride_ptr,
            ctypes.cast(pad_arr, c_void_p),
        )
    )

    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAvgPool3dWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_avg_pool3d():
        check_error(
            LIBINFINIOP.infiniopAvgPool3d(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_avg_pool3d()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch_avg_pool3d(x.torch_tensor(), kernel_size, stride, padding), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_avg_pool3d(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyAvgPool3dDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")
