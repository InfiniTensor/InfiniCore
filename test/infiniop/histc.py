import torch
import ctypes
from ctypes import c_uint64, c_int64, c_double
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
    # x_shape, x_stride, bins, min, max
    ((100,), None, 10, 0.0, 1.0),
    ((50,), None, 5, -1.0, 1.0),
    ((20,), (2,), 8, 0.0, 2.0),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.F32, InfiniDtype.BF16]

_TOLERANCE_MAP = {
    # histc produces exact integer counts in float32 for these sizes
    InfiniDtype.F16: {"atol": 0.0, "rtol": 0.0},
    InfiniDtype.BF16: {"atol": 0.0, "rtol": 0.0},
    InfiniDtype.F32: {"atol": 0.0, "rtol": 0.0},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def torch_histc(x, bins, min_val, max_val):
    # torch.histc on CUDA does not support float16/bfloat16 directly.
    return torch.histc(x.to(torch.float32), bins=bins, min=min_val, max=max_val)


def test(
    handle,
    device,
    x_shape,
    x_stride,
    bins,
    min_val,
    max_val,
    dtype=torch.float16,
    sync=None,
):
    torch.manual_seed(0)
    if device != 0:
        torch.cuda.manual_seed_all(0)

    x = TestTensor(x_shape, x_stride, dtype, device)
    # Make values fall into [min, max] and force edge hits.
    rng = max_val - min_val
    x_tensor = x.torch_tensor() * rng + min_val
    if x_tensor.numel() > 0:
        flat = x_tensor.reshape(-1)
        flat[0] = min_val
        flat[-1] = max_val
    x.set_tensor(x_tensor)

    y = TestTensor((bins,), None, InfiniDtype.F32, device, mode="zeros")

    print(
        f"Testing Histc on {InfiniDeviceNames[device]} with x_shape:{x_shape} x_stride:{x_stride} "
        f"bins:{bins} range:[{min_val},{max_val}] x_dtype:{InfiniDtypeNames[dtype]}"
    )

    y_ref = torch_histc(x.torch_tensor(), bins, min_val, max_val)
    y.update_torch_tensor(y_ref)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateHistcDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            x.descriptor,
            c_int64(bins),
            c_double(min_val),
            c_double(max_val),
        )
    )

    for tensor in [x, y]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetHistcWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_histc():
        check_error(
            LIBINFINIOP.infiniopHistc(
                descriptor,
                workspace.data(),
                workspace.size(),
                y.data(),
                x.data(),
                None,
            )
        )

    lib_histc()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: y_ref, device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_histc(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyHistcDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92m  Test passed!  \033[0m")
