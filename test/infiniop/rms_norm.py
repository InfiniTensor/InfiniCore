
import torch
import ctypes
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from pyinfini import (
    to_tensor,
    get_test_devices,
    check_error,
    rearrange_if_needed,
    create_workspace,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    lib,
    infiniopRMSNormDescriptor_t
)

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # y_shape, x_shape, w_shape, y_stride, x_stride, w_dtype
    ((1, 4), (1, 4), (4,), None, None, torch.float32),
    ((16, 2048), (16, 2048), (2048,), None, None, torch.float32),
    ((16, 2048), (16, 2048), (2048,), None, None, torch.float16),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1), torch.float32),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1), torch.float16),
]

# x types used for testing
_TENSOR_DTYPES = [torch.float16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    torch.float16: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

def rms_norm(x, w, eps):
    input_dtype = x.dtype
    hidden_states = x.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + eps)
    return (w * hidden_states).to(input_dtype)

def test(
    lib,
    handle,
    torch_device,
    y_shape,
    x_shape,
    w_shape,
    y_stride,
    x_stride,
    w_dtype=torch.float16,
    dtype=torch.float16,
):
    print(
        f"Testing RMS_Norm on {torch_device} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape}"
        f" y_stride:{y_stride} x_stride:{x_stride} w_dtype:{w_dtype} dtype:{dtype}"
    )

    y = torch.zeros(y_shape, dtype=dtype).to(torch_device)
    x = torch.rand(x_shape, dtype=dtype).to(torch_device)
    w = torch.rand(w_shape, dtype=w_dtype).to(torch_device)

    eps = 1e-5
    ans = rms_norm(x, w, eps)

    x, y = [
        rearrange_if_needed(tensor, stride)
        for tensor, stride in zip([x, y], [x_stride, y_stride])
    ]

    x_tensor, y_tensor, w_tensor = [to_tensor(tensor, lib) for tensor in [x, y, w]]

    descriptor = infiniopRMSNormDescriptor_t()

    check_error(
        lib.infiniopCreateRMSNormDescriptor(
            handle,
            ctypes.byref(descriptor),
            y_tensor.descriptor,
            x_tensor.descriptor,
            w_tensor.descriptor,
            eps,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x_tensor, y_tensor, w_tensor]:
        tensor.destroyDesc(lib)

    workspace_size = ctypes.c_uint64(0)
    check_error(
        lib.infiniopGetRMSNormWorkspaceSize(descriptor, ctypes.byref(workspace_size))
    )
    workspace = create_workspace(workspace_size.value, y.device)

    def lib_rms_norm():
        check_error(
            lib.infiniopRMSNorm(
                descriptor,
                workspace.data_ptr() if workspace is not None else None,
                workspace_size.value,
                y_tensor.data,
                x_tensor.data,
                w_tensor.data,
                None,
            )
        )

    lib_rms_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y, ans, atol=atol, rtol=rtol)
    assert torch.allclose(y, ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: rms_norm(x, w, eps), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_rms_norm(), torch_device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(lib.infiniopDestroyRMSNormDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    # Execute tests
    for device in get_test_devices(args):
        test_operator(lib, device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
