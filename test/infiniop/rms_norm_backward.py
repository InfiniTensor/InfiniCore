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

# Test cases
_TEST_CASES_ = [
    # Basic cases
    ((1, 4), (4,), (1, 4), (1, 4), (4,), None, None, None),
    ((2, 4), (4,), (2, 4), (2, 4), (4,), None, None, None),
    ((2, 4, 8), (8,), (2, 4, 8), (2, 4, 8), (8,), None, None, None),
    ((2, 8, 16), (16,), (2, 8, 16), (2, 8, 16), (16,), None, None, None),
    # Larger cases
    ((2, 16, 2048), (2048,), (2, 16, 2048), (2, 16, 2048), (2048,), None, None, None),
    ((16, 2048), (2048,), (16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1), (4096, 1)),
    ((32, 1024), (1024,), (32, 1024), (32, 1024), (1024,), None, None, None),
    ((64, 512), (512,), (64, 512), (64, 512), (512,), None, None, None),
    # Multi-dimensional cases
    ((4, 32, 1024), (1024,), (4, 32, 1024), (4, 32, 1024), (1024,), None, None, None),
    ((8, 64, 512), (512,), (8, 64, 512), (8, 64, 512), (512,), None, None, None),
    ((2, 128, 768), (768,), (2, 128, 768), (2, 128, 768), (768,), None, None, None),
    ((1, 256, 1536), (1536,), (1, 256, 1536), (1, 256, 1536), (1536,), None, None, None),
    # Strided cases
    ((2, 16, 2048), (2048,), (2, 16, 2048), (2, 16, 2048), (2048,), (65536, 2048, 1), (65536, 2048, 1), (65536, 2048, 1)),
    ((4, 32, 1024), (1024,), (4, 32, 1024), (4, 32, 1024), (1024,), (32768, 1024, 1), (32768, 1024, 1), (32768, 1024, 1))
]

# Tensor dtypes
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]

# Tolerance map (slightly relaxed for adaptive epsilon)
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def compute_adaptive_epsilon(x, base_epsilon=1e-6):
    """
    Compute adaptive epsilon matching the kernel logic
    Convert to float32 first to match kernel's preciseCast behavior
    """
    # Convert to float32 to match kernel's preciseCast logic
    x_float = x.to(torch.float32)
    x_min = torch.min(x_float)
    x_max = torch.max(x_float)
    data_range = x_max - x_min
    adaptive_epsilon = max(base_epsilon, data_range * 1e-8)
    return float(adaptive_epsilon)


def torch_rms_norm_backward(x, dy, w, epsilon=1e-6):
    """
    PyTorch RMSNorm backward reference implementation with configurable epsilon
    """
    # Create RMSNorm layer with specified epsilon
    rms_norm = torch.nn.RMSNorm(x.shape[-1], eps=epsilon, device=x.device, dtype=x.dtype)
    rms_norm.weight.data = w.data
    
    x_copy = x.detach().requires_grad_(True)
    w_copy = rms_norm.weight
    
    y = rms_norm(x_copy)
    y.backward(dy)
    
    return x_copy.grad, w_copy.grad


def test(
    handle,
    device,
    dx_shape,
    dw_shape,
    dy_shape,
    x_shape,
    w_shape,
    dx_stride,
    dy_stride,
    x_stride,
    dtype=InfiniDtype.F16,
    sync=None,
):

    base_eps = 1e-6
    
    # Print test info
    device_name = InfiniDeviceNames[device]
    dtype_name = InfiniDtypeNames[dtype]
    print(f"Testing RMSNormBackward on {device_name} with shape:{x_shape}, dtype:{dtype_name}")

    # Create tensors
    dx = TestTensor(dx_shape, dx_stride, dtype, device, mode="ones")
    dw = TestTensor(dw_shape, None, dtype, device, mode="ones")
    dy = TestTensor(dy_shape, dy_stride, dtype, device, scale=0.01)
    x = TestTensor(x_shape, x_stride, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, dtype, device)
    
    # Compute adaptive epsilon matching kernel logic
    adaptive_eps = compute_adaptive_epsilon(x.torch_tensor(), base_eps)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    check_error(
        LIBINFINIOP.infiniopCreateRMSNormBackwardDescriptor(
            handle,
            ctypes.byref(descriptor),
            dx.descriptor,  # input_grad_desc
            dw.descriptor,  # weight_grad_desc
            dy.descriptor,  # output_grad_desc
            x.descriptor,   # input_desc
            w.descriptor,   # weight_desc
            ctypes.c_float(adaptive_eps),  # adaptive epsilon
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [dx, dw, dy, x, w]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRMSNormBackwardWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, dx.device)

    def lib_rms_norm_backward():
        check_error(
            LIBINFINIOP.infiniopRMSNormBackward(
                descriptor,
                workspace.data(),
                workspace_size.value,
                dx.data(),
                dw.data(),
                dy.data(),
                x.data(),
                w.data(),
                None,
            )
        )

    # Get PyTorch reference gradients using same adaptive epsilon
    dx_ref, dw_ref = torch_rms_norm_backward(
        x.torch_tensor(),
        dy.torch_tensor(),
        w.torch_tensor(),
        epsilon=adaptive_eps
    )

    # Run our implementation
    lib_rms_norm_backward()
    
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    if DEBUG:
        debug(dx.actual_tensor(), dx_ref, atol=atol, rtol=rtol)
        debug(dw.actual_tensor(), dw_ref, atol=atol, rtol=rtol)
    
    # Compare with PyTorch reference
    dx_actual = dx.actual_tensor()
    dw_actual = dw.actual_tensor()
    
    dx_close = torch.allclose(dx_actual, dx_ref, atol=atol, rtol=rtol)
    dw_close = torch.allclose(dw_actual, dw_ref, atol=atol, rtol=rtol)
    
    if not dx_close or not dw_close:
        dx_max_diff = torch.max(torch.abs(dx_actual - dx_ref))
        dw_max_diff = torch.max(torch.abs(dw_actual - dw_ref))
        print(f"  ✗ Test failed - dx_diff: {dx_max_diff:.2e}, dw_diff: {dw_max_diff:.2e}")
        assert dx_close, f"dx mismatch: max_diff={dx_max_diff}"
        assert dw_close, f"dw mismatch: max_diff={dw_max_diff}"
    else:
        print(f"  ✓ Test passed")

    if PROFILE:
        profile_operation(lib_rms_norm_backward, NUM_PRERUN, NUM_ITERATIONS)


if __name__ == "__main__":
    # Set fixed random seed for reproducible tests
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    
    args = get_args()
    
    # Update global variables based on args
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations
    
    # Run tests
    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")