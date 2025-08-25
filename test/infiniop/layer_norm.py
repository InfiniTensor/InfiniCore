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
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # 3D测试用例（LayerNorm要求至少3D）
    ((2, 16, 2048), (2, 16, 2048), (2048,), (2048,), None, None),
    ((2, 16, 2048), (2, 16, 2048), (2048,), None, None, None),  # 无bias
    ((4, 8, 1024), (4, 8, 1024), (1024,), (1024,), None, None),
    ((4, 8, 1024), (4, 8, 1024), (1024,), None, None, None),  # 无bias
    ((1, 32, 512), (1, 32, 512), (512,), (512,), None, None),
    ((1, 32, 512), (1, 32, 512), (512,), None, None, None),  # 无bias
    # 4D测试用例（测试更高维度支持）
    ((2, 4, 8, 256), (2, 4, 8, 256), (256,), (256,), None, None),
    ((2, 4, 8, 256), (2, 4, 8, 256), (256,), None, None, None),  # 无bias
    # 5D测试用例（测试更高维度支持）
    ((1, 2, 3, 4, 128), (1, 2, 3, 4, 128), (128,), (128,), None, None),
    ((1, 2, 3, 4, 128), (1, 2, 3, 4, 128), (128,), None, None, None),  # 无bias
]

# w (weight) and b (bias) types
# Note: 'None' means the same as input dtype
# Only test same precision cases to avoid mixed precision complexity
_WEIGHT_DTYPES = [None]
_BIAS_DTYPES = [None]
# x types used for testing - support F32, F16, BF16 as required by competition
_TENSOR_DTYPES = [InfiniDtype.F32, InfiniDtype.F16, InfiniDtype.BF16]

# Form the test cases by appending each element of _WEIGHT_DTYPES and _BIAS_DTYPES to each tuple in _TEST_CASES_
_TEST_CASES = [
    test_case + (w_dtype, b_dtype) for test_case in _TEST_CASES_ 
    for w_dtype in _WEIGHT_DTYPES for b_dtype in _BIAS_DTYPES
]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
    InfiniDtype.F16: {"atol": 5e-3, "rtol": 5e-3},  
    InfiniDtype.BF16: {"atol": 2e-2, "rtol": 2e-2},  
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def layer_norm(ans, x, w, b, eps):
    # Calculate mean and variance along the last dimension
    mean = torch.mean(x, dim=-1, keepdim=True)
    var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
    
    # Normalize
    torch.sub(x, mean, out=ans)
    torch.div(ans, torch.sqrt(var + eps), out=ans)
    
    # Scale and shift
    ans.mul_(w)
    if b is not None:
        ans.add_(b)


def test(
    handle,
    device,
    y_shape,
    x_shape,
    w_shape,
    b_shape,
    y_stride,
    x_stride,
    w_dtype=InfiniDtype.F32,
    b_dtype=InfiniDtype.F32,
    dtype=InfiniDtype.F16,
    sync=None,
):
    w_dtype = w_dtype if w_dtype else dtype
    b_dtype = b_dtype if b_dtype else dtype
    print(
        f"Testing LayerNorm on {InfiniDeviceNames[device]} with y_shape:{y_shape} x_shape:{x_shape} w_shape:{w_shape} b_shape:{b_shape}"
        f" y_stride:{y_stride} x_stride:{x_stride} w_dtype:{InfiniDtypeNames[w_dtype]} b_dtype:{InfiniDtypeNames[b_dtype]} dtype:{InfiniDtypeNames[dtype]}"
    )

    y = TestTensor(y_shape, y_stride, dtype, device, mode="zeros")
    x = TestTensor(x_shape, x_stride, dtype, device, scale=0.01)
    w = TestTensor(w_shape, None, w_dtype, device)
    b = TestTensor(b_shape, None, b_dtype, device) if b_shape is not None else None
    # Create output tensors for std_deviation and standardization
    input_std_deviation = TestTensor(x_shape[:-1], None, dtype, device, mode="ones")
    input_standardization = TestTensor(x_shape, x_stride, dtype, device, mode="ones")

    eps = 1e-5
    layer_norm(y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), 
               b.torch_tensor() if b is not None else None, eps)

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()

    print(f"Creating LayerNorm descriptor with shapes:")
    print(f"  y: {y_shape}, x: {x_shape}, w: {w_shape}, b: {b_shape}")
    print(f"  eps: {eps}, dtype: {dtype}")
    
    status = LIBINFINIOP.infiniopCreateLayerNormDescriptor(
        handle,
        ctypes.byref(descriptor),
        y.descriptor,
        x.descriptor,
        w.descriptor,
        b.descriptor if b is not None else None,
        input_std_deviation.descriptor,
        input_standardization.descriptor,
        ctypes.c_float(eps),
    )
    print(f"Descriptor creation status: {status}")
    check_error(status)

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    for tensor in [x, y, w, input_std_deviation, input_standardization] + ([b] if b is not None else []):
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetLayerNormWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, y.device)

    def lib_layer_norm():
        print(f"Executing LayerNorm with workspace_size: {workspace_size.value}")
        status = LIBINFINIOP.infiniopLayerNorm(
            descriptor,
            workspace.data(),
            workspace_size.value,
            y.data(),
            x.data(),
            w.data(),
            b.data() if b is not None else None,
            input_std_deviation.data(),
            input_standardization.data(),
            None,
        )
        print(f"LayerNorm execution status: {status}")
        check_error(status)

    lib_layer_norm()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)
    assert torch.allclose(y.actual_tensor(), y.torch_tensor(), atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: layer_norm(y.torch_tensor(), x.torch_tensor(), w.torch_tensor(), b.torch_tensor(), eps), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_layer_norm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    check_error(LIBINFINIOP.infiniopDestroyLayerNormDescriptor(descriptor))


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