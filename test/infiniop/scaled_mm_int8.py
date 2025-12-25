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
from enum import Enum, auto

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # x_shape, w_shape, y_shape, alpha, beta
    # ((8, 8), (8, 8), False, (8, 8), 1.0, 0.0),
    ((128, 512), (512, 1024), True, (128, 1024), 1.0, 0.0),
    # ((128, 128), (128, 128), False, (128, 128), 2.0, 1.0),
    ((256, 1024), (1024, 2048), True, (256, 2048), 1.0, 1.0),
    ((1024, 2048), (2048, 1024), True, (1024, 1024), 1.0, 0.0),
]


class Inplace(Enum):
    OUT_OF_PLACE = auto()
    INPLACE = auto()


# Inplace options applied for each test case in _TEST_CASES_
_INPLACE = [
    # Inplace.OUT_OF_PLACE,
    Inplace.INPLACE,
]

_TEST_CASES = [
    test_case + (inplace_item,)
    for test_case in _TEST_CASES_
    for inplace_item in _INPLACE
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 3e-1, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 3e-1, "rtol": 1e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000
    
def to_int8(tensor: torch.Tensor) -> torch.Tensor:
    return torch.round(tensor.clamp(min=-128, max=127)).to(dtype=torch.int8)

def torch_scaled_mm(a, b, scale_a, scale_b, out_dtype, bias):
    o = torch.matmul(a.to(torch.float32), b.to(torch.float32))
    if bias is not None:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1) + bias
    else:
        o = o.to(torch.float32) * scale_a.view(-1, 1) * scale_b.view(1, -1)
    return o.to(out_dtype)

def test(
    handle,
    device,
    x_shape,
    w_shape,
    symmetric,
    y_shape,
    alpha,
    beta,
    inplace=Inplace.OUT_OF_PLACE,
    dtype=InfiniDtype.BF16,
    sync=None,
):
    print(
        f"Testing Linear on {InfiniDeviceNames[device]} with x_shape:{x_shape}, w_shape:{w_shape}, symmetric:{symmetric}, alpha:{alpha}, beta:{beta}, inplace:{inplace} dtype:{InfiniDtypeNames[dtype]}"
    )
    M, K = x_shape
    N = w_shape[1]
    
    x_packed = to_int8(torch.randn((M, K), device="cuda") * 5)
    weights = to_int8(torch.randn((N, K), device="cuda").t() * 5)
    
    x_scale = torch.randn((M,), device="cuda", dtype=torch.float32)
    weights_scale = torch.randn((N,), device="cuda", dtype=torch.float32)
    bias = torch.randn((N,), device="cuda", dtype=torch.float16 if dtype == InfiniDtype.F16 else torch.bfloat16) * 10

    ans = torch_scaled_mm(x_packed, weights, x_scale, weights_scale, torch.float16 if dtype == InfiniDtype.F16 else torch.bfloat16, bias=bias)
    
    x_packed = TestTensor(
        (M, K), x_packed.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=x_packed
    )
    x_scale = TestTensor(
        (M,), x_scale.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=x_scale
    )
    weights = TestTensor(
        (K, N), weights.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=weights
    )
    weights_scale = TestTensor(
        (N,), weights_scale.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=weights_scale
    )
    y = TestTensor(y_shape, None, dtype, device)
    bias = TestTensor((N,), bias.stride(), dtype, device, mode="manual", set_tensor=bias)

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateI8GemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            y.descriptor,
            bias.descriptor,
            x_packed.descriptor,
            x_scale.descriptor,
            weights.descriptor,
            weights_scale.descriptor,
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetI8GemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x_packed.device)

    def lib_linear():
        check_error(
            LIBINFINIOP.infiniopI8Gemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                y.data(),
                bias.data(),
                x_packed.data(),
                x_scale.data(),
                weights.data(),
                weights_scale.data(),
                None,
            )
        )

    lib_linear()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: linearFunction(y.torch_tensor(), bias.torch_tensor(), x.torch_tensor(), w.torch_tensor(), alpha, beta), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyI8GemmDescriptor(descriptor))


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
