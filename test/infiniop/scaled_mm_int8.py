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
# _TENSOR_DTYPES = [InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 5e-2},
    InfiniDtype.BF16: {"atol": 1e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 5e-3},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def linearFunction(c, bias, x, w, alpha, beta):
    ans = (
        alpha * torch.matmul(x.to(torch.float32), w.to(torch.float32)).to(x.dtype)
        + beta * c
        + bias
    )
    return ans

def scaled_mm_int8_with_bias(
    A_int8,           # (M, K) int8
    B_int8,           # (K, N) int8
    bias,             # (N,) or (M, N) — 通常为 (N,)
    scale_a,          # scalar or (M,) or ()
    scale_b,          # scalar or (N,) or ()
    scale_bias=None,  # scalar or (N,) — 可选：若 bias 也是量化过的
    out_dtype=torch.float16
):
    """
    Perform scaled int8 matrix multiplication with bias:
        output = (A_int8 * scale_a) @ (B_int8 * scale_b) + bias

    If bias is already in the correct scale (e.g., float), apply directly.
    If bias is quantized (e.g., int32), use scale_bias to dequantize it.
    """
    # Dequantize A and B
    A_f = A_int8.to(out_dtype) * scale_a
    B_f = B_int8.to(out_dtype) * scale_b

    # Matrix multiplication
    output = torch.mm(A_f, B_f)  # (M, N)

    # Handle bias
    if bias is not None:
        if scale_bias is not None:
            # Assume bias is int32 or int8 and needs scaling
            bias_f = bias.to(out_dtype) * scale_bias
        else:
            # Assume bias is already in float
            bias_f = bias.to(out_dtype)
        output = output + bias_f

    return output


def quantWeights(w: torch.Tensor, symmetric, axis):
    """
    对权重矩阵 w ∈ [K, N] 做 per-channel (按列) 量化。
    返回:
      w_packed: int8 量化权重，形状 [K, N]
      w_scale:  每列的scale，形状 [1, N]，dtype与w相同
      w_zero:   每列的zero point，形状 [1, N]，dtype与w相同
    """
    assert w.dim() == 2, "w must be [K, N]"
    if symmetric:
        # 对称量化：zero=0, 只用最大绝对值
        w_abs_max = torch.max(w.abs(), dim=axis, keepdim=True)[0]

        # 避免除 0
        w_scale = w_abs_max / 127.0
        w_scale = torch.clamp(w_scale, min=1e-8)

        # 计算量化值 q = round(w / scale)
        w_q = torch.round(w / w_scale)

        # 限制到 [-128, 127]
        w_q = torch.clamp(w_q, -128, 127)

        # 转 int8
        w_packed = w_q.to(torch.int8)

        # 对称量化 zero 固定为 0
        w_zero = None

        return w_packed, w_scale.to(w.dtype), w_zero
    else:
        # 计算每列的最小值和最大值
        w_min = w.min(dim=axis, keepdim=True)[0]
        w_max = w.max(dim=axis, keepdim=True)[0]

        # 避免除以零
        w_scale = (w_max - w_min) / 255.0
        w_scale = torch.clamp(w_scale, min=1e-8)

        # 计算zero point
        w_zero = -w_min / w_scale - 128.0

        # 计算量化值
        w_q = torch.round(w / w_scale + w_zero)

        # 限制范围[-128, 127]
        w_q = torch.clamp(w_q, -128, 127)

        # 转为int8
        w_packed = w_q.to(torch.int8)

        return w_packed, w_scale.to(w.dtype), w_zero.to(w.dtype)


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
    bias = TestTensor((N,), None, dtype, device)
    x = TestTensor(x_shape, None, dtype, device)
    w = TestTensor(w_shape, None, dtype, device)
    y = TestTensor(y_shape, None, dtype, device)

    if inplace == Inplace.INPLACE:
        d = y
    else:
        d = TestTensor(y_shape, None, dtype, device)
    ans1 = linearFunction(
        y.torch_tensor(),
        bias.torch_tensor(),
        x.torch_tensor(),
        w.torch_tensor(),
        alpha,
        beta,
    )

    x_p, x_s, x_z = quantWeights(x.torch_tensor(), symmetric, 1)
    x_packed = TestTensor(
        x_shape, x_p.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=x_p
    )
    x_scale = TestTensor((M, 1), x_s.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=x_s)
    if symmetric:
        x_zero = None
    else:
        x_zero = TestTensor((M, 1), x_z.stride(), dtype, device, mode="manual", set_tensor=x_z)

    w_packed, w_scale, w_zero = quantWeights(w.torch_tensor(), symmetric, 0)
    weights = TestTensor(
        w_shape, w_packed.stride(), InfiniDtype.I8, device, mode="manual", set_tensor=w_packed
    )
    weights_scale = TestTensor(
        (1, N), w_scale.stride(), InfiniDtype.F32, device, mode="manual", set_tensor=w_scale
    )
    if symmetric:
        weights_zero = None
    else:
        weights_zero = TestTensor(
            (1, N), w_zero.stride(), dtype, device, mode="manual", set_tensor=w_zero
        )

    if sync is not None:
        sync()
        
    ans = scaled_mm_int8_with_bias(
        x_packed.torch_tensor().to(torch.int8),
        weights.torch_tensor().to(torch.int8),
        bias.torch_tensor(),
        x_scale.torch_tensor(),
        weights_scale.torch_tensor(),
        out_dtype=torch.float16 if dtype == InfiniDtype.F16 else torch.bfloat16
    )

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

    # # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
    x.destroy_desc()
    y.destroy_desc()
    d.destroy_desc()
    bias.destroy_desc()
    x_packed.destroy_desc()
    x_scale.destroy_desc()
    if symmetric == False:
        x_zero.destroy_desc()
    weights.destroy_desc()
    weights_scale.destroy_desc()
    if symmetric == False:
        weights_zero.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetI8GemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, x.device)

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
        debug(d.actual_tensor(), ans, atol=atol, rtol=rtol)

    # print(y.actual_tensor())
    # print(ans1)
    
    # assert torch.allclose(y.actual_tensor(), ans, atol=atol, rtol=rtol)

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