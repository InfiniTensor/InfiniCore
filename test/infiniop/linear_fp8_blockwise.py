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
from libinfiniop import to_torch_dtype, torch_device_map
import numpy as np

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES = [
    # alpha, beta, a_shape, b_shape, c_shape, a_stride, b_stride, c_stride
    (1.0, 1.0, (256, 1024), (384, 1024), (256, 384), None, None, None),
    (2.0, 0.0, (256, 1024), (384, 1024), (256, 384), None, None, None),
    (1.0, 2.0, (256, 1024), (384, 1024), (256, 384), None, None, None),
    (0.5, 1.5, (128, 2048), (512, 2048), (128, 512), None, None, None),
]

# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F8E4M3]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-2},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-3},
    InfiniDtype.BF16: {"atol": 0, "rtol": 5e-2},
    InfiniDtype.F8E4M3: {"atol": 0, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def pyop_ref_1x128_1x128(a, alpha, b, beta, a_descales, b_descales, C, bias):
    """Reference implementation for 1x128 @ 1x128 pattern"""
    assert a.dtype == torch.float8_e4m3fn, "a.dtype != torch.float8_e4m3fn"
    assert b.dtype == torch.float8_e4m3fn, "b.dtype != torch.float8_e4m3fn"
    assert a_descales.dtype == torch.float32, "a_descales.dtype != torch.float32"
    assert b_descales.dtype == torch.float32, "b_descales.dtype != torch.float32"
    a = a.to(torch.float32)
    a *= alpha
    a = a.to(torch.float8_e4m3fn)

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1], "K != b.shape[1]"

    a_scales_m = a_descales.shape[1]
    a_scales_k = a_descales.shape[0]
    b_scales_n = b_descales.shape[1]
    b_scales_k = b_descales.shape[0]

    assert a_scales_m == M, "a_scales_m != M"
    assert a_scales_k * 128 == K, "a_scales_k * 128 != K"
    assert b_scales_n == N, "b_scales_n != N"
    assert b_scales_k * 128 == K, "b_scales_k * 128 != K"

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    out = torch.zeros((M, N), dtype=torch.float32).to(a.device)

    for i in range(0, M):
        for j in range(0, N):
            for k in range(0, K, 128):
                out[i, j] += (
                    (a[i, k : k + 128] @ b[j, k : k + 128].t())
                    * a_descales[k // 128, i]
                    * b_descales[k // 128, j]
                )
    C *= beta
    C += out + bias


def pyop_ref_1x128_128x128(a, alpha, b, beta, a_descales, b_descales, C, bias):
    """Reference implementation for 1x128 @ 128x128 pattern"""
    assert a.dtype == torch.float8_e4m3fn, "a.dtype != torch.float8_e4m3fn"
    assert b.dtype == torch.float8_e4m3fn, "b.dtype != torch.float8_e4m3fn"
    assert a_descales.dtype == torch.float32, "a_descales.dtype != torch.float32"
    assert b_descales.dtype == torch.float32, "b_descales.dtype != torch.float32"
    a = a.to(torch.float32)
    a *= alpha
    a = a.to(torch.float8_e4m3fn)

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1], "K != b.shape[1]"

    a_scales_m = a_descales.shape[1]
    a_scales_k = a_descales.shape[0]
    b_scales_k = b_descales.shape[1]
    b_scales_n = b_descales.shape[0]

    assert a_scales_m == M, "a_scales_m != M"
    assert a_scales_k * 128 == K, "a_scales_k * 128 != K"
    assert b_scales_n * 128 == N, "b_scales_n * 128 != N"
    assert b_scales_k * 128 == K, "b_scales_k * 128 != K"

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    out = torch.zeros((M, N), dtype=torch.float32).to(a.device)

    for i in range(0, M):
        for j in range(0, N, 128):
            for k in range(0, K, 128):
                out[i, j : j + 128] += (
                    (a[i, k : k + 128] @ b[j : j + 128, k : k + 128].t())
                    * a_descales[k // 128, i]
                    * b_descales[j // 128, k // 128]
                )
    C *= beta
    C += out + bias


def pyop_ref_128x128_1x128(a, alpha, b, beta, a_descales, b_descales, C, bias):
    """Reference implementation for 128x128 @ 1x128 pattern"""
    assert a.dtype == torch.float8_e4m3fn, "a.dtype != torch.float8_e4m3fn"
    assert b.dtype == torch.float8_e4m3fn, "b.dtype != torch.float8_e4m3fn"
    assert a_descales.dtype == torch.float32, "a_descales.dtype != torch.float32"
    assert b_descales.dtype == torch.float32, "b_descales.dtype != torch.float32"
    a = a.to(torch.float32)
    a *= alpha
    a = a.to(torch.float8_e4m3fn)

    M, N, K = a.shape[0], b.shape[0], a.shape[1]
    assert K == b.shape[1], "K != b.shape[1]"

    a_scales_m = a_descales.shape[0]
    a_scales_k = a_descales.shape[1]
    b_scales_k = b_descales.shape[0]
    b_scales_n = b_descales.shape[1]

    assert a_scales_m * 128 == M, "a_scales_m * 128 != M"
    assert a_scales_k * 128 == K, "a_scales_k * 128 != K"
    assert b_scales_n == N, "b_scales_n != N"
    assert b_scales_k * 128 == K, "b_scales_k * 128 != K"

    a = a.to(torch.float32)
    b = b.to(torch.float32)

    out = torch.zeros((M, N), dtype=torch.float32).to(a.device)

    for i in range(0, M, 128):
        for j in range(0, N):
            for k in range(0, K, 128):
                out[i : i + 128, j] += (
                    (a[i : i + 128, k : k + 128] @ b[j, k : k + 128].t())
                    * a_descales[i // 128, k // 128]
                    * b_descales[k // 128, j]
                )
    C *= beta
    C += out + bias


# The argument list should be (lib, handle, torch_device, <param list>, dtype)
# The <param list> should keep the same order as the one specified in _TEST_CASES

# A B C D BIAS
FP8_SUPPORT_COMBINES = [
    [
        InfiniDtype.F8E4M3,
        InfiniDtype.F8E4M3,
        InfiniDtype.F16,
        InfiniDtype.F16,
        InfiniDtype.F16,
    ],
    [
        InfiniDtype.F8E4M3,
        InfiniDtype.F8E4M3,
        InfiniDtype.BF16,
        InfiniDtype.BF16,
        InfiniDtype.BF16,
    ],
]


def test(
    handle,
    device,
    alpha,
    beta,
    a_shape,
    b_shape,
    c_shape,
    a_stride=None,
    b_stride=None,
    c_stride=None,
    dtype=InfiniDtype.F16,
    sync=None,
):

    for precision in range(0, len(FP8_SUPPORT_COMBINES)):
        print(
            f"Testing FP8 Linear BlockWise on {InfiniDeviceNames[device]} with alpha:{alpha}, beta:{beta},"
            f" a_shape:{a_shape}, b_shape:{b_shape}, c_shape:{c_shape},"
            f" a_stride:{a_stride}, b_stride:{b_stride}, c_stride:{c_stride}"
            f" input_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][0]]}, output_dtype:{InfiniDtypeNames[FP8_SUPPORT_COMBINES[precision][-1]]}"
        )
        for a_1d, b_1d in [(True, False), (False, True)]:
            # 1x128 @ 1x128 的乘法非常慢，所以先注掉，等需要的时候再加上
            # for a_1d, b_1d in [(True, False), (False, True), (True, True)]:
            # Initialize tensors
            a = TestTensor(a_shape, None, InfiniDtype.F16, device)
            if a.dt != FP8_SUPPORT_COMBINES[precision][0]:
                a.convert_pricesion(FP8_SUPPORT_COMBINES[precision][0])
            b = TestTensor(b_shape, None, InfiniDtype.F16, device)
            if b.dt != FP8_SUPPORT_COMBINES[precision][1]:
                b.convert_pricesion(FP8_SUPPORT_COMBINES[precision][1])
            c = TestTensor(c_shape, None, FP8_SUPPORT_COMBINES[precision][2], device)
            d = TestTensor(
                c_shape, None, FP8_SUPPORT_COMBINES[precision][3], device, mode="zeros"
            )

            bias = (
                torch.ones(
                    c_shape,
                    device=torch_device_map[device],
                    dtype=c.torch_tensor().dtype,
                )
                * 0.6
            )

            if a_1d and not b_1d:
                scale_a_ = TestTensor(
                    (int(a_shape[0] / 128), int(a_shape[1] / 128)),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                scale_b_ = TestTensor(
                    (int(b_shape[1] / 128), int(b_shape[0])),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                pyop_ref_128x128_1x128(
                    a.torch_tensor(),
                    alpha,
                    b.torch_tensor(),
                    beta,
                    scale_a_.torch_tensor(),
                    scale_b_.torch_tensor(),
                    c.torch_tensor(),
                    bias,
                )
            elif not a_1d and b_1d:
                scale_a_ = TestTensor(
                    (int(a_shape[1] / 128), int(a_shape[0])),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                scale_b_ = TestTensor(
                    (int(b_shape[0] / 128), int(b_shape[1] / 128)),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                pyop_ref_1x128_128x128(
                    a.torch_tensor(),
                    alpha,
                    b.torch_tensor(),
                    beta,
                    scale_a_.torch_tensor(),
                    scale_b_.torch_tensor(),
                    c.torch_tensor(),
                    bias,
                )
            elif a_1d and b_1d:
                scale_a_ = TestTensor(
                    (int(a_shape[1] / 128), int(a_shape[0])),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                scale_b_ = TestTensor(
                    (int(b_shape[1] / 128), int(b_shape[0])),
                    None,
                    InfiniDtype.F32,
                    device,
                )
                pyop_ref_1x128_1x128(
                    a.torch_tensor(),
                    alpha,
                    b.torch_tensor(),
                    beta,
                    scale_a_.torch_tensor(),
                    scale_b_.torch_tensor(),
                    c.torch_tensor(),
                    bias,
                )
            else:
                raise Exception("不支持scale均为二维块量化的情况")

            if sync is not None:
                sync()

            descriptor = infiniopOperatorDescriptor_t()
            check_error(
                LIBINFINIOP.infiniopCreateLinearDescriptor(
                    handle,
                    ctypes.byref(descriptor),
                    d.descriptor,
                    a.descriptor,
                    b.descriptor,
                    c.descriptor,
                )
            )

            # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel
            for tensor in [a, b, c]:
                tensor.destroy_desc()

            # Get workspace size and create workspace
            workspace_size = c_uint64(33554432)
            check_error(
                LIBINFINIOP.infiniopGetLinearWorkspaceSize(
                    descriptor, ctypes.byref(workspace_size)
                )
            )
            workspace = TestWorkspace(workspace_size.value, device)
            bias_ = bias.clone()

            # Execute infiniop gemm operator
            def lib_linear():
                check_error(
                    LIBINFINIOP.infiniopLinear(
                        descriptor,
                        alpha,
                        a.data(),
                        scale_a_.data(),
                        b.data(),
                        scale_b_.data(),
                        beta,
                        c.data(),
                        None,
                        bias_.data_ptr(),
                        d.data(),
                        None,
                        True,  # block_wise
                        a_1d,  # a_1d
                        b_1d,  # b_1d
                        workspace.data(),
                        0,
                        None,
                    )
                )

            lib_linear()

            # Validate results
            atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)

            # if DEBUG:
            #     debug(c.actual_tensor(), f, atol=atol, rtol=rtol)
            assert torch.allclose(
                d.actual_tensor(),
                c.torch_tensor().to(d.torch_tensor().dtype),
                atol=atol,
                rtol=rtol,
            )

            # Profiling workflow
            if PROFILE:
                raise NotImplementedError
                # fmt: off
                # profile_operation("PyTorch", lambda: torch_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
                # profile_operation("    lib", lambda: lib_linear(), device, NUM_PRERUN, NUM_ITERATIONS)
                # fmt: on
            check_error(LIBINFINIOP.infiniopDestroyLinearDescriptor(descriptor))


# ==============================================================================
#  Main Execution
# ==============================================================================
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
