import torch
import numpy
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
_TEST_CASES = [
    # M, K, N, use_exllama, quant_bit, group_size
    (1, 2048, 2048, True, 4, 128),
    (1, 2048, 4096, False, 4, 128),
    (1, 4096, 2048, False, 4, 128),
    (8, 2048, 2048, False, 4, 128),
    (8, 2048, 4096, False, 4, 128),
    (8, 4096, 2048, False, 4, 128),
    (128, 2048, 2048, False, 4, 128),
    (128, 2048, 4096, False, 4, 128),
    (128, 4096, 2048, False, 4, 128),
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 5e-2},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def pack_cols(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_n % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k, size_n // pack_factor), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[:, i::pack_factor] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    q_res = q_res.contiguous()

    return q_res


def pack_rows(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    assert q_w.shape == (size_k, size_n)

    pack_factor = get_pack_factor(num_bits)
    assert size_k % pack_factor == 0

    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_res = numpy.zeros((size_k // pack_factor, size_n), dtype=numpy.uint32)

    for i in range(pack_factor):
        q_res |= q_w[i::pack_factor, :] << num_bits * i

    q_res = torch.from_numpy(q_res.astype(numpy.int32)).to(orig_device)
    return q_res


def torch_dequantize(q_weight, q_zeros, scales, g_idx, use_shuffle, bit, K, N):
    assert bit == 4, "Reference dequantization only supports 4-bit"
    group_size = K // scales.shape[0]
    pack_factor = 32 // bit

    # unpack q_weight: (K//pack_factor, N) -> (K, N)
    unpacked_q_weight = torch.empty(
        q_weight.shape[0] * pack_factor,
        q_weight.shape[1],
        dtype=torch.uint8,
        device=q_weight.device,
    )
    for i in range(pack_factor):
        unpacked_q_weight[i::pack_factor, :] = (q_weight >> (i * 4)) & 0x0F

    # unpack q_zeros: (num_groups, N//pack_factor) -> (num_groups, N)
    unpacked_q_zeros = torch.empty(
        q_zeros.shape[0],
        q_zeros.shape[1] * pack_factor,
        dtype=torch.uint8,
        device=q_zeros.device,
    )
    for i in range(pack_factor):
        unpacked_q_zeros[:, i::pack_factor] = (q_zeros >> (i * 4)) & 0x0F

    unpacked_q_zeros += 1
    unpacked_q_zeros = unpacked_q_zeros.to(scales.dtype)

    scale_zeros = unpacked_q_zeros * scales  # (num_groups, N)

    current_g_idx = torch.tensor(
        [i // group_size for i in range(K)], dtype=torch.int32, device=q_weight.device
    )

    scale_mat = scales[current_g_idx]  # (K, N)
    scale_zeros_mat = scale_zeros[current_g_idx]  # (K, N)

    # dequant: weight * scale - scale_zeros
    dequantized_b = unpacked_q_weight.to(scales.dtype) * scale_mat - scale_zeros_mat

    return dequantized_b.reshape(K, N)


def torch_gptq_gemm(
    a, b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit
):
    K, N = a.shape[1], b_q_weight.shape[1]

    b_dequant = torch_dequantize(
        b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx, use_shuffle, bit, K, N
    )
    c = torch.matmul(a, b_dequant)
    return c


def test(
    handle,
    device,
    M,
    K,
    N,
    use_exllama,
    quant_bit,
    group_size,
    dtype=InfiniDtype.F16,
    sync=None,
):

    print(
        f"Testing Gptq Gemm on {InfiniDeviceNames[device]} with M-K-N:{M, K, N}, use_exllama:{use_exllama}, quant_bit:{quant_bit}, group_size:{group_size}, dtype:{InfiniDtypeNames[dtype]}"
    )
    b_fp = TestTensor((K, N), None, dtype, device)

    assert K % group_size == 0, "K must be divisible by group_size"
    num_groups = K // group_size
    use_shuffle = use_exllama

    if use_shuffle:
        print(f"not support use_shuffle:{use_shuffle}")
        return
    else:
        g_idx = torch.tensor(
            [i // group_size for i in range(K)],
            dtype=torch.int32,
            device=b_fp.torch_tensor().device,
        )
        b_shuffled = b_fp.torch_tensor()[g_idx]

    b_grouped = b_shuffled.reshape(num_groups, group_size, N)

    b_max = torch.max(b_grouped, dim=1, keepdim=True)[0]
    b_min = torch.min(b_grouped, dim=1, keepdim=True)[0]

    scales = (b_max - b_min) / (2**quant_bit - 1)
    scales = scales.clamp(min=1e-6)

    zeros_float = (-b_min / scales).round()

    q_b = (
        (b_grouped / scales + zeros_float)
        .round()
        .clamp(0, 2**quant_bit - 1)
        .to(torch.uint8)
    )

    q_zeros_unpacked = zeros_float.to(torch.uint8) - 1

    b_q_weight = pack_rows(q_b.reshape(K, N), quant_bit, K, N)

    q_zeros_unpacked = q_zeros_unpacked.reshape(num_groups, N)
    b_gptq_qzeros = pack_cols(q_zeros_unpacked, quant_bit, num_groups, N)
    b_gptq_scales = scales.squeeze(1)

    A = TestTensor((M, K), None, dtype, device)
    C = TestTensor((M, N), None, dtype, device)

    B = TestTensor(
        b_q_weight.shape,
        b_q_weight.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=b_q_weight,
    )
    b_scales = TestTensor(
        b_gptq_scales.shape,
        b_gptq_scales.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=b_gptq_scales,
    )
    b_zeros = TestTensor(
        b_gptq_qzeros.shape,
        b_gptq_qzeros.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=b_gptq_qzeros,
    )
    b_g_idx = TestTensor(
        (K,), g_idx.stride(), InfiniDtype.I32, device, mode="manual", set_tensor=g_idx
    )

    if sync is not None:
        sync()

    ans = torch_gptq_gemm(
        A.torch_tensor(),
        B.torch_tensor(),
        b_zeros.torch_tensor(),
        b_scales.torch_tensor(),
        b_g_idx.torch_tensor(),
        use_shuffle,
        quant_bit,
    )

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGptqGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            C.descriptor,
            A.descriptor,
            B.descriptor,
            b_scales.descriptor,
            b_zeros.descriptor,
            b_g_idx.descriptor,
            use_exllama,
            quant_bit,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    for tensor in [C, A, B, b_scales, b_zeros, b_g_idx]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGptqGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, A.device)

    def lib_gptq_gemm():
        check_error(
            LIBINFINIOP.infiniopGptqGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                C.data(),
                A.data(),
                B.data(),
                b_scales.data(),
                b_zeros.data(),
                b_g_idx.data(),
                None,
            )
        )

    lib_gptq_gemm()

    if sync is not None:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(C.actual_tensor(), ans, atol=atol, rtol=rtol)

    assert torch.allclose(C.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: gptq_gemm_torch(A.torch_tensor(), B.torch_tensor(), b_scales.torch_tensor(), b_zeros.torch_tensor(), b_g_idx.torch_tensor(), group_size, quant_bit), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gptq_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGptqGemmDescriptor(descriptor))


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
