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
    to_torch_dtype,
)
from enum import Enum, auto
import itertools

# ==============================================================================
#  Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
# Test configurations

BLOCK_SIZE = [[128, 128]]
M_list = [1, 7, 83, 512, 2048]
N_list = [128, 512, 1024, 4096, 7748, 13824]
K_list = [256, 4096, 5120, 3884, 13824]
SEEDS = 0

def to_iter(x):
    return x if isinstance(x, (list, tuple)) else (x,)


_TEST_CASES = list(
    itertools.product(
        to_iter(M_list),
        to_iter(K_list),
        to_iter(N_list),
        to_iter(BLOCK_SIZE),
    )
)


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.F16]


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def native_w8a16_block_int8_matmul(
    A,
    B,
    Bs,
    block_size,
    output_dtype: torch.float16,
) -> torch.Tensor:
    """Matrix multiplication with block-wise quantization using native torch."""
    A = A.to(torch.float32)
    B = B.to(torch.float32)
    assert A.shape[-1] == B.shape[-1]
    assert B.ndim == 2 and B.is_contiguous() and Bs.ndim == 2
    assert len(block_size) == 2
    block_n, block_k = block_size[0], block_size[1]

    M = A.numel() // A.shape[-1]
    N, K = B.shape
    origin_C_shape = A.shape[:-1] + (N, )
    A = A.reshape(M, A.shape[-1])
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k
    assert n_tiles == Bs.shape[0]
    assert k_tiles == Bs.shape[1]

    C_shape = (M, N)
    C = torch.zeros(C_shape, dtype=torch.float32, device=A.device)

    A_tiles = [
        A[:, i * block_k:min((i + 1) * block_k, K)] for i in range(k_tiles)
    ]
    B_tiles = [[
        B[j * block_n:min((j + 1) * block_n, N),
          i * block_k:min((i + 1) * block_k, K), ] for i in range(k_tiles)
    ] for j in range(n_tiles)]
    C_tiles = [
        C[:, j * block_n:min((j + 1) * block_n, N)] for j in range(n_tiles)
    ]

    for i in range(k_tiles):
        for j in range(n_tiles):
            a = A_tiles[i]
            b = B_tiles[j][i]
            c = C_tiles[j]
            s = Bs[j][i]
            c[:, :] += torch.matmul(a, b.t()) * s

    C = C.reshape(origin_C_shape).to(output_dtype)
    return C


def native_w8a16_block_fp8_matmul(
    A,
    B,
    Bs,
    block_size,
    output_dtype: torch.float16,
) -> torch.Tensor:
    return native_w8a16_block_int8_matmul(A, B, Bs, block_size, output_dtype)


def test_w8a8_block_fp8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    fp8_info = torch.finfo(torch.float8_e4m3fn)
    fp8_max, fp8_min = fp8_info.max, fp8_info.min

    A_fp32 = (torch.rand(M, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    #A_fp32 = A_fp32.fill_(1)
    A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * fp8_max
    #B_fp32 = B_fp32.fill_(1)
    B_fp8 = B_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    #As = As.fill_(1)
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale
    #Bs = Bs.fill_(1.5)
    #ref_out = native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size,
    #                                       out_dtype)
    ref_out = native_w8a16_block_fp8_matmul(A_fp32.to(torch.bfloat16), B_fp8, Bs, block_size, out_dtype)
    #out = w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    
    B_fp8_T = B_fp8.t()
    #print('B_fp8_T', B_fp8_T.size(), B_fp8_T)
    
    Bs_T = Bs
    quant_type = 3
    bit = 8
    return ref_out, A_fp32.to(torch.bfloat16), B_fp8_T, Bs_T, Bs_T, quant_type, bit
    

def test_w8a8_block_int8_matmul(M, N, K, block_size, out_dtype, seed):
    torch.manual_seed(seed)
    factor_for_scale = 1e-2
    int8_info = torch.iinfo(torch.int8)
    int8_max, int8_min = int8_info.max, int8_info.min

    A_fpb16 = torch.rand(M, K, dtype=torch.float32) / 10


    #A_fp32 = A_fp32.fill_(1)
    #A_fp8 = A_fp32.clamp(min=fp8_min, max=fp8_max).to(torch.float8_e4m3fn)

    B_fp32 = (torch.rand(N, K, dtype=torch.float32) - 0.5) * 2 * int8_max
    #B_fp32 = B_fp32.fill_(1)
    B_int8 = B_fp32.clamp(min=int8_min, max=int8_max).to(torch.int8)

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    A_fpb16 =A_fpb16.to(torch.float16)

    #As = torch.rand(M, k_tiles, dtype=torch.float32) * factor_for_scale
    #As = As.fill_(1)
    Bs = torch.rand(n_tiles, k_tiles, dtype=torch.float32) * factor_for_scale
    #Bs = Bs.fill_(1.5)
    #ref_out = native_w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)

    ref_out = native_w8a16_block_fp8_matmul(A_fpb16, B_int8, Bs, block_size, out_dtype)
    #a_q, a_s = native_per_token_group_quant_int8(A_fpb16, block_k)
    #ref_out = native_w8a8_block_int8_matmul(a_q, B_int8, a_s, Bs, block_size, output_dtype=A_fpb16.dtype)
    ##out = w8a8_block_fp8_matmul(A_fp8, B_fp8, As, Bs, block_size, out_dtype)
    #print('Bs', Bs.size(), Bs.dtype)
    quant_type = 3
    bit = 8
    return ref_out, A_fpb16, B_int8, Bs, Bs, quant_type, bit


def test_int8(
    handle,
    device,
    M,
    K,
    N,
    block_size,
    dtype=InfiniDtype.BF16,
    sync=None,
):

    print(
        f"Testing int8 Gptq Qyblas Gemm on {InfiniDeviceNames[device]} with M-K-N:{M, K, N}, block_size:{block_size}, dtype:{InfiniDtypeNames[dtype]}"
    )
    out_dtype = to_torch_dtype(dtype)
    ans, a, b_orig, b_scales, b_zeros, quant_type, bit = test_w8a8_block_int8_matmul(M, N, K, block_size, out_dtype, SEEDS)
    b = b_orig.t()
    
    A = TestTensor(
        a.shape,
        a.stride(),
        InfiniDtype.F16,
        device,
        mode="manual",
        set_tensor=a,
    )
    B_orig = TestTensor(
        b_orig.shape,
        b_orig.stride(),
        InfiniDtype.I8,
        device,
        mode="manual",
        set_tensor=b_orig,
    )
    B = TestTensor(
        b.shape,
        b.stride(),
        InfiniDtype.I8,
        device,
        mode="manual",
        set_tensor=b,
    )
    b_scales = TestTensor(
        b_scales.shape,
        b_scales.stride(),
        InfiniDtype.F32,
        device,
        mode="manual",
        set_tensor=b_scales,
    )
    b_zeros = TestTensor(
        b_zeros.shape,
        b_zeros.stride(),
        InfiniDtype.F32,
        device,
        mode="manual",
        set_tensor=b_zeros,
    )
    out = TestTensor(
        ans.shape,
        None,
        dtype,
        device,
    )
    
    print("a: ", A.torch_tensor().shape, A.torch_tensor().stride(), A.torch_tensor().dtype)
    print("b: ", B.torch_tensor().shape, B.torch_tensor().stride(), B.torch_tensor().dtype)
    print("scales: ", b_scales.torch_tensor().shape, b_scales.torch_tensor().dtype)
    print("zeros: ", b_zeros.torch_tensor().shape, b_zeros.torch_tensor().dtype)
    print("out: ", out.torch_tensor().shape, out.torch_tensor().dtype)
    if sync is not None:
        sync()


    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGptqQyblasGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            A.descriptor,
            B.descriptor,
            b_scales.descriptor,
            b_zeros.descriptor,
        )
    )

    # Invalidate the shape and strides in the descriptor to prevent them from being directly used by the kernel

    for tensor in [out, A, B, b_scales, b_zeros]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGptqQyblasGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, A.device)

    def lib_gptq_qyblas_gemm():
        check_error(
            LIBINFINIOP.infiniopGptqQyblasGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                A.data(),
                B.data(),
                b_scales.data(),
                b_zeros.data(),
                quant_type,
                bit,
                None,
            )
        )

    lib_gptq_qyblas_gemm()

    if sync is not None:
        sync()

    tmpa = out.torch_tensor().to(torch.float32).detach().to('cpu').numpy().flatten()
    tmpb = ans.to(torch.float32).to('cpu').detach().numpy().flatten()
    
    atol = max(abs(tmpa - tmpb))

    rtol = atol / (max(abs(tmpb)) + 1e-8)


    print("absolute error:%.4e"%(atol))
    print("relative error:%.4e"%(rtol))
    print(out.torch_tensor().device, ans.device)
    # print(out.torch_tensor())
    # print(ans)
    ans = ans.to(out.torch_tensor().device)
    rel_diff = (torch.mean(
        torch.abs(out.torch_tensor().to(torch.float32) - ans.to(torch.float32))) /
                torch.mean(torch.abs(ans.to(torch.float32))))
    print(rel_diff)
    assert rel_diff < 0.05
    

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: native_w8a16_block_fp8_matmul(A.torch_tensor(), B_orig.torch_tensor(), b_scales.torch_tensor(), block_size, out_dtype), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gptq_qyblas_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGptqQyblasGemmDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test_int8, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
