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
M_list = [1, 7]#, 83, 512, 2048]
N_list = [128, 512]#, 1024, 4096, 7748, 13824]
K_list = [256, 4096]#, 5120, 3884, 13824]
_WEIGHT_DTYPES = [InfiniDtype.I8]

SEEDS = 0

def to_iter(x):
    return x if isinstance(x, (list, tuple)) else (x,)


_TEST_CASES = list(
    itertools.product(
        to_iter(M_list),
        to_iter(K_list),
        to_iter(N_list),
        to_iter(BLOCK_SIZE),
        to_iter(_WEIGHT_DTYPES),
    )
)


_TEST_CASES_W4 = [(32768, 3584, 4608, [128, 128], InfiniDtype.U8),]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

_TENSOR_DTYPES_W4 = [InfiniDtype.F16]


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


def test(
    handle,
    device,
    M,
    K,
    N,
    block_size,
    weight_dtype=InfiniDtype.I8,
    dtype=InfiniDtype.BF16,
    sync=None,
):

    print(
        f"Testing int8 Gptq Qyblas Gemm on {InfiniDeviceNames[device]} with M-K-N:{M, K, N}, block_size:{block_size}, weight dtype:{InfiniDtypeNames[weight_dtype]}, dtype:{InfiniDtypeNames[dtype]}"
    )
    quant_type = 3
    bit = 8

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    A = TestTensor(
        (M, K),
        None,
        dtype,
        device,
    )
    if weight_dtype == InfiniDtype.I8:
        _info = torch.iinfo(torch.int8)
    elif weight_dtype == InfiniDtype.U8:
        _info = torch.iinfo(torch.uint8)
    elif weight_dtype == InfiniDtype.F8:
        _info = torch.iinfo(float8_e4m3fn)
    B_orig = TestTensor(
        (N, K),
        None,
        weight_dtype,
        device,
        randint_low=_info.min,
        randint_high=_info.max,
    )
    B_torch = B_orig.torch_tensor().t()
    B = TestTensor(
        (K, N),
        B_torch.stride(),
        weight_dtype,
        device,
        mode="manual",
        set_tensor=B_torch,
    )
    
    b_scales = TestTensor(
        (n_tiles, k_tiles),
        None,
        InfiniDtype.F32,
        device,
    )

    b_zeros = TestTensor(
        (n_tiles, k_tiles),
        None,
        InfiniDtype.F32,
        device,
        mode="zeros",
    )
    
    out = TestTensor(
        (M, N),
        None,
        dtype,
        device,
        mode="zeros",
    )

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

    out_dtype = to_torch_dtype(dtype)
    ans = native_w8a16_block_int8_matmul(A.torch_tensor(), B_orig.torch_tensor(), b_scales.torch_tensor(), block_size, out_dtype)
    
    rel_diff = (torch.mean(
        torch.abs(out.actual_tensor().to(torch.float32) - ans.to(torch.float32))) /
                torch.mean(torch.abs(ans.to(torch.float32))))

    assert rel_diff < 0.05
    

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: native_w8a16_block_int8_matmul(A.torch_tensor(), B_orig.torch_tensor(), b_scales.torch_tensor(), block_size, out_dtype), device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lambda: lib_gptq_qyblas_gemm(), device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    check_error(LIBINFINIOP.infiniopDestroyGptqQyblasGemmDescriptor(descriptor))


def test_w4(
    handle,
    device,
    M,
    K,
    N,
    block_size,
    weight_dtype=InfiniDtype.I8,
    dtype=InfiniDtype.BF16,
    sync=None,
):
    print(
        f"Testing w4 Gptq Qyblas Gemm on {InfiniDeviceNames[device]} with M-K-N:{M, K, N}, block_size:{block_size}, weight dtype:{InfiniDtypeNames[weight_dtype]}, dtype:{InfiniDtypeNames[dtype]}"
    )
    quant_type = 0
    bit = 4

    block_n, block_k = block_size[0], block_size[1]
    n_tiles = (N + block_n - 1) // block_n
    k_tiles = (K + block_k - 1) // block_k

    A = TestTensor(
        (M, K),
        None,
        dtype,
        device,
    )
    if weight_dtype == InfiniDtype.I8:
        _info = torch.iinfo(torch.int8)
    elif weight_dtype == InfiniDtype.U8:
        _info = torch.iinfo(torch.uint8)
    elif weight_dtype == InfiniDtype.F8:
        _info = torch.iinfo(float8_e4m3fn)
    # B_orig = TestTensor(
    #     (N, K // 2),
    #     None,
    #     weight_dtype,
    #     device,
    #     randint_low=_info.min,
    #     randint_high=_info.max,
    # )
    # B_torch = B_orig.torch_tensor().t()
    # B = TestTensor(
    #     (K // 2, N),
    #     B_torch.stride(),
    #     weight_dtype,
    #     device,
    #     mode="manual",
    #     set_tensor=B_torch,
    # )

    B = TestTensor(
        (K // 2, N),
        None,
        weight_dtype,
        device,
        randint_low=_info.min,
        randint_high=_info.max,
    )
    
    b_scales = TestTensor(
        (k_tiles, N),
        None,
        dtype,
        device,
    )

    b_zeros = TestTensor(
        (k_tiles, N),
        None,
        dtype,
        device,
        mode="zeros",
    )
    
    out = TestTensor(
        (M, N),
        None,
        dtype,
        device,
        mode="zeros",
    )

    print("A", A.torch_tensor().shape, A.torch_tensor().dtype, A.torch_tensor().stride())
    print("B", B.torch_tensor().shape, B.torch_tensor().dtype, B.torch_tensor().stride())
    print("scales", b_scales.torch_tensor().shape, b_scales.torch_tensor().dtype, b_scales.torch_tensor().stride())
    print("zeros", b_zeros.torch_tensor().shape, b_zeros.torch_tensor().dtype, b_zeros.torch_tensor().stride())
    print("out", out.torch_tensor().shape, out.torch_tensor().dtype, out.torch_tensor().stride())

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

    out_dtype = to_torch_dtype(dtype)
    ans = native_w8a16_block_int8_matmul(A.torch_tensor(), B_orig.torch_tensor(), b_scales.torch_tensor(), block_size, out_dtype)
    
    rel_diff = (torch.mean(
        torch.abs(out.actual_tensor().to(torch.float32) - ans.to(torch.float32))) /
                torch.mean(torch.abs(ans.to(torch.float32))))

    assert rel_diff < 0.05
    

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: native_w8a16_block_int8_matmul(A.torch_tensor(), B_orig.torch_tensor(), b_scales.torch_tensor(), block_size, out_dtype), device, NUM_PRERUN, NUM_ITERATIONS)
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

    # for device in get_test_devices(args):
    #     test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    for device in get_test_devices(args):
        test_operator(device, test_w4, _TEST_CASES_W4, _TENSOR_DTYPES_W4)

    print("\033[92mTest passed!\033[0m")
