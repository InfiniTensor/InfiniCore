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
from libinfiniop.scalar_type import scalar_types, ScalarType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

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

_TEST_CASES_BIT = [
    # M , K, N, group_size, bit
    (128, 128, 128, 128, 4),
    (32768, 3584, 4608, 128, 4),
    (32768, 3584, 4608, 128, 8),
]


# Data types used for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16]

_TENSOR_DTYPES_BIT = [InfiniDtype.BF16, InfiniDtype.F16]


DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]


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


def _gguf_quantize_weights(w: torch.Tensor,
                     quant_type: ScalarType,
                     group_size: Optional[int],
                     zero_points: bool = False,
                     ref_zero_points_after_scales: bool = False,
                     need_weight_ref: bool = True):
    assert quant_type.is_integer(), \
        "Floating point quantization may work but has not been tested"
    assert not zero_points or group_size is not None, \
        "to have group zero points, group_size must be provided "\
        "(-1 group_size is channelwise)"

    orig_device = w.device
    orig_type = w.dtype
    size_k, size_n = w.shape

    assert w.is_floating_point(), "w must be float"

    if group_size == -1:
        group_size = size_k

    # Reshape to [groupsize, -1]
    if group_size is not None and group_size < size_k:
        w = w.reshape((-1, group_size, size_n))
        w = w.permute(1, 0, 2)
        w = w.reshape((group_size, -1))

    # Compute scale for each group
    max_val = torch.max(w, 0, keepdim=True).values
    min_val = torch.min(w, 0, keepdim=True).values

    max_q_val = quant_type.max()
    min_q_val = quant_type.min()

    w_s = torch.Tensor([1.0]).to(w.device)  # unscaled case
    maybe_w_zp = None
    if group_size is not None:
        if zero_points:
            assert not quant_type.is_signed() and quant_type.max() > 0
            w_s = (max_val - min_val).clamp(min=1e-5) / quant_type.max()
            maybe_w_zp = torch.round(torch.abs(min_val / w_s)) \
                .clamp(min_q_val, max_q_val).int()
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)))

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
    if need_weight_ref:
        if ref_zero_points_after_scales and maybe_w_zp is not None:
            w_ref = w_q.to(orig_type) * w_s - maybe_w_zp.to(orig_type) * w_s
        else:
            w_ref = (w_q - (maybe_w_zp if zero_points else 0)).to(orig_type) * w_s

    if quant_type.has_bias():
        w_q += quant_type.bias

    # Restore original shapes
    if group_size is not None and group_size < size_k:

        def reshape_w(w):
            w = w.reshape((group_size, -1, size_n))
            w = w.permute(1, 0, 2)
            w = w.reshape((size_k, size_n)).contiguous()
            return w

        w_q = reshape_w(w_q)
        if need_weight_ref:
            w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device) if need_weight_ref else None,
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )

def gguf_quantize_weights(w: torch.Tensor,
                          group_size: int,
                          zero_points: bool = False,
                          need_weight_ref: bool = False,
                          bits: int = 4,
                          ref_zero_points_after_scales: bool = False,
                          params_dtype: torch.dtype = torch.float16):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert group_size in SUPPORTED_GROUP_SIZES + [
        size_k
    ], f"Unsupported groupsize = {group_size}"

    w_ref, w_q, w_s, w_z = _gguf_quantize_weights(w, quant_type=scalar_types.uint4 if bits == 4 else scalar_types.uint8,
                                          group_size=group_size,
                                          zero_points=zero_points,
                                          need_weight_ref=need_weight_ref,
                                          ref_zero_points_after_scales=ref_zero_points_after_scales)

    if zero_points:
        w_z = w_z.to(params_dtype)

    w_q = w_q.to(torch.uint8)

    return w_ref, w_q, w_s, w_z

def gguf_linear_quantize_weights(w: torch.Tensor,
                          group_size: int,
                          zero_points: bool = False,
                          need_weight_ref: bool = False,
                          bits: int =4,
                          params_dtype: torch.dtype = torch.float16):
    w_ref, w_q, w_s, w_z = gguf_quantize_weights(
        w=w,
        group_size=group_size,
        zero_points=zero_points,
        need_weight_ref=need_weight_ref,
        bits=bits,
        ref_zero_points_after_scales=False,
        params_dtype=params_dtype,
    )

    if bits == 4:
        w_q = (w_q[:,1::2] << 4) | w_q[:, ::2]
        w_q = w_q.reshape(w_q.shape[0]//2, -1) # This step is to match the parameters of the dlblasGemmExV2

    return w_ref, w_q, w_s, w_z



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


def test_bit(
    handle,
    device,
    M,
    K,
    N,
    group_size,
    bit,
    dtype=InfiniDtype.BF16,
    sync=None,
):

    print(
        f"Testing Gptq Qyblas Gemm on {InfiniDeviceNames[device]} with M-K-N:{M, K, N}, group_size:{group_size}, bit:{bit}, dtype:{InfiniDtypeNames[dtype]}"
    )
    quant_type = 0
    bit = 4

    k_tiles = (K + group_size - 1) // group_size

    A = TestTensor(
        (M, K),
        None,
        dtype,
        device,
    )
    B_orig = TestTensor(
        (K, N),
        None,
        dtype,
        device,
    )
    w_ref, w_q, w_s, w_z = gguf_linear_quantize_weights(B_orig.torch_tensor(),
                                                 group_size=group_size,
                                                 zero_points=True,
                                                 need_weight_ref=True,
                                                 bits=bit,
                                                 params_dtype=to_torch_dtype(dtype))                                           
    
    B = TestTensor(
        w_q.shape,
        w_q.stride(),
        InfiniDtype.U8,
        device,
        mode="manual",
        set_tensor=w_q,
    )

    
    b_scales = TestTensor(
        w_s.shape,
        w_s.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=w_s,
    )

    b_zeros = TestTensor(
        w_z.shape,
        w_z.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=w_z,
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
    
    atol, rtol = 2e-2, 2e-2
    if bit == 8:
        atol, rtol = 2e-2, 0
    else:
        atol, rtol = 2e-2, 2e-2
    ans = torch.matmul(A.torch_tensor(), w_ref.to(A.torch_tensor().device))
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: torch.matmul(A.torch_tensor(), w_ref.to(A.torch_tensor().device)), device, NUM_PRERUN, NUM_ITERATIONS)
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
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)
    for device in get_test_devices(args):
        test_operator(device, test_bit, _TEST_CASES_BIT, _TENSOR_DTYPES_BIT)

    print("\033[92mTest passed!\033[0m")
