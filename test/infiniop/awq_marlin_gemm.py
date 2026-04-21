import torch
import ctypes
from ctypes import c_uint64
from libinfiniop import (
    LIBINFINIOP,
    TestTensor,
    TestWorkspace,
    get_test_devices,
    check_error,
    test_operator,
    get_args,
    debug,
    get_tolerance,
    profile_operation,
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    to_torch_dtype,
)
import itertools
from libinfiniop.scalar_type import scalar_types, ScalarType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np


_TEST_CASES_SUBSET_INPUT = [
    # (size_m, size_k, size_n, group_size, quant_type)
    (32, 1024, 2048, 128, scalar_types.uint4b8),
]

_TEST_CASES_WITH_BIAS = [
    # (size_m, size_k, size_n, group_size, quant_type)
    (1, 1024, 2048, 128, scalar_types.uint4b8),
    (256, 1024, 2048, 128, scalar_types.uint4b8),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16]

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def rand_data(shape, dtype, device):
    return torch.randn(shape, dtype=dtype, device=device)

def get_scale_perms():
    scale_perm: list[int] = []
    for i in range(8):
        scale_perm.extend([i + 8 * j for j in range(8)])
    scale_perm_single: list[int] = []
    for i in range(4):
        scale_perm_single.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return scale_perm, scale_perm_single

def marlin_permute_bias(s: torch.Tensor) -> torch.Tensor:
    origin_shape = s.shape
    _, scale_perm_single = get_scale_perms()
    s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    return s.reshape(*origin_shape).contiguous()

def quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int | None,
    zero_points: bool = False,
    ref_zero_points_after_scales: bool = False,
):
    assert quant_type.is_integer(), (
        "Floating point quantization may work but has not been tested"
    )
    assert not zero_points or group_size is not None, (
        "to have group zero points, group_size must be provided "
        "(-1 group_size is channelwise)"
    )

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
            maybe_w_zp = (
                torch.round(torch.abs(min_val / w_s)).clamp(min_q_val, max_q_val).int()
            )
        else:
            # If the bias is such that there are no possible negative/positive
            #  values, set the max value to inf to avoid divide by 0
            w_s = torch.max(
                abs(max_val / (max_q_val if max_q_val != 0 else torch.inf)),
                abs(min_val / (min_q_val if min_q_val != 0 else torch.inf)),
            )

    # Quantize
    w_q = torch.round(w / w_s).int() + (maybe_w_zp if zero_points else 0)
    w_q = torch.clamp(w_q, min_q_val, max_q_val)

    # Compute ref (dequantized)
    # For some kernels (namely Machete) the zero-points are applied after the
    # scales are applied, for this case computing the reference in similar way
    # allows us to use tighter error tolerances in our unit tests.
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
        w_ref = reshape_w(w_ref)
        w_s = w_s.reshape((-1, size_n)).contiguous()

    if maybe_w_zp is not None:
        maybe_w_zp = maybe_w_zp.reshape((-1, size_n)).contiguous()
        maybe_w_zp = maybe_w_zp.to(device=orig_device)

    return (
        w_ref.to(device=orig_device),
        w_q.to(device=orig_device),
        w_s if group_size is not None else None,
        maybe_w_zp,
    )

SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]

def permute_rows(
    q_w: torch.Tensor,
    w_ref: torch.Tensor,
    group_size: int,
    test_perm: torch.Tensor | None = None,
):
    assert q_w.shape == w_ref.shape

    orig_device = q_w.device
    k_size, _ = q_w.shape

    g_idx = torch.zeros((k_size,), dtype=torch.int32)
    for i in range(k_size):
        g_idx[i] = i // group_size

    # Simulate act_order by doing a random permutation on K
    rand_perm = test_perm if test_perm is not None else torch.randperm(k_size)

    g_idx = g_idx[rand_perm].contiguous()
    q_w = q_w[rand_perm, :].contiguous()
    w_ref = w_ref[rand_perm, :].contiguous()

    return (
        w_ref.to(device=orig_device),
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        rand_perm.to(device=orig_device),
    )


def gptq_quantize_weights(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
):
    size_k, _ = w.shape

    assert w.is_floating_point(), "w must be float"
    assert quant_type in SUPPORTED_GPTQ_QUANT_TYPES, (
        f"Unsupported gptq type = {quant_type}"
    )
    assert group_size in SUPPORTED_GROUP_SIZES + [size_k], (
        f"Unsupported groupsize = {group_size}"
    )

    w_ref, w_q, w_s, _ = quantize_weights(w, quant_type, group_size)

    # Apply act_order
    g_idx = torch.empty(0, dtype=torch.int, device=w.device)
    rand_perm = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        assert group_size < size_k, (
            "For act_order, groupsize = {} must be less than size_k = {}".format(
                group_size, size_k
            )
        )

        w_ref, w_q, g_idx, rand_perm = permute_rows(w_q, w_ref, group_size, test_perm)

    return w_ref, w_q, w_s, g_idx, rand_perm
def sort_weights(q_w: torch.Tensor, g_idx: torch.Tensor):
    orig_device = q_w.device

    sort_indices = torch.argsort(g_idx).to(dtype=torch.int32)  # Sort based on g_idx

    g_idx = g_idx[sort_indices].contiguous()
    q_w = q_w[sort_indices, :].contiguous()

    return (
        q_w.to(device=orig_device),
        g_idx.to(device=orig_device),
        sort_indices.to(device=orig_device),
    )

def get_weight_perm(num_bits: int, is_a_8bit: bool = False):
    perm_list: list[int] = []
    if is_a_8bit:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    4 * (i % 4),
                    4 * (i % 4) + 1,
                    4 * (i % 4) + 2,
                    4 * (i % 4) + 3,
                    4 * (i % 4 + 4),
                    4 * (i % 4 + 4) + 1,
                    4 * (i % 4 + 4) + 2,
                    4 * (i % 4 + 4) + 3,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(2):
                perm_list.extend([p + 512 * j for p in perm1])
    else:
        for i in range(32):
            perm1 = []
            col = i // 4
            for block in [0, 1]:
                for row in [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ]:
                    perm1.append(16 * row + col + 8 * block)
            for j in range(4):
                perm_list.extend([p + 256 * j for p in perm1])

    perm = np.array(perm_list)

    if num_bits == 4:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 4, 1, 5, 2, 6, 3, 7])
        else:
            interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    elif num_bits == 8:
        if is_a_8bit:  # noqa: SIM108
            interleave = np.array([0, 1, 2, 3])
        else:
            interleave = np.array([0, 2, 1, 3])
    else:
        raise Exception("num_bits must be 4 or 8, got {}".format(num_bits))

    perm = perm.reshape((-1, len(interleave)))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    return perm

GPTQ_MARLIN_TILE = 16
GPTQ_MARLIN_MIN_THREAD_N = 64
GPTQ_MARLIN_MIN_THREAD_K = 128
GPTQ_MARLIN_MAX_PARALLEL = 16

def marlin_permute_weights(
    q_w, size_k, size_n, perm, tile=GPTQ_MARLIN_TILE, is_a_8bit=False
):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    if is_a_8bit:
        # Permute weights to 32x32 marlin tiles
        q_w = q_w.reshape((size_k // (tile * 2), tile * 2, size_n // tile, tile))
    else:
        # Permute weights to 16x64 marlin tiles
        q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w
def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits


def marlin_weights(q_w, size_k, size_n, num_bits, perm, is_a_8bit=False):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm, is_a_8bit=is_a_8bit)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed

def marlin_permute_scales(
    s: torch.Tensor, size_k: int, size_n: int, group_size: int, is_a_8bit: bool = False
) -> torch.Tensor:
    scale_perm, scale_perm_single = get_scale_perms()
    if group_size < size_k and group_size != -1 and not is_a_8bit:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()

    return s
    
def marlin_quantize(
    w: torch.Tensor,
    quant_type: ScalarType,
    group_size: int,
    act_order: bool,
    test_perm: torch.Tensor | None = None,
    input_dtype: torch.dtype | None = None,
):
    is_a_8bit = input_dtype is not None and input_dtype.itemsize == 1

    size_k, size_n = w.shape
    num_bits = quant_type.size_bits

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Quantize (and apply act_order if provided)
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        w, quant_type, group_size, act_order, test_perm
    )

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=w.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Reformat to marlin
    weight_perm = get_weight_perm(num_bits, is_a_8bit)
    marlin_q_w = marlin_weights(
        q_w, size_k, size_n, num_bits, weight_perm, is_a_8bit=is_a_8bit
    )
    marlin_s = marlin_permute_scales(s, size_k, size_n, group_size, is_a_8bit=is_a_8bit)

    if input_dtype == torch.float8_e4m3fn and quant_type == scalar_types.uint4b8:
        print("not support dtype")
        return

    # Create result
    res_list = [w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, rand_perm]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list

def marlin_make_empty_g_idx(device: torch.device) -> torch.Tensor:
    return torch.nn.Parameter(
        torch.empty(0, dtype=torch.int, device=device), requires_grad=False
    )
    
def compute_max_diff(output, output_ref):
    return torch.mean(torch.abs(output - output_ref)) / torch.mean(
        torch.abs(output_ref)
    )
    
def awq_marlin_gemm_torch(a_input, w_ref, b_bias):
    if b_bias == None:
        return torch.matmul(a_input, w_ref)
    else:
        return torch.matmul(a_input, w_ref) + b_bias.view(1, -1)

    
def test_marlin_gemm_subset_input(handle,
    device,
    size_m,
    size_k,
    size_n,
    group_size,
    quant_type,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing awq_marlin_gemm_subset_input on {device} with M-K-N:({size_m, size_k, size_n}), group_size:{group_size}, dtype:{InfiniDtypeNames[dtype]}"
    )
    big_m = size_m * 2
    big_k = size_k * 2
    test_dtype = to_torch_dtype(dtype)
    
    a_input = torch.randn((big_m, big_k), dtype=test_dtype)[8 : size_m + 8, 8 : size_k + 8]
    A = TestTensor(
        a_input.shape,
        a_input.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=a_input,
    )
    b_weight = TestTensor((size_k, size_n), None, dtype, device)

    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight.torch_tensor(), quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    
    ans = awq_marlin_gemm_torch(A.torch_tensor(), w_ref, None)
    # print("w", w_ref.shape, w_ref.dtype, w_ref.stride())
    # print("b", marlin_q_w.shape, marlin_q_w.dtype, marlin_q_w.stride())
    # print("b_scales", marlin_s.shape, marlin_s.dtype, marlin_s.stride())
    # print("g_idx", g_idx.shape, g_idx.dtype, g_idx.stride())
    # print("perm", sort_indices.shape, sort_indices.dtype, sort_indices.stride())
    # print("g_idx", marlin_zp)
    output = TestTensor(ans.shape, None, dtype, device, mode="zeros")
    
    B = TestTensor(
        marlin_q_w.shape,
        marlin_q_w.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=marlin_q_w,
    )
    b_bias = None
    b_scales = TestTensor(
        marlin_s.shape,
        marlin_s.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=marlin_s,
    )
    a_scales = None
    global_scales = None
    if marlin_zp is not None:
        b_zeros = TestTensor(
            marlin_zp.shape,
            marlin_zp.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=marlin_zp,
        )
    else:
        b_zeros = None
    if g_idx is not None:
        b_g_idx = TestTensor(
            g_idx.shape,
            g_idx.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=g_idx,
        )
    else:
        b_g_idx = None
    if sort_indices is not None:
        perm = TestTensor(
            sort_indices.shape,
            sort_indices.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=sort_indices,
        )
    else:
        perm = None
    is_k_full=True
    use_atomic_add=False
    use_fp32_reduce=True
    is_zp_float=False
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAwqMarlinGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            A.descriptor,
            B.descriptor,
            b_bias.descriptor if b_bias is not None else None,
            b_scales.descriptor,
            a_scales.descriptor if a_scales is not None else None,
            global_scales.descriptor if global_scales is not None else None,
            b_zeros.descriptor if b_zeros is not None else None,
            b_g_idx.descriptor if b_g_idx is not None else None,
            perm.descriptor if perm is not None else None,
        )
    )

    # Invalidate descriptors (same pattern as other tests)
    for tensor in [output, A, B, b_bias, b_scales, a_scales, global_scales, b_zeros, b_g_idx, perm]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAwqMarlinGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_awq_marlin_gemm():
        check_error(
            LIBINFINIOP.infiniopAwqMarlinGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                A.data(),
                B.data(),
                b_bias.data() if b_bias is not None else None,
                b_scales.data(),
                a_scales.data() if a_scales is not None else None,
                global_scales.data() if global_scales is not None else None,
                b_zeros.data() if b_zeros is not None else None,
                b_g_idx.data() if b_g_idx is not None else None,
                perm.data() if perm is not None else None,
                quant_type.id,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
                None,
            )
        )

    lib_awq_marlin_gemm()
    
    max_diff = compute_max_diff(output.actual_tensor(), ans)
    assert max_diff < 0.04
    
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: awq_marlin_gemm_torch(A.torch_tensor(), w_ref, None),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_awq_marlin_gemm(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(LIBINFINIOP.infiniopDestroyAwqMarlinGemmDescriptor(descriptor))
    
    
def test_marlin_gemm_with_bias(handle,
    device,
    size_m,
    size_k,
    size_n,
    group_size,
    quant_type,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing awq_marlin_gemm_with_bias on {device} with M-K-N:({size_m, size_k, size_n}), group_size:{group_size}, dtype:{InfiniDtypeNames[dtype]}"
    )

    test_dtype = to_torch_dtype(dtype)
    
    A = TestTensor((size_m, size_k), None, dtype, device)
    b_weight = TestTensor((size_k, size_n), None, dtype, device)
    b_bias = TestTensor((size_n, ), None, dtype, device)

    marlin_bias = marlin_permute_bias(b_bias.torch_tensor())
    w_ref, marlin_q_w, marlin_s, g_idx, sort_indices, _ = marlin_quantize(
        b_weight.torch_tensor(), quant_type, group_size, False
    )

    marlin_zp = marlin_make_empty_g_idx(marlin_s.device)
    
    ans = awq_marlin_gemm_torch(A.torch_tensor(), w_ref, b_bias.torch_tensor())
    # print("w", w_ref.shape, w_ref.dtype, w_ref.stride())
    # print("b", marlin_q_w.shape, marlin_q_w.dtype, marlin_q_w.stride())
    # print("b_scales", marlin_s.shape, marlin_s.dtype, marlin_s.stride())
    # print("g_idx", g_idx.shape, g_idx.dtype, g_idx.stride())
    # print("perm", sort_indices.shape, sort_indices.dtype, sort_indices.stride())
    # print("g_idx", marlin_zp)
    output = TestTensor(ans.shape, None, dtype, device, mode="zeros")
    
    B = TestTensor(
        marlin_q_w.shape,
        marlin_q_w.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=marlin_q_w,
    )
    
    b_scales = TestTensor(
        marlin_s.shape,
        marlin_s.stride(),
        dtype,
        device,
        mode="manual",
        set_tensor=marlin_s,
    )
    a_scales = None
    global_scales = None
    if marlin_zp is not None:
        b_zeros = TestTensor(
            marlin_zp.shape,
            marlin_zp.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=marlin_zp,
        )
    else:
        b_zeros = None
    if g_idx is not None:
        b_g_idx = TestTensor(
            g_idx.shape,
            g_idx.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=g_idx,
        )
    else:
        b_g_idx = None
    if sort_indices is not None:
        perm = TestTensor(
            sort_indices.shape,
            sort_indices.stride(),
            InfiniDtype.I32,
            device,
            mode="manual",
            set_tensor=sort_indices,
        )
    else:
        perm = None
    is_k_full=True
    use_atomic_add=False
    use_fp32_reduce=True
    is_zp_float=False
    
    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateAwqMarlinGemmDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            A.descriptor,
            B.descriptor,
            b_bias.descriptor if b_bias is not None else None,
            b_scales.descriptor,
            a_scales.descriptor if a_scales is not None else None,
            global_scales.descriptor if global_scales is not None else None,
            b_zeros.descriptor if b_zeros is not None else None,
            b_g_idx.descriptor if b_g_idx is not None else None,
            perm.descriptor if perm is not None else None,
        )
    )

    # Invalidate descriptors (same pattern as other tests)
    for tensor in [output, A, B, b_bias, b_scales, a_scales, global_scales, b_zeros, b_g_idx, perm]:
        if tensor is not None:
            tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetAwqMarlinGemmWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    def lib_awq_marlin_gemm():
        check_error(
            LIBINFINIOP.infiniopAwqMarlinGemm(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                A.data(),
                B.data(),
                b_bias.data() if b_bias is not None else None,
                b_scales.data(),
                a_scales.data() if a_scales is not None else None,
                global_scales.data() if global_scales is not None else None,
                b_zeros.data() if b_zeros is not None else None,
                b_g_idx.data() if b_g_idx is not None else None,
                perm.data() if perm is not None else None,
                quant_type.id,
                is_k_full,
                use_atomic_add,
                use_fp32_reduce,
                is_zp_float,
                None,
            )
        )

    lib_awq_marlin_gemm()
    
    max_diff = compute_max_diff(output.actual_tensor(), ans)
    assert max_diff < 0.04
    
    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: awq_marlin_gemm_torch(A.torch_tensor(), w_ref, b_bias.torch_tensor()),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_awq_marlin_gemm(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(LIBINFINIOP.infiniopDestroyAwqMarlinGemmDescriptor(descriptor))  
    
    


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test_marlin_gemm_subset_input, _TEST_CASES_SUBSET_INPUT, _TENSOR_DTYPES)
        test_operator(device, test_marlin_gemm_with_bias, _TEST_CASES_WITH_BIAS, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
