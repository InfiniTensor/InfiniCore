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
    torch_device_map,
)
import itertools
import numpy
from libinfiniop.scalar_type import scalar_types, ScalarType
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union
import numpy as np


ACT_ORDER_OPTS = [False, True]
MARLIN_K_CHUNKS = [128]
MARLIN_N_CHUNKS = [64, 256]
ACT_ORDER_OPTS = [False, True]
MARLIN_REPACK_NK_FACTORS = [
    (4, 8),
    (7, 5),
    (13, 11),
]

def to_iter(x):
    return x if isinstance(x, (list, tuple)) else (x,)


_TEST_CASES = list(
    itertools.product(
        to_iter(MARLIN_K_CHUNKS),
        to_iter(MARLIN_N_CHUNKS),
        to_iter([scalar_types.uint4b8, scalar_types.uint8b128]),
        to_iter(ACT_ORDER_OPTS),
        to_iter([True, False]),
        to_iter(MARLIN_REPACK_NK_FACTORS),
        to_iter([128]),
    )
)

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 3e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000

SUPPORTED_GPTQ_QUANT_TYPES = [scalar_types.uint4b8, scalar_types.uint8b128]
SUPPORTED_GROUP_SIZES = [-1, 32, 64, 128]
GPTQ_MARLIN_TILE = 16

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

def get_pack_factor(num_bits):
    assert 32 % num_bits == 0, f"Unsupported num_bits = {num_bits}"
    return 32 // num_bits

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


def gptq_pack(
    q_w: torch.Tensor,
    num_bits: int,
    size_k: int,
    size_n: int,
):
    return pack_rows(q_w, num_bits, size_k, size_n)

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

def gptq_marlin_repack_torch(b_weight, size_k, size_n, group_size, quant_type, act_order, is_a_8bit):
    w_ref, q_w, s, g_idx, rand_perm = gptq_quantize_weights(
        b_weight, quant_type, group_size, act_order
    )

    # Pack to GPTQ format
    q_w_gptq = gptq_pack(q_w, quant_type.size_bits, size_k, size_n)

    # For act_order, sort the "weights" and "g_idx" so that group ids are
    # increasing
    sort_indices = torch.empty(0, dtype=torch.int, device=b_weight.device)
    if act_order:
        q_w, g_idx, sort_indices = sort_weights(q_w, g_idx)

    # Pack to Marlin format
    weight_perm = get_weight_perm(quant_type.size_bits, is_a_8bit)
    marlin_q_w_1 = marlin_weights(
        q_w, size_k, size_n, quant_type.size_bits, weight_perm, is_a_8bit
    )
    return q_w_gptq, sort_indices, marlin_q_w_1
    
def test(
    handle,
    device,
    k_chunk, 
    n_chunk, 
    quant_type, 
    act_order,
    is_a_8bit, 
    nk_factors,
    group_size=128,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing gptq_marlin_repack on {device} with k_chunk:{k_chunk}, n_chunk:{n_chunk}, act_order:{act_order}, is_a_8bit:{is_a_8bit}, nk_factors:{nk_factors}, group_size:{group_size}, dtype:{InfiniDtypeNames[dtype]}"
    )
    n_factor, k_factor = nk_factors

    size_k = k_chunk * k_factor
    size_n = n_chunk * n_factor

    if act_order:
        if group_size == -1:
            return
        if group_size == size_k:
            return
        if is_a_8bit:
            return

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k
    
    test_dtype = to_torch_dtype(dtype)
    device_str = torch_device_map[device]
    b_weight = torch.randn((size_k, size_n), dtype=test_dtype, device=device_str) #must be randn
    
    q_w_gptq, sort_indices, ans = gptq_marlin_repack_torch(b_weight, size_k, size_n, group_size, quant_type, act_order, is_a_8bit)

    input = TestTensor(
        q_w_gptq.shape,
        q_w_gptq.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=q_w_gptq,
    )
    perm = TestTensor(
        sort_indices.shape,
        sort_indices.stride(),
        InfiniDtype.I32,
        device,
        mode="manual",
        set_tensor=sort_indices,
    )
    
    output = TestTensor(ans.shape, None, InfiniDtype.I32, device, mode="zeros")

    if sync is not None:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateGptqMarlinRepackDescriptor(
            handle,
            ctypes.byref(descriptor),
            output.descriptor,
            input.descriptor,
            perm.descriptor,
            quant_type.size_bits,
            is_a_8bit,
        )
    )

    for tensor in [
        output,
        input,
        perm,
    ]:
        tensor.destroy_desc()

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetGptqMarlinRepackWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)
    
    def lib_gptq_marlin_repack():
        check_error(
            LIBINFINIOP.infiniopGptqMarlinRepack(
                descriptor,
                workspace.data(),
                workspace_size.value,
                output.data(),
                input.data(),
                perm.data(),
                None,
            )
        )

    lib_gptq_marlin_repack()
    
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(output.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(output.actual_tensor().to(ans.dtype), ans, atol=atol, rtol=rtol)

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: gptq_marlin_repack_torch(b_weight.torch_tensor(), size_k, size_n, group_size, quant_type, act_order, is_a_8bit),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lambda: lib_gptq_marlin_repack(),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(LIBINFINIOP.infiniopDestroyGptqMarlinRepackDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
