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
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceEnum,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    TestWorkspace,
)


# ==============================================================================
#  Reference Implementation
# ==============================================================================
def ref_paged_caching(key_cache_pool, value_cache_pool, key, value, slot_mapping):
    """
    Reference implementation for paged_caching operator.

    Args:
        key_cache_pool (torch.Tensor): K cache pool, shape [num_blocks, nkvh, block_size, dh]
        value_cache_pool (torch.Tensor): V cache pool, shape [num_blocks, nkvh, block_size, dv]
        key (torch.Tensor): Keys, shape [ntok, nkvh, dh]
        value (torch.Tensor): Values, shape [ntok, nkvh, dv]
        slot_mapping (torch.Tensor): Slot mapping, shape [ntok]
    """
    ntok = key.shape[0]
    block_size = key_cache_pool.shape[2]

    # This reference implementation operates on a cloned cache to avoid modifying the original input tensor,
    # mimicking the behavior where the custom operator writes to its output tensor.
    k_cache_ref = key_cache_pool.clone()
    v_cache_ref = value_cache_pool.clone()

    for i in range(ntok):
        slot = slot_mapping[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size

        key_token = key[i]
        value_token = value[i]

        k_cache_ref[block_idx, :, block_offset, :] = key_token
        v_cache_ref[block_idx, :, block_offset, :] = value_token

    return k_cache_ref, v_cache_ref


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
_TEST_CASES_ = [
    # (num_seqs, max_seq_len, num_kv_heads, head_size, block_size[, value_size])
    (1, 128, 8, 128, 16),
    (5, 512, 40, 128, 16),
    (16, 1024, 8, 64, 32),
    (10, 1024, 40, 64, 32),
    # New DeepSeek MLA case: verifies paged_caching writes K/V caches when
    # the key head size and value head size differ.
    (2, 128, 1, 576, 16, 512),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 0, "rtol": 1e-5},
    InfiniDtype.BF16: {"atol": 0, "rtol": 1e-5},
    InfiniDtype.F32: {"atol": 0, "rtol": 1e-5},
}

# Global flags for controlling test behavior
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def test(
    handle,
    device,
    num_seqs,  # nreq
    max_seq_len,
    num_kv_heads,  # nkvh
    head_size,  # dh
    block_size,
    *tail,
):
    if len(tail) == 2:
        dtype, sync = tail
        value_size = head_size
    elif len(tail) == 3:
        value_size, dtype, sync = tail
    else:
        raise ValueError(f"Unexpected paged_caching test arguments: {tail}")
    print(
        f"Testing PagedCaching on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, max_seq_len={max_seq_len}, num_kv_heads={num_kv_heads}, "
        f"head_size={head_size}, value_size={value_size}, block_size={block_size}, "
        f"dtype={InfiniDtypeNames[dtype]}"
    )

    num_blocks = 4096  # A reasonably large cache pool for testing

    # Create metadata: variable context lengths for each sequence in the batch
    context_lens_torch = torch.randint(
        1, max_seq_len + 1, (num_seqs,), dtype=torch.int64
    )
    ntok = torch.sum(context_lens_torch).item()

    # If ntok is 0 (all sequences have length 0), skip the test
    if ntok == 0:
        print("Skipping test case with ntok=0")
        return

    # Simulate the scheduler's behavior to create the slot_mapping
    slot_mapping_list = []
    current_slot = 0
    for length in context_lens_torch:
        # Find a contiguous chunk of 'length' slots
        start_slot = current_slot
        slot_mapping_list.extend(range(start_slot, start_slot + length.item()))
        current_slot += length.item()

    # Ensure we don't exceed the total number of slots in the cache
    assert (
        current_slot <= num_blocks * block_size
    ), "Not enough blocks in the cache pool for this test case"

    slot_mapping_torch = torch.tensor(slot_mapping_list, dtype=torch.int64)

    # Create input tensors based on the calculated total tokens (ntok)
    k = TestTensor((ntok, num_kv_heads, head_size), None, dtype, device)
    v = TestTensor((ntok, num_kv_heads, value_size), None, dtype, device)
    slot_mapping = TestTensor.from_torch(slot_mapping_torch, InfiniDtype.I64, device)

    # The cache pools are the "output" tensors for this operator
    k_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )
    v_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, value_size), None, dtype, device
    )

    # Run reference implementation
    k_cache_ref, v_cache_ref = ref_paged_caching(
        k_cache_pool.torch_tensor(),
        v_cache_pool.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        slot_mapping.torch_tensor(),
    )

    if sync:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePagedCachingDescriptor(
            handle,
            ctypes.byref(descriptor),
            k_cache_pool.descriptor,
            v_cache_pool.descriptor,
            k.descriptor,
            v.descriptor,
            slot_mapping.descriptor,
            None,
            None,
        )
    )

    # Get workspace size (likely 0 for this operator, but good practice to include)
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPagedCachingWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    # Invalidate descriptors to ensure kernel does not rely on them
    k.destroy_desc()
    v.destroy_desc()
    k_cache_pool.destroy_desc()
    v_cache_pool.destroy_desc()
    slot_mapping.destroy_desc()

    # Define the library call as a lambda for profiling
    def lib_paged_caching():
        check_error(
            LIBINFINIOP.infiniopPagedCaching(
                descriptor,
                workspace.data(),
                workspace_size.value,
                k_cache_pool.data(),
                v_cache_pool.data(),
                k.data(),
                v.data(),
                slot_mapping.data(),
                None,
                None,
                None,
            )
        )

    # Execute the custom operator
    lib_paged_caching()

    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        print("Verifying K cache...")
        debug(k_cache_pool.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol)
        print("Verifying V cache...")
        debug(v_cache_pool.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol)

    assert torch.allclose(
        k_cache_pool.actual_tensor(), k_cache_ref, atol=atol, rtol=rtol
    )
    assert torch.allclose(
        v_cache_pool.actual_tensor(), v_cache_ref, atol=atol, rtol=rtol
    )

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: ref_paged_caching(
            k.torch_tensor(), v.torch_tensor(), 
            k_cache_pool.torch_tensor(), v_cache_pool.torch_tensor(), 
            slot_mapping.torch_tensor()), 
            device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_paged_caching, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # Clean up resources
    check_error(LIBINFINIOP.infiniopDestroyPagedCachingDescriptor(descriptor))


# ==============================================================================
#  FP8 (E4M3) Cache Test
# ==============================================================================
# FP8 cases reuse the shape tuples of _TEST_CASES_; the cache pools are F8 while
# the source K/V stay in _TENSOR_DTYPES_FP8_ and per-token scales are F32.
_TENSOR_DTYPES_FP8_ = [InfiniDtype.F16, InfiniDtype.BF16]


def quantize_per_token_ref(x):
    """
    Reference dynamic per-token-per-head FP8(E4M3) quantization, replicating the
    kernel's exact float32 op order:
        amax = max(|x|) over the head dim
        scale = amax / 448   (scale = 1 when amax == 0)
        q = e4m3_encode(x * (1 / scale))

    NOTE: computed on CPU deliberately. On CUDA, torch lowers tensor/scalar
    division to multiply-by-reciprocal, which differs from the kernel's IEEE
    division by 1 ulp; at exact E4M3 rounding midpoints that flips the chosen
    code and breaks bitwise comparison. CPU torch uses true division and
    matches the kernel bit-for-bit.
    """
    x = x.cpu()
    x_float = x.float()
    amax = x_float.abs().amax(dim=-1)
    scale = torch.where(amax > 0, amax / 448.0, torch.ones_like(amax))
    inv_scale = 1.0 / scale
    q = (x_float * inv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return q, scale


def test_fp8(
    handle,
    device,
    num_seqs,
    max_seq_len,
    num_kv_heads,
    head_size,
    block_size,
    *tail,
):
    if len(tail) == 2:
        dtype, sync = tail
        value_size = head_size
    elif len(tail) == 3:
        value_size, dtype, sync = tail
    else:
        raise ValueError(f"Unexpected paged_caching FP8 test arguments: {tail}")
    print(
        f"Testing PagedCaching FP8 on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, max_seq_len={max_seq_len}, num_kv_heads={num_kv_heads}, "
        f"head_size={head_size}, value_size={value_size}, block_size={block_size}, "
        f"kv dtype={InfiniDtypeNames[dtype]}, cache dtype=F8"
    )

    num_blocks = 4096

    context_lens_torch = torch.randint(
        1, max_seq_len + 1, (num_seqs,), dtype=torch.int64
    )
    ntok = torch.sum(context_lens_torch).item()
    if ntok == 0:
        print("Skipping test case with ntok=0")
        return

    slot_mapping_list = []
    current_slot = 0
    for length in context_lens_torch:
        start_slot = current_slot
        slot_mapping_list.extend(range(start_slot, start_slot + length.item()))
        current_slot += length.item()
    assert current_slot <= num_blocks * block_size
    slot_mapping_torch = torch.tensor(slot_mapping_list, dtype=torch.int64)

    # Source K/V stay in half precision; caches hold raw F8 bytes.
    k = TestTensor(
        (ntok, num_kv_heads, head_size), None, dtype, device, scale=4.0, bias=-2.0
    )
    v = TestTensor(
        (ntok, num_kv_heads, value_size), None, dtype, device, scale=4.0, bias=-2.0
    )
    slot_mapping = TestTensor.from_torch(slot_mapping_torch, InfiniDtype.I64, device)

    k_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size),
        None,
        InfiniDtype.F8,
        device,
        mode="float8_e4m3fn",
    )
    v_cache_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size, value_size),
        None,
        InfiniDtype.F8,
        device,
        mode="float8_e4m3fn",
    )
    k_scale_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size), None, InfiniDtype.F32, device, mode="zeros"
    )
    v_scale_pool = TestTensor(
        (num_blocks, num_kv_heads, block_size), None, InfiniDtype.F32, device, mode="zeros"
    )

    # Reference: quantize every token, then scatter codes and scales into the pools.
    qk, sk = quantize_per_token_ref(k.torch_tensor())
    qv, sv = quantize_per_token_ref(v.torch_tensor())
    k_cache_ref = k_cache_pool.torch_tensor().view(torch.uint8).clone()
    v_cache_ref = v_cache_pool.torch_tensor().view(torch.uint8).clone()
    k_scale_ref = k_scale_pool.torch_tensor().clone()
    v_scale_ref = v_scale_pool.torch_tensor().clone()
    for i in range(ntok):
        slot = slot_mapping_torch[i].item()
        block_idx = slot // block_size
        block_offset = slot % block_size
        k_cache_ref[block_idx, :, block_offset, :] = qk[i].view(torch.uint8)
        v_cache_ref[block_idx, :, block_offset, :] = qv[i].view(torch.uint8)
        k_scale_ref[block_idx, :, block_offset] = sk[i]
        v_scale_ref[block_idx, :, block_offset] = sv[i]

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePagedCachingDescriptor(
            handle,
            ctypes.byref(descriptor),
            k_cache_pool.descriptor,
            v_cache_pool.descriptor,
            k.descriptor,
            v.descriptor,
            slot_mapping.descriptor,
            k_scale_pool.descriptor,
            v_scale_pool.descriptor,
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPagedCachingWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, device)

    k.destroy_desc()
    v.destroy_desc()
    k_cache_pool.destroy_desc()
    v_cache_pool.destroy_desc()
    slot_mapping.destroy_desc()
    k_scale_pool.destroy_desc()
    v_scale_pool.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopPagedCaching(
            descriptor,
            workspace.data(),
            workspace_size.value,
            k_cache_pool.data(),
            v_cache_pool.data(),
            k.data(),
            v.data(),
            slot_mapping.data(),
            k_scale_pool.data(),
            v_scale_pool.data(),
            None,
        )
    )

    if sync:
        sync()

    if DEBUG:
        debug(
            k_cache_pool.actual_tensor().view(torch.uint8),
            k_cache_ref,
            atol=0,
            rtol=0,
        )
        debug(k_scale_pool.actual_tensor(), k_scale_ref, atol=0, rtol=1e-5)

    kc = k_cache_pool.actual_tensor().view(torch.uint8)
    vc = v_cache_pool.actual_tensor().view(torch.uint8)
    for name, actual, ref, src in (
        ("K", kc, k_cache_ref, k.torch_tensor()),
        ("V", vc, v_cache_ref, v.torch_tensor()),
    ):
        m = actual != ref
        if m.any():
            print(f"[DIAG] {name} cache mismatch {m.sum().item()}/{actual.numel()}")
            inv = {s: t for t, s in enumerate(slot_mapping_torch.tolist())}
            for idx in m.nonzero()[:6]:
                b, h, off, d = [int(x) for x in idx.tolist()]
                tok = inv.get(b * block_size + off, -1)
                x = src[tok, h, d].float().item() if tok >= 0 else float("nan")
                amax = src[tok, h].float().abs().max().item() if tok >= 0 else float("nan")
                sc = amax / 448.0 if amax > 0 else 1.0
                y = x * (1.0 / sc)
                print(
                    f"[DIAG] {name} blk{b} h{h} off{off} d{d} tok{tok} "
                    f"ref=0x{ref[b, h, off, d].item():02x} act=0x{actual[b, h, off, d].item():02x} "
                    f"x={x!r} amax={amax!r} x*inv_scale={y!r}"
                )
    print(
        f"[DIAG] scale maxdiff "
        f"k={(k_scale_pool.actual_tensor() - k_scale_ref).abs().max().item()} "
        f"v={(v_scale_pool.actual_tensor() - v_scale_ref).abs().max().item()}"
    )
    assert torch.equal(k_cache_pool.actual_tensor().view(torch.uint8), k_cache_ref)
    assert torch.equal(v_cache_pool.actual_tensor().view(torch.uint8), v_cache_ref)
    assert torch.allclose(
        k_scale_pool.actual_tensor(), k_scale_ref, atol=0, rtol=1e-5
    )
    assert torch.allclose(
        v_scale_pool.actual_tensor(), v_scale_ref, atol=0, rtol=1e-5
    )

    check_error(LIBINFINIOP.infiniopDestroyPagedCachingDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options from command line arguments
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
        if device == InfiniDeviceEnum.NVIDIA:
            test_operator(device, test_fp8, _TEST_CASES_, _TENSOR_DTYPES_FP8_)

    print("\033[92mTest passed!\033[0m")
