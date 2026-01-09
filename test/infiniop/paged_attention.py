import torch
import ctypes
from ctypes import c_uint64
import math
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
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    TestWorkspace,
)


# ==============================================================================
#  Reference Implementation
# ==============================================================================
def get_alibi_slopes(n):
    # 简化版的ALiBi斜率计算方法
    # 参考: https://github.com/ofirpress/attention_with_linear_biases/blob/master/fairseq/models/transformer.py#L742
    closest_power_of_2 = 2 ** math.floor(math.log2(n))
    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = [base**i for i in range(1, closest_power_of_2 + 1)]
    if n > closest_power_of_2:
        extra = [base ** (i * 2) for i in range(1, 2 * (n - closest_power_of_2) + 1, 2)]
        powers += extra
    return powers[:n]


def ref_masked_attention(query, key, value, scale, attn_mask=None):
    # Reference implementation for a single masked attention head.
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def ref_single_query_cached_kv_attention(
    query, key_cache, value_cache, block_tables, seq_lens, scale, alibi_slopes
):
    # Reference implementation for paged attention, iterating through each sequence.
    output = torch.empty_like(query)
    num_query_heads, num_kv_heads = query.shape[1], value_cache.shape[1]
    num_queries_per_kv = num_query_heads // num_kv_heads
    head_size, block_size = value_cache.shape[3], value_cache.shape[2]
    num_seqs = query.shape[0]

    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        seq_len = seq_lens[i].item()
        block_table = block_tables[i]

        keys_lst, values_lst = [], []
        for j in range(seq_len):
            block_num = block_table[j // block_size].item()
            block_off = j % block_size
            k = key_cache[block_num, :, block_off, :]
            v = value_cache[block_num, :, block_off, :]
            keys_lst.append(k)
            values_lst.append(v)

        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)
        alibi_bias = None
        if alibi_slopes is not None:
            pos = torch.arange(seq_len, device=query.device).int()
            alibi_bias = (pos - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(q, keys, values, scale, alibi_bias)
        output[i] = out.view(num_query_heads, head_size)
    return output


# ==============================================================================
#  Test Configuration (Internal Use Only)
# ==============================================================================
# These are not meant to be imported from other modules
_TEST_CASES_ = [
    # (num_seqs, num_heads, num_kv_heads, head_size, block_size, max_seq_len, use_alibi)
    (1, 1, 1, 128, 16, 1024, False),
    (4, 40, 40, 128, 16, 1024, False),
    (6, 40, 40, 128, 16, 1024, False),
    (3, 8, 8, 128, 16, 1024, False),
    (3, 8, 8, 64, 16, 1024, False),
    (8, 64, 8, 128, 16, 2048, False),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.BF16, InfiniDtype.F16, InfiniDtype.F32]

# Tolerance map for different data types
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

# Global flags for controlling test behavior
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 1000


def test(
    handle,
    device,
    num_seqs,
    num_heads,
    num_kv_heads,
    head_size,
    block_size,
    max_seq_len,
    use_alibi,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing PagedAttention on {InfiniDeviceNames[device]} with "
        f"num_seqs={num_seqs}, num_heads={num_heads}, head_size={head_size}, "
        f"block_size={block_size}, dtype={InfiniDtypeNames[dtype]}, use_alibi={use_alibi}"
    )

    scale = 1.0 / (head_size**0.5)
    max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = num_seqs * max_blocks_per_seq  # A reasonable number for testing

    # Create input tensors
    q = TestTensor((num_seqs, num_heads, head_size), None, dtype, device)
    out = TestTensor((num_seqs, num_heads, head_size), None, dtype, device)
    k_cache = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )
    v_cache = TestTensor(
        (num_blocks, num_kv_heads, block_size, head_size), None, dtype, device
    )

    seq_lens_torch = torch.randint(1, max_seq_len, (num_seqs,), dtype=torch.int64)

    seq_lens = TestTensor.from_torch(seq_lens_torch, InfiniDtype.I64, device)

    block_tables_py = torch.arange(
        0, num_seqs * max_blocks_per_seq, dtype=torch.int64
    ).view(num_seqs, max_blocks_per_seq)
    block_tables = TestTensor.from_torch(block_tables_py, InfiniDtype.I64, device)

    alibi_slopes_desc = ctypes.c_void_p(0)
    alibi_slopes_data = ctypes.c_void_p(0)
    alibi_slopes_torch = None
    if use_alibi:
        alibi_slopes = TestTensor((num_heads,), None, InfiniDtype.F32, device)
        alibi_slopes_desc = alibi_slopes.descriptor
        alibi_slopes_data = alibi_slopes.data()
        alibi_slopes_torch = alibi_slopes.torch_tensor()

    # Run reference implementation
    ans = ref_single_query_cached_kv_attention(
        q.torch_tensor(),
        k_cache.torch_tensor(),
        v_cache.torch_tensor(),
        block_tables.torch_tensor(),
        seq_lens.torch_tensor(),
        scale,
        alibi_slopes_torch,
    )

    if sync:
        sync()

    scale = 1.0 / (head_size**0.5)
    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreatePagedAttentionDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            q.descriptor,
            k_cache.descriptor,
            v_cache.descriptor,
            block_tables.descriptor,
            seq_lens.descriptor,
            alibi_slopes_desc,
            scale,
        )
    )

    # Get workspace size and allocate memory
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetPagedAttentionWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    # Invalidate descriptors to ensure kernel does not rely on them
    q.destroy_desc()
    out.destroy_desc()
    k_cache.destroy_desc()
    v_cache.destroy_desc()
    block_tables.destroy_desc()
    seq_lens.destroy_desc()
    if use_alibi:
        alibi_slopes.destroy_desc()

    # Define the library call as a lambda for profiling
    def lib_paged_attention():
        check_error(
            LIBINFINIOP.infiniopPagedAttention(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                q.data(),
                k_cache.data(),
                v_cache.data(),
                block_tables.data(),
                seq_lens.data(),
                alibi_slopes_data,
                None,
            )
        )

    # Execute the custom operator
    lib_paged_attention()

    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans, atol=atol, rtol=rtol)

    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: ref_single_query_cached_kv_attention(
            q.torch_tensor(), k_cache.torch_tensor(), v_cache.torch_tensor(),
            block_tables.torch_tensor(), seq_lens.torch_tensor(),
            scale, alibi_slopes_torch), 
            device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_paged_attention, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on

    # Clean up resources
    check_error(LIBINFINIOP.infiniopDestroyPagedAttentionDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()

    # Configure testing options from command line arguments
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
