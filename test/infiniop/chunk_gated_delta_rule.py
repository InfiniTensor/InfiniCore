# test_chunk_gated_delta_rule.py

import torch
import torch.nn.functional as F
import ctypes
from ctypes import c_uint32, c_float, c_uint64, c_size_t, POINTER, addressof

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
# From modeling_qwen3_next.py, the production PyTorch fallback implementation
def ref_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    # ==================== DEBUG PRINT 2b =====================
    print("--- Python ref key (BEFORE l2, b=0, h=0, chunk=0) ---")
    print(key[0, 0, :, :chunk_size])
    # =========================================================
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    
    
    # The production implementation expects (B, T, H, D) and transposes internally
    # query, key, value, beta, g = [
    #     x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    # ]

    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    print(batch_size, sequence_length, num_heads, k_head_dim)
    v_head_dim = value.shape[-1]
    # print("before pad", query.shape, key.shape, value.shape, beta.shape, g.shape)
    
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0,  pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0,  pad_size))
    # print("after pad", query.shape, key.shape, value.shape, beta.shape, g.shape)
    
    tot_seqs = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    
    # Reshape to chunks (in the head dimension)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)


    
    
    
    # # ==================== DEBUG PRINT 2b =====================
    # print("--- Python ref key (AFTER l2, b=0, h=0, chunk=0) ---")
    # print(key[0, 0, 0, :, :chunk_size])
    # # =========================================================

    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    # ... (The rest of the complex logic from the reference implementation)
    # This part is quite intricate and involves parallel scan logic.
    # We will trust the reference implementation as the ground truth.
    g = g.cumsum(dim=-1)

    # ==================== DEBUG PRINT 2b =====================
    print("--- Python ref g (AFTER cumsum, b=0, h=0, chunk=0) ---")
    print(g[0, 0, 0, :], g.shape)
    # =========================================================

    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

    # # ==================== DEBUG PRINT 2b =====================
    # print("--- Python ref decay_mask (AFTER cumsum, b=0, h=0, chunk=0) ---")
    # print(decay_mask.shape)
    # print(decay_mask[0, 0, 0, :, :])
    # # =========================================================

    # # ==================== DEBUG PRINT =====================
    # print("--- Python ref decay_mask (b=0, h=0, chunk=0) ---")
    # print(decay_mask[0, 0, 0, :, :]) # Print for batch 0, head 0, chunk 0
    # # ======================================================

    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

    # for i in range(1, chunk_size):
    #     row = attn[..., i, :i].clone()
    #     sub = attn[..., :i, :i].clone()
    #     attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    
    
    # ==================== DEBUG PRINT 2a =====================
    print("--- Python ref attn (BEFORE scan, b=0, h=0, chunk=0) ---")
    print(attn[0, 0, 0, :, :])
    # =========================================================

    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)


    # ==================== DEBUG PRINT 2b =====================
    print("--- Python ref attn ( torch.eye(chunk_size, dtype=attn.dtype, d) ---")
    print(attn[0, 0, 0, :, :])
    # =========================================================

    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )
    # print("--- Python ref value (b=0, h=0, chunk=0) ---")
    # print(value[0, 0, 0, :, :])
    # # =========================================================
    # print("--- Python ref k_cumdecay (b=0, h=0, chunk=0) ---")
    # print(k_cumdecay[0, 0, 0, :, :])
    # # =========================================================
    print(k_cumdecay.shape)

    core_attn_out = torch.zeros_like(value)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    for i in range(0, tot_seqs // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_intra = (q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_intra @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
        
    core_attn_out = core_attn_out.reshape(core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1])
    core_attn_out = core_attn_out[:, :, :sequence_length] # Unpad
    # core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    core_attn_out = core_attn_out.contiguous().to(initial_dtype)
    
    if last_recurrent_state is not None:
        last_recurrent_state = last_recurrent_state.contiguous().to(initial_dtype)

    return core_attn_out, last_recurrent_state


# ==============================================================================
#  Test Configuration
# ==============================================================================
# (B, T, H, Dk, Dv, chunk_size, use_qk_l2norm)
# T (seq_len) must be > 1 for this operator
_TEST_CASES_ = [
    (2, 511, 40, 64, 64, 8, True),
    # (2, 511, 40, 64, 64, 16, True),
    (4, 1024, 64, 128, 128, 64, False),
    (1, 63, 32, 80, 80, 16, True),
    (8, 2047, 32, 128, 128, 32, True),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

# Tolerance map
_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2}, # Higher tolerance due to complex ops
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-4, "rtol": 1e-4},
}

# Global flags
DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100

def test(
    handle,
    device,
    B, T, H, Dk, Dv, chunk_size, use_qk_l2norm,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing ChunkGatedDeltaRule on {InfiniDeviceNames[device]} with "
        f"B={B}, T={T}, H={H}, Dk={Dk}, Dv={Dv}, chunk_size={chunk_size}, "
        f"dtype={InfiniDtypeNames[dtype]}, use_qk_l2norm={use_qk_l2norm}"
    )

    # # Input tensors are in (B, T, H, D) layout as they come from the model layers
    # q = TestTensor((B, T, H, Dk), None, dtype, device)
    # k = TestTensor((B, T, H, Dk), None, dtype, device)
    # v = TestTensor((B, T, H, Dv), None, dtype, device)
    
    # g_logsigmoid = torch.randn(B, T, H, dtype=torch.float32)
    # g = TestTensor.from_torch(F.logsigmoid(g_logsigmoid), dtype, device)
    # beta_sigmoid = torch.randn(B, T, H, dtype=torch.float32)
    # beta = TestTensor.from_torch(torch.sigmoid(beta_sigmoid), dtype, device)


    # Input tensors are in (B, T, H, D) layout as they come from the model layers
    q = TestTensor((B, H, T, Dk), None, dtype, device)
    k = TestTensor((B, H, T, Dk), None, dtype, device)
    v = TestTensor((B, H, T, Dv), None, dtype, device)
    
    g_logsigmoid = torch.randn(B, H, T, dtype=torch.float32)
    g = TestTensor.from_torch(F.logsigmoid(g_logsigmoid), dtype, device)
    beta_sigmoid = torch.randn(B, H, T, dtype=torch.float32)
    beta = TestTensor.from_torch(torch.sigmoid(beta_sigmoid), dtype, device)

    # q = torch.ones(B, H, T, Dk, dtype=torch.float32, device=device)
    # k = torch.ones(B, H, T, Dk, dtype=torch.float32, device=device)
    # v = torch.ones(B, H, T, Dv, dtype=torch.float32, device=device)
    # g = torch.ones(B, H, T, dtype=torch.float32, device=device)*0.2
    # beta = torch.ones(B, H, T, dtype=torch.float32, device=device)*0.2


    # q = TestTensor.from_torch(q, dtype, device)
    # k = TestTensor.from_torch(k, dtype, device)
    # v = TestTensor.from_torch(v, dtype, device)
    # g = TestTensor.from_torch(g, dtype, device)
    # beta = TestTensor.from_torch(beta, dtype, device)
    
    
    # initial_state shape is (B, H, Dk, Dv) - Note the T dimension
    initial_state = TestTensor((B, H, Dk, Dv), None, dtype, device)

    # Output tensors
    out = TestTensor((B, H, T, Dv), None, dtype, device)
    # final_state shape is (B, H, Dk, Dv)
    final_state = TestTensor((B, H, Dk, Dv), None, dtype, device)

    # Run reference implementation
    ans_out, ans_final_state = ref_chunk_gated_delta_rule(
        q.torch_tensor(), k.torch_tensor(), v.torch_tensor(),
        g.torch_tensor(), beta.torch_tensor(),
        chunk_size=chunk_size,
        initial_state=initial_state.torch_tensor(),
        output_final_state=True, 
        use_qk_l2norm_in_kernel=use_qk_l2norm
    )
    
    if sync:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateChunkGatedDeltaRuleDescriptor(
        handle, ctypes.byref(descriptor),
        out.descriptor, final_state.descriptor,
        q.descriptor, k.descriptor, v.descriptor,
        g.descriptor, beta.descriptor, initial_state.descriptor,
        ctypes.c_bool(use_qk_l2norm),
        ctypes.c_size_t(chunk_size)
    ))
    
    # Get workspace size
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetChunkGatedDeltaRuleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    # Invalidate descriptors
    # ... (destroy all descriptors) ...

    # Define the library call
    def lib_chunk_gated_delta_rule():
        check_error(LIBINFINIOP.infiniopChunkGatedDeltaRule(
            descriptor, workspace.data(), workspace_size.value,
            out.data(), final_state.data(),
            q.data(), k.data(), v.data(),
            g.data(), beta.data(), initial_state.data(), None
        ))
    
    # Execute the custom operator
    lib_chunk_gated_delta_rule()
    
    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    print("atol", atol, "rtol", rtol)
    print("out", out.actual_tensor())
    print("ans_out", ans_out)
    
    if DEBUG:
        print("--- Verifying Output Tensor ---")
        debug(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    
    if DEBUG:
        print("--- Verifying Final State Tensor ---")
        debug(final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol)
    assert torch.allclose(final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol)
    
    # Profiling
    # ... (profiling logic) ...
    
    # Clean up
    check_error(LIBINFINIOP.infiniopDestroyChunkGatedDeltaRuleDescriptor(descriptor))


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
    
    print("\033[92mTest passed!\033[0m")