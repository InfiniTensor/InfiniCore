# test_recurrent_gated_delta_rule.py

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
# 从 modeling_qwen3_next.py 提供的生产环境PyTorch备选实现
# 我们将严格对照此函数进行测试
def ref_recurrent_gated_delta_rule(
    query, key, value, g, beta, initial_state, output_final_state, use_qk_l2norm_in_kernel=False
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)
    
    # 生产环境的实现期望输入已经是 (B, H, T, D)
    # 我们在测试数据生成时会直接生成这种格式，以模拟真实调用场景
    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(batch_size, num_heads, sequence_length, v_head_dim, device=value.device, dtype=torch.float32)
    
    # 注意：这里的 initial_state 形状是 (B, H, Dk, Dv)
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, device=value.device, dtype=torch.float32)
        if initial_state is None
        else initial_state.to(torch.float32)
    )

    for i in range(sequence_length):
        q_t = query[:, :, i]
        k_t = key[:, :, i]
        v_t = value[:, :, i]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)
        beta_t = beta[:, :, i].unsqueeze(-1)

        last_recurrent_state = last_recurrent_state * g_t
        kv_mem = (last_recurrent_state * k_t.unsqueeze(-1)).sum(dim=-2)
        delta = (v_t - kv_mem) * beta_t
        last_recurrent_state = last_recurrent_state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
        core_attn_out[:, :, i] = (last_recurrent_state * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None
        
    core_attn_out = core_attn_out.contiguous().to(initial_dtype)
    if last_recurrent_state is not None:
        last_recurrent_state = last_recurrent_state.contiguous().to(initial_dtype)
        
    return core_attn_out, last_recurrent_state


# ==============================================================================
#  Test Configuration
# ==============================================================================
# (B, T, H, Dk, Dv, use_qk_l2norm)
# T (seq_len) is typically 1 for decode stage
_TEST_CASES_ = [
    (7, 1, 40, 128, 128, True),
    (5, 1, 64, 128, 128, False),
    (1, 1, 8, 64, 64, True),
    # (16, 1, 32, 80, 80, True),
]

# Data types for testing
_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

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
NUM_ITERATIONS = 100

def test(
    handle,
    device,
    B, T, H, Dk, Dv, use_qk_l2norm,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RecurrentGatedDeltaRule on {InfiniDeviceNames[device]} with "
        f"B={B}, T={T}, H={H}, Dk={Dk}, Dv={Dv}, dtype={InfiniDtypeNames[dtype]}, "
        f"use_qk_l2norm={use_qk_l2norm}"
    )

    # Create input tensors. 
    # IMPORTANT: We directly create tensors in (B, H, T, D) layout to match the production environment.
    q = TestTensor((B, H, T, Dk), None, dtype, device)
    k = TestTensor((B, H, T, Dk), None, dtype, device)
    v = TestTensor((B, H, T, Dv), None, dtype, device)
    # g and beta have shape (B, H, T)
    g_logsigmoid = torch.randn(B, H, T, dtype=torch.float32)
    g = TestTensor.from_torch(F.logsigmoid(g_logsigmoid), dtype, device)
    beta_sigmoid = torch.randn(B, H, T, dtype=torch.float32)
    beta = TestTensor.from_torch(torch.sigmoid(beta_sigmoid), dtype, device)
    
    initial_state = TestTensor((B, H, Dk, Dv), None, dtype, device)

    # Create output tensors
    out = TestTensor((B, H, T, Dv), None, dtype, device)
    final_state = TestTensor((B, H, Dk, Dv), None, dtype, device)

    # Run reference implementation
    ans_out, ans_final_state = ref_recurrent_gated_delta_rule(
        q.torch_tensor(), k.torch_tensor(), v.torch_tensor(),
        g.torch_tensor(), beta.torch_tensor(), initial_state.torch_tensor(),
        output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm)
    
    if sync:
        sync()

    # Create operator descriptor
    descriptor = infiniopOperatorDescriptor_t()
    check_error(LIBINFINIOP.infiniopCreateRecurrentGatedDeltaRuleDescriptor(
        handle, ctypes.byref(descriptor),
        out.descriptor, final_state.descriptor,
        q.descriptor, k.descriptor, v.descriptor,
        g.descriptor, beta.descriptor, initial_state.descriptor,
        ctypes.c_bool(use_qk_l2norm)
    ))
    
    # Get workspace size and allocate memory
    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    # Invalidate descriptors to ensure kernel does not rely on them
    q.destroy_desc()
    k.destroy_desc()
    v.destroy_desc()
    g.destroy_desc()
    beta.destroy_desc()
    initial_state.destroy_desc()
    out.destroy_desc()
    final_state.destroy_desc()

    # Define the library call as a lambda for profiling
    def lib_recurrent_gated_delta_rule():
        check_error(LIBINFINIOP.infiniopRecurrentGatedDeltaRule(
            descriptor, workspace.data(), workspace_size.value,
            out.data(), final_state.data(),
            q.data(), k.data(), v.data(),
            g.data(), beta.data(), initial_state.data(), None
        ))
    
    # Execute the custom operator
    lib_recurrent_gated_delta_rule()
    
    if sync:
        sync()

    # Verify correctness
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    
    # Verify main output
    if DEBUG:
        print("--- Verifying Output Tensor ---")
        debug(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    
    # Verify final state
    if DEBUG:
        print("--- Verifying Final State Tensor ---")
        debug(final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol)
    assert torch.allclose(final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol)
    # print(final_state.actual_tensor(), ans_final_state)
    
    # Profiling workflow
    if PROFILE:
        # fmt: off
        profile_operation("PyTorch", lambda: ref_recurrent_gated_delta_rule(
            q.torch_tensor(), k.torch_tensor(), v.torch_tensor(),
            g.torch_tensor(), beta.torch_tensor(), initial_state.torch_tensor(),
            output_final_state=True, use_qk_l2norm_in_kernel=use_qk_l2norm),
            device, NUM_PRERUN, NUM_ITERATIONS)
        profile_operation("    lib", lib_recurrent_gated_delta_rule, device, NUM_PRERUN, NUM_ITERATIONS)
        # fmt: on
    
    # Clean up resources
    check_error(LIBINFINIOP.infiniopDestroyRecurrentGatedDeltaRuleDescriptor(descriptor))


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