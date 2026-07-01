# test_recurrent_gated_delta_rule.py

import ctypes
from ctypes import c_uint64

import torch
import torch.nn.functional as F

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


def ref_recurrent_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    output_final_state,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    initial_state = (
        initial_state.transpose(-1, -2).contiguous().to(torch.float32).clone()
    )

    batch_size, sequence_length, key_heads, k_head_dim = key.shape
    value_heads, v_head_dim = value.shape[2], value.shape[-1]
    value_heads_per_key_head = value_heads // key_heads
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    core_attn_out = torch.zeros(
        batch_size,
        sequence_length,
        value_heads,
        v_head_dim,
        device=value.device,
        dtype=torch.float32,
    )
    last_recurrent_state = initial_state

    for i in range(sequence_length):
        for vh in range(value_heads):
            kh = vh // value_heads_per_key_head
            q_t = query[:, i, kh]
            k_t = key[:, i, kh]
            v_t = value[:, i, vh]
            g_t = g[:, i, vh].exp().view(batch_size, 1, 1)
            beta_t = beta[:, i, vh].view(batch_size, 1)
            state_t = last_recurrent_state[:, vh]

            state_t = state_t * g_t
            kv_mem = (state_t * k_t.unsqueeze(-1)).sum(dim=-2)
            delta = (v_t - kv_mem) * beta_t
            state_t = state_t + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
            last_recurrent_state[:, vh] = state_t
            core_attn_out[:, i, vh] = (state_t * q_t.unsqueeze(-1)).sum(dim=-2)

    if not output_final_state:
        last_recurrent_state = None

    core_attn_out = core_attn_out.contiguous().to(initial_dtype)
    if last_recurrent_state is not None:
        last_recurrent_state = (
            last_recurrent_state.transpose(-1, -2).contiguous().to(initial_dtype)
        )

    return core_attn_out, last_recurrent_state


_TEST_CASES_ = [
    (7, 1, 40, 40, 128, 128, True, False),
    (5, 1, 64, 64, 128, 128, False, False),
    (1, 1, 8, 8, 64, 64, True, False),
    (2, 1, 4, 8, 64, 64, False, False),
    (2, 1, 4, 8, 64, 64, True, True),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-3, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-3, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-5, "rtol": 1e-5},
}

DEBUG = False
PROFILE = False
NUM_PRERUN = 10
NUM_ITERATIONS = 100


def bthd_strides(B, T, H, D, strided):
    if not strided:
        return None
    return (T * H * D * 2, H * D * 2, D * 2, 1)


def make_gate(shape, device):
    return TestTensor.from_torch(
        F.logsigmoid(torch.randn(*shape, dtype=torch.float32)), InfiniDtype.F32, device
    )


def make_beta(shape, device):
    return TestTensor.from_torch(
        torch.sigmoid(torch.randn(*shape, dtype=torch.float32)), InfiniDtype.F32, device
    )


def test(
    handle,
    device,
    B,
    T,
    Hk,
    Hv,
    Dk,
    Dv,
    use_qk_l2norm,
    strided_qkv,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RecurrentGatedDeltaRule on {InfiniDeviceNames[device]} with "
        f"B={B}, T={T}, Hk={Hk}, Hv={Hv}, Dk={Dk}, Dv={Dv}, "
        f"dtype={InfiniDtypeNames[dtype]}, gate_dtype=F32, strided_qkv={strided_qkv}, "
        f"use_qk_l2norm={use_qk_l2norm}"
    )

    q = TestTensor(
        (B, T, Hk, Dk), bthd_strides(B, T, Hk, Dk, strided_qkv), dtype, device
    )
    k = TestTensor(
        (B, T, Hk, Dk), bthd_strides(B, T, Hk, Dk, strided_qkv), dtype, device
    )
    v = TestTensor(
        (B, T, Hv, Dv), bthd_strides(B, T, Hv, Dv, strided_qkv), dtype, device
    )
    g = make_gate((B, T, Hv), device)
    beta = make_beta((B, T, Hv), device)

    initial_state = TestTensor((B, Hv, Dv, Dk), None, dtype, device)
    out = TestTensor(
        (B, T, Hv, Dv),
        bthd_strides(B, T, Hv, Dv, strided_qkv),
        dtype,
        device,
        mode="zeros",
    )
    final_state = TestTensor((B, Hv, Dv, Dk), None, dtype, device)

    ans_out, ans_final_state = ref_recurrent_gated_delta_rule(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        g.torch_tensor(),
        beta.torch_tensor(),
        initial_state.torch_tensor(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRecurrentGatedDeltaRuleDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            initial_state.descriptor,
            final_state.descriptor,
            q.descriptor,
            k.descriptor,
            v.descriptor,
            g.descriptor,
            beta.descriptor,
            ctypes.c_void_p(0),
            ctypes.c_void_p(0),
            ctypes.c_bool(use_qk_l2norm),
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    q.destroy_desc()
    k.destroy_desc()
    v.destroy_desc()
    g.destroy_desc()
    beta.destroy_desc()
    initial_state.destroy_desc()
    out.destroy_desc()
    final_state.destroy_desc()

    def lib_recurrent_gated_delta_rule():
        check_error(
            LIBINFINIOP.infiniopRecurrentGatedDeltaRule(
                descriptor,
                workspace.data(),
                workspace_size.value,
                out.data(),
                initial_state.data(),
                final_state.data(),
                q.data(),
                k.data(),
                v.data(),
                g.data(),
                beta.data(),
                None,
                None,
                None,
            )
        )

    lib_recurrent_gated_delta_rule()

    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
        debug(final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    assert torch.allclose(
        final_state.actual_tensor(), ans_final_state, atol=atol, rtol=rtol
    )

    if PROFILE:
        profile_operation(
            "PyTorch",
            lambda: ref_recurrent_gated_delta_rule(
                q.torch_tensor(),
                k.torch_tensor(),
                v.torch_tensor(),
                g.torch_tensor(),
                beta.torch_tensor(),
                initial_state.torch_tensor(),
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm,
            ),
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )
        profile_operation(
            "    lib",
            lib_recurrent_gated_delta_rule,
            device,
            NUM_PRERUN,
            NUM_ITERATIONS,
        )

    check_error(
        LIBINFINIOP.infiniopDestroyRecurrentGatedDeltaRuleDescriptor(descriptor)
    )


def test_indexed_pool_inplace(
    handle,
    device,
    B,
    T,
    Hk,
    Hv,
    Dk,
    Dv,
    use_qk_l2norm,
    strided_qkv,
    dtype=InfiniDtype.F16,
    sync=None,
):
    print(
        f"Testing RecurrentGatedDeltaRule indexed pool inplace on {InfiniDeviceNames[device]} with "
        f"B={B}, T={T}, Hk={Hk}, Hv={Hv}, Dk={Dk}, Dv={Dv}, "
        f"dtype={InfiniDtypeNames[dtype]}, gate_dtype=F32, strided_qkv={strided_qkv}, "
        f"use_qk_l2norm={use_qk_l2norm}"
    )

    q = TestTensor(
        (B, T, Hk, Dk), bthd_strides(B, T, Hk, Dk, strided_qkv), dtype, device
    )
    k = TestTensor(
        (B, T, Hk, Dk), bthd_strides(B, T, Hk, Dk, strided_qkv), dtype, device
    )
    v = TestTensor(
        (B, T, Hv, Dv), bthd_strides(B, T, Hv, Dv, strided_qkv), dtype, device
    )
    g = make_gate((B, T, Hv), device)
    beta = make_beta((B, T, Hv), device)

    pool_size = B * 2 + 3
    initial_state_pool = TestTensor((pool_size, Hv, Dv, Dk), None, dtype, device)
    index_device = q.torch_tensor().device
    initial_state_indices_torch = torch.arange(
        1, B + 1, dtype=torch.int64, device=index_device
    )
    final_state_indices_torch = torch.arange(
        B + 1, 2 * B + 1, dtype=torch.int64, device=index_device
    )
    initial_state_indices = TestTensor.from_torch(
        initial_state_indices_torch, InfiniDtype.I64, device
    )
    final_state_indices = TestTensor.from_torch(
        final_state_indices_torch, InfiniDtype.I64, device
    )

    out = TestTensor(
        (B, T, Hv, Dv),
        bthd_strides(B, T, Hv, Dv, strided_qkv),
        dtype,
        device,
        mode="zeros",
    )

    gathered_initial_state = initial_state_pool.torch_tensor()[
        initial_state_indices_torch
    ].contiguous()
    ans_out, ans_final_state = ref_recurrent_gated_delta_rule(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        g.torch_tensor(),
        beta.torch_tensor(),
        gathered_initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    ans_initial_state_pool = initial_state_pool.torch_tensor().clone()
    ans_initial_state_pool[final_state_indices_torch] = ans_final_state.contiguous()

    if sync:
        sync()

    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateRecurrentGatedDeltaRuleDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            initial_state_pool.descriptor,
            ctypes.c_void_p(0),
            q.descriptor,
            k.descriptor,
            v.descriptor,
            g.descriptor,
            beta.descriptor,
            initial_state_indices.descriptor,
            final_state_indices.descriptor,
            ctypes.c_bool(use_qk_l2norm),
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetRecurrentGatedDeltaRuleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    q.destroy_desc()
    k.destroy_desc()
    v.destroy_desc()
    g.destroy_desc()
    beta.destroy_desc()
    initial_state_pool.destroy_desc()
    initial_state_indices.destroy_desc()
    final_state_indices.destroy_desc()
    out.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopRecurrentGatedDeltaRule(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            initial_state_pool.data(),
            None,
            q.data(),
            k.data(),
            v.data(),
            g.data(),
            beta.data(),
            initial_state_indices.data(),
            final_state_indices.data(),
            None,
        )
    )

    if sync:
        sync()

    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
        debug(
            initial_state_pool.actual_tensor(),
            ans_initial_state_pool,
            atol=atol,
            rtol=rtol,
        )
    assert torch.allclose(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    assert torch.allclose(
        initial_state_pool.actual_tensor(), ans_initial_state_pool, atol=atol, rtol=rtol
    )

    check_error(
        LIBINFINIOP.infiniopDestroyRecurrentGatedDeltaRuleDescriptor(descriptor)
    )


if __name__ == "__main__":
    args = get_args()

    DEBUG = args.debug
    PROFILE = args.profile
    NUM_PRERUN = args.num_prerun
    NUM_ITERATIONS = args.num_iterations

    for device in get_test_devices(args):
        test_operator(device, test, _TEST_CASES_, _TENSOR_DTYPES)
        test_operator(device, test_indexed_pool_inplace, _TEST_CASES_, _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
