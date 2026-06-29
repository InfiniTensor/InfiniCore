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
    InfiniDtype,
    InfiniDtypeNames,
    InfiniDeviceNames,
    infiniopOperatorDescriptor_t,
    TestWorkspace,
)


def ref_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    initial_state,
    cu_seqlens=None,
    use_qk_l2norm_in_kernel=False,
):
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = F.normalize(query, p=2, dim=-1)
        key = F.normalize(key, p=2, dim=-1)

    query, key, value, beta, g = [
        x.contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]
    state = initial_state.transpose(-1, -2).contiguous().to(torch.float32).clone()

    if cu_seqlens is None:
        batch_size, sequence_length, key_heads, k_head_dim = key.shape
        spans = [(b, 0, sequence_length) for b in range(batch_size)]
    else:
        key_heads, k_head_dim = key.shape[2], key.shape[3]
        batch_size = cu_seqlens.numel() - 1
        spans = [
            (b, int(cu_seqlens[b].item()), int(cu_seqlens[b + 1].item()))
            for b in range(batch_size)
        ]

    value_heads, v_head_dim = value.shape[2], value.shape[3]
    value_heads_per_key_head = value_heads // key_heads
    scale = 1 / (k_head_dim**0.5)
    query = query * scale
    out = torch.zeros_like(value, dtype=torch.float32)

    for b, start, end in spans:
        for vh in range(value_heads):
            kh = vh // value_heads_per_key_head
            state_t = state[b, vh]
            for t in range(start, end):
                token_b = 0 if cu_seqlens is not None else b
                q_t = query[token_b, t, kh]
                k_t = key[token_b, t, kh]
                v_t = value[token_b, t, vh]
                g_t = g[token_b, t, vh].exp()
                beta_t = beta[token_b, t, vh]

                state_t = state_t * g_t
                kv_mem = (state_t * k_t.unsqueeze(-1)).sum(dim=-2)
                delta = (v_t - kv_mem) * beta_t
                state_t = state_t + k_t.unsqueeze(-1) * delta.unsqueeze(-2)
                state[b, vh] = state_t
                out[token_b, t, vh] = (state_t * q_t.unsqueeze(-1)).sum(dim=-2)

    return out.contiguous().to(initial_dtype), state.transpose(-1, -2).contiguous().to(
        initial_dtype
    )


_PADDED_TEST_CASES_DATA = [
    # B, T, n_khead, kdim, n_vhead, vdim, chunk_size, use_qk_l2norm, strided_qkv
    (2, 17, 4, 64, 4, 64, 8, True, False),
    (2, 19, 4, 64, 8, 64, 8, False, False),
    (2, 13, 4, 64, 8, 64, 8, True, True),
]

# Test cases: (n_khead, kdim, n_vhead, vdim, (seqlens), (init_state_indices),
# final_state_indices, state_pool_size)
_VARLEN_TEST_CASES_DATA = [
    (4, 64, 8, 64, (1, 17, 3, 9), (1, 2, 3, 4), (5, 6, 7, 8), 13),
    (16, 128, 48, 128, (13,), (0,), (0,), 1),
]

_TENSOR_DTYPES = [InfiniDtype.F16, InfiniDtype.BF16, InfiniDtype.F32]

_TOLERANCE_MAP = {
    InfiniDtype.F16: {"atol": 1e-2, "rtol": 1e-2},
    InfiniDtype.BF16: {"atol": 5e-2, "rtol": 5e-2},
    InfiniDtype.F32: {"atol": 1e-3, "rtol": 1e-3},
}

DEBUG = False


def parse_test_cases():
    tests = []
    for case in _PADDED_TEST_CASES_DATA:
        tests.append(("padded", case))
    for case in _VARLEN_TEST_CASES_DATA:
        tests.append(("varlen_indexed_pool", case))
    return tests


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


def run_op(
    handle,
    device,
    out,
    initial_state,
    final_state,
    q,
    k,
    v,
    g,
    beta,
    cu_seqlens,
    initial_state_indices,
    final_state_indices,
    use_qk_l2norm,
    chunk_size,
):
    descriptor = infiniopOperatorDescriptor_t()
    check_error(
        LIBINFINIOP.infiniopCreateChunkGatedDeltaRuleDescriptor(
            handle,
            ctypes.byref(descriptor),
            out.descriptor,
            initial_state.descriptor,
            final_state.descriptor if final_state is not None else ctypes.c_void_p(0),
            q.descriptor,
            k.descriptor,
            v.descriptor,
            g.descriptor,
            beta.descriptor,
            cu_seqlens.descriptor if cu_seqlens is not None else ctypes.c_void_p(0),
            initial_state_indices.descriptor
            if initial_state_indices is not None
            else ctypes.c_void_p(0),
            final_state_indices.descriptor
            if final_state_indices is not None
            else ctypes.c_void_p(0),
            ctypes.c_bool(use_qk_l2norm),
            ctypes.c_size_t(chunk_size),
        )
    )

    workspace_size = c_uint64(0)
    check_error(
        LIBINFINIOP.infiniopGetChunkGatedDeltaRuleWorkspaceSize(
            descriptor, ctypes.byref(workspace_size)
        )
    )
    workspace = TestWorkspace(workspace_size.value, q.device)

    for tensor in [
        out,
        initial_state,
        final_state,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
    ]:
        if tensor is not None:
            tensor.destroy_desc()

    check_error(
        LIBINFINIOP.infiniopChunkGatedDeltaRule(
            descriptor,
            workspace.data(),
            workspace_size.value,
            out.data(),
            initial_state.data(),
            final_state.data() if final_state is not None else None,
            q.data(),
            k.data(),
            v.data(),
            g.data(),
            beta.data(),
            cu_seqlens.data() if cu_seqlens is not None else None,
            initial_state_indices.data() if initial_state_indices is not None else None,
            final_state_indices.data() if final_state_indices is not None else None,
            None,
        )
    )
    check_error(LIBINFINIOP.infiniopDestroyChunkGatedDeltaRuleDescriptor(descriptor))


def test_padded(
    handle,
    device,
    test_case,
    dtype=InfiniDtype.F16,
    sync=None,
):
    B, T, Hk, Dk, Hv, Dv, chunk_size, use_qk_l2norm, strided_qkv = test_case
    print(
        f"Testing ChunkGatedDeltaRule on {InfiniDeviceNames[device]} with "
        f"B={B}, T={T}, Hk={Hk}, Hv={Hv}, Dk={Dk}, Dv={Dv}, chunk={chunk_size}, "
        f"dtype={InfiniDtypeNames[dtype]}, gate_dtype=F32, strided_qkv={strided_qkv}, "
        f"l2norm={use_qk_l2norm}"
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
    final_state = TestTensor((B, Hv, Dv, Dk), None, dtype, device)
    out = TestTensor(
        (B, T, Hv, Dv),
        bthd_strides(B, T, Hv, Dv, strided_qkv),
        dtype,
        device,
        mode="zeros",
    )

    ans_out, ans_final_state = ref_chunk_gated_delta_rule(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        g.torch_tensor(),
        beta.torch_tensor(),
        initial_state.torch_tensor(),
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    if sync:
        sync()

    run_op(
        handle,
        device,
        out,
        initial_state,
        final_state,
        q,
        k,
        v,
        g,
        beta,
        None,
        None,
        None,
        use_qk_l2norm,
        chunk_size,
    )

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


def test_varlen_indexed_pool(
    handle,
    device,
    test_case,
    dtype=InfiniDtype.F16,
    sync=None,
):
    (
        Hk,
        Dk,
        Hv,
        Dv,
        lengths,
        initial_state_indices_data,
        final_state_indices_data,
        pool_size,
    ) = test_case
    chunk_size = 8
    use_qk_l2norm = True
    lengths = tuple(lengths)
    initial_state_indices_data = tuple(initial_state_indices_data)
    final_state_indices_data = tuple(final_state_indices_data)
    B = len(lengths)
    total_tokens = sum(lengths)
    print(
        f"Testing ChunkGatedDeltaRule varlen indexed pool on {InfiniDeviceNames[device]} with "
        f"lengths={lengths}, Hk={Hk}, Hv={Hv}, Dk={Dk}, Dv={Dv}, chunk={chunk_size}, "
        f"dtype={InfiniDtypeNames[dtype]}, gate_dtype=F32, "
        f"initial_indices={initial_state_indices_data}, final_indices={final_state_indices_data}, "
        f"pool_size={pool_size}"
    )
    q = TestTensor((1, total_tokens, Hk, Dk), None, dtype, device)
    k = TestTensor((1, total_tokens, Hk, Dk), None, dtype, device)
    v = TestTensor((1, total_tokens, Hv, Dv), None, dtype, device)
    g = make_gate((1, total_tokens, Hv), device)
    beta = make_beta((1, total_tokens, Hv), device)
    out = TestTensor((1, total_tokens, Hv, Dv), None, dtype, device, mode="zeros")

    cu = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        dtype=torch.int64,
        device=q.torch_tensor().device,
    )
    cu_seqlens = TestTensor.from_torch(cu, InfiniDtype.I64, device)
    initial_state_pool = TestTensor((pool_size, Hv, Dv, Dk), None, dtype, device)
    initial_state_indices_torch = torch.tensor(
        initial_state_indices_data, dtype=torch.int64, device=q.torch_tensor().device
    )
    final_state_indices_torch = torch.tensor(
        final_state_indices_data, dtype=torch.int64, device=q.torch_tensor().device
    )
    initial_state_indices = TestTensor.from_torch(
        initial_state_indices_torch, InfiniDtype.I64, device
    )
    final_state_indices = TestTensor.from_torch(
        final_state_indices_torch, InfiniDtype.I64, device
    )

    gathered_initial = initial_state_pool.torch_tensor()[
        initial_state_indices_torch
    ].contiguous()
    ans_out, ans_final_state = ref_chunk_gated_delta_rule(
        q.torch_tensor(),
        k.torch_tensor(),
        v.torch_tensor(),
        g.torch_tensor(),
        beta.torch_tensor(),
        gathered_initial,
        cu_seqlens=cu,
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    ans_pool = initial_state_pool.torch_tensor().clone()
    ans_pool[final_state_indices_torch] = ans_final_state.contiguous()
    if sync:
        sync()

    run_op(
        handle,
        device,
        out,
        initial_state_pool,
        None,
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        use_qk_l2norm,
        chunk_size,
    )

    if sync:
        sync()
    atol, rtol = get_tolerance(_TOLERANCE_MAP, dtype)
    if DEBUG:
        debug(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
        debug(initial_state_pool.actual_tensor(), ans_pool, atol=atol, rtol=rtol)
    assert torch.allclose(out.actual_tensor(), ans_out, atol=atol, rtol=rtol)
    assert torch.allclose(
        initial_state_pool.actual_tensor(), ans_pool, atol=atol, rtol=rtol
    )


def test(
    handle,
    device,
    mode,
    test_case,
    dtype=InfiniDtype.F16,
    sync=None,
):
    if mode == "padded":
        return test_padded(handle, device, test_case, dtype=dtype, sync=sync)
    if mode == "varlen_indexed_pool":
        return test_varlen_indexed_pool(
            handle, device, test_case, dtype=dtype, sync=sync
        )
    raise ValueError(f"Unknown chunk_gated_delta_rule test mode: {mode}")


if __name__ == "__main__":
    args = get_args()
    DEBUG = args.debug

    for device in get_test_devices(args):
        test_operator(device, test, parse_test_cases(), _TENSOR_DTYPES)

    print("\033[92mTest passed!\033[0m")
