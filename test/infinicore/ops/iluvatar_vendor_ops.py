import torch
from vllm_iluvatar.custom_kernels.argsort_bincount_with_inv_pos import (
    argsort_bincount_with_inv_pos,
)
from vllm_iluvatar.custom_kernels.concat_and_cache_mla import concat_and_cache_mla
from vllm_iluvatar.custom_kernels.concat_mla_q import concat_mla_q
from vllm_iluvatar.custom_kernels.dynamic_scaled_int8_quant import (
    dynamic_scaled_int8_quant,
)
from vllm_iluvatar.custom_kernels.grouped_topk import grouped_topk
from vllm_iluvatar.custom_kernels.moe_sum import moe_sum
from vllm_iluvatar.custom_kernels.rotary_embedding import rotary_embedding
from vllm_iluvatar.custom_kernels.scaled_mm_w4a8 import scaled_mm_w4a8
from vllm_iluvatar.custom_kernels.topk_sigmoid import topk_sigmoid
from vllm_iluvatar.custom_kernels.topk_softmax import topk_softmax
from vllm_iluvatar.custom_kernels.w4a8_group_gemm import w4a8_group_gemm
from vllm_iluvatar.custom_kernels.w8a8_group_gemm import w8a8_group_gemm
from vllm_iluvatar.custom_kernels.w16a16_group_gemm import w16a16_group_gemm

import infinicore

DEVICE = "cuda"
DTYPE = torch.bfloat16


def ic(tensor):
    return infinicore.from_torch(tensor)


def close(actual, expected, *, rtol=1e-2, atol=1e-2):
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def test_add_rms_norm_inplace():
    x = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device=DEVICE, dtype=DTYPE)
    expected_residual = x + residual
    expected = (
        expected_residual.float()
        * torch.rsqrt(expected_residual.float().pow(2).mean(-1, keepdim=True) + 1e-5)
        * weight.float()
    ).to(DTYPE)
    infinicore.add_rms_norm_inplace(ic(x), ic(residual), ic(weight))
    close(residual, expected_residual)
    close(x, expected)


def test_mla_concat_and_cache():
    ql = torch.randn(2, 8, 512, device=DEVICE, dtype=DTYPE)
    qpe = torch.randn(2, 8, 64, device=DEVICE, dtype=DTYPE)
    q_ref = torch.empty(2, 8, 576, device=DEVICE, dtype=DTYPE)
    q_actual = torch.empty_like(q_ref)
    concat_mla_q(ql, qpe, q_ref)
    infinicore.concat_mla_q(ic(ql), ic(qpe), out=ic(q_actual))
    close(q_actual, q_ref)

    kv = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
    kpe = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)
    cache_ref = torch.zeros(2, 64, 576, device=DEVICE, dtype=DTYPE)
    cache_out = cache_ref.clone()
    concat_and_cache_mla(kv, kpe, cache_ref, slots, "auto", scale)
    infinicore.concat_and_cache_mla(
        ic(kv), ic(kpe), ic(cache_out), ic(slots), "auto", ic(scale)
    )
    close(cache_out, cache_ref)


def test_mla_int8_cache():
    kv = torch.randint(-100, 100, (2, 512), device=DEVICE, dtype=torch.int8)
    kpe = torch.randint(-100, 100, (2, 64), device=DEVICE, dtype=torch.int8)
    kv_scale = torch.rand(2, device=DEVICE)
    kpe_scale = torch.rand(2, device=DEVICE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    cache = torch.zeros(2, 64, 576, device=DEVICE, dtype=torch.int8)
    cache_scale = torch.zeros(2, 64, 2, device=DEVICE)
    infinicore.concat_and_cache_mla_int8(
        ic(kv),
        ic(kv_scale),
        ic(kpe),
        ic(kpe_scale),
        ic(cache),
        ic(cache_scale),
        ic(slots),
    )
    infinicore.sync_device()
    flat_cache = cache.view(-1, 576)
    flat_scale = cache_scale.view(-1, 2)
    torch.testing.assert_close(flat_cache[slots], torch.cat((kv, kpe), dim=-1))
    torch.testing.assert_close(
        flat_scale[slots], torch.stack((kv_scale, kpe_scale), dim=-1)
    )


def test_dynamic_int8_quant():
    x = torch.randn(8, 128, device=DEVICE, dtype=DTYPE)
    ref = torch.empty_like(x, dtype=torch.int8)
    ref_scale = torch.empty(8, device=DEVICE)
    dynamic_scaled_int8_quant(ref, x, ref_scale)
    out = torch.empty_like(ref)
    scale = torch.empty_like(ref_scale)
    infinicore.dynamic_scaled_int8_quant(ic(x), ic(scale), out=ic(out))
    close(out, ref, rtol=0, atol=0)
    close(scale, ref_scale, rtol=1e-5, atol=1e-6)


def test_allocating_wrappers():
    ql = torch.randn(2, 8, 512, device=DEVICE, dtype=DTYPE)
    qpe = torch.randn(2, 8, 64, device=DEVICE, dtype=DTYPE)
    q_ref = torch.empty(2, 8, 576, device=DEVICE, dtype=DTYPE)
    concat_mla_q(ql, qpe, q_ref)
    q_result = infinicore.concat_mla_q(ic(ql), ic(qpe))
    q_actual = torch.empty_like(q_ref)
    ic(q_actual).copy_(q_result)
    close(q_actual, q_ref)

    x = torch.randn(8, 128, device=DEVICE, dtype=DTYPE)
    quant_ref = torch.empty_like(x, dtype=torch.int8)
    scale_ref = torch.empty(8, device=DEVICE)
    dynamic_scaled_int8_quant(quant_ref, x, scale_ref)
    scale_actual = torch.empty_like(scale_ref)
    quant_result = infinicore.dynamic_scaled_int8_quant(ic(x), ic(scale_actual))
    quant_actual = torch.empty_like(quant_ref)
    ic(quant_actual).copy_(quant_result)
    close(quant_actual, quant_ref, rtol=0, atol=0)
    close(scale_actual, scale_ref, rtol=1e-5, atol=1e-6)

    m, n, k = 16, 128, 128
    a = torch.randint(-8, 8, (m, k), device=DEVICE, dtype=torch.int8)
    b4 = torch.randint(-128, 127, (k, n // 2), device=DEVICE, dtype=torch.int8)
    b8 = torch.randint(-8, 8, (n, k), device=DEVICE, dtype=torch.int8)
    a_scale = torch.rand(m, 1, device=DEVICE)
    b_scale = torch.rand(n, 1, device=DEVICE)
    ref4 = torch.empty(m, n, device=DEVICE, dtype=torch.float16)
    scaled_mm_w4a8(ref4, a, b4, a_scale, b_scale)
    result4 = infinicore.scaled_mm_w4a8(
        ic(a), ic(b4), ic(a_scale), ic(b_scale), trans_weight=False
    )
    actual4 = torch.empty_like(ref4)
    ic(actual4).copy_(result4)
    close(actual4, ref4)

    result8 = infinicore.scaled_mm_w8a8(
        ic(a), ic(b8), ic(a_scale), ic(b_scale), trans_weight=True
    )
    actual8 = torch.empty_like(ref4)
    ic(actual8).copy_(result8)
    expected8 = ((a.float() * a_scale) @ (b8.float() * b_scale).transpose(0, 1)).half()
    close(actual8, expected8, rtol=5e-2, atol=1.0)


def test_rotary_embedding():
    positions = torch.tensor([0, 3], device=DEVICE, dtype=torch.int64)
    query_ref = torch.randn(2, 8, 128, device=DEVICE, dtype=DTYPE)
    key_ref = torch.randn(2, 2, 128, device=DEVICE, dtype=DTYPE)
    query_out, key_out = query_ref.clone(), key_ref.clone()
    cache = torch.randn(16, 128, device=DEVICE, dtype=DTYPE)
    rotary_embedding(
        positions, query_ref.view(2, -1), key_ref.view(2, -1), 128, cache, True
    )
    infinicore.fused_rotary_embedding_(
        ic(query_out), ic(key_out), ic(positions), 128, ic(cache), True
    )
    close(query_out, query_ref)
    close(key_out, key_ref)


def test_routing_ops():
    scores = torch.randn(4, 64, device=DEVICE, dtype=DTYPE)
    bias = torch.randn(64, device=DEVICE, dtype=DTYPE)
    ref_w, ref_i = grouped_topk(scores, 8, 4, 4, True, 1.25, bias, 1)
    actual_w = torch.empty_like(ref_w)
    actual_i = torch.empty_like(ref_i)
    infinicore.grouped_topk_vendor(
        ic(scores),
        8,
        4,
        4,
        True,
        1.25,
        ic(bias),
        "sigmoid",
        out=(ic(actual_w), ic(actual_i)),
    )
    close(actual_w, ref_w)
    close(actual_i, ref_i, rtol=0, atol=0)

    for reference, candidate in (
        (topk_softmax, infinicore.moe_topk_softmax_vendor),
        (topk_sigmoid, infinicore.moe_topk_sigmoid_vendor),
    ):
        ref_w = torch.empty(4, 4, device=DEVICE)
        ref_i = torch.empty(4, 4, device=DEVICE, dtype=torch.int32)
        ref_t = torch.empty_like(ref_i)
        reference(ref_w, ref_i, ref_t, scores, True, bias)
        actual_w = torch.empty_like(ref_w)
        actual_i = torch.empty_like(ref_i)
        actual_t = torch.empty_like(ref_t)
        candidate(
            ic(scores),
            4,
            renormalize=True,
            correction_bias=ic(bias),
            out=(ic(actual_w), ic(actual_i), ic(actual_t)),
        )
        close(actual_w, ref_w)
        close(actual_i, ref_i, rtol=0, atol=0)
        assert torch.all((actual_t >= 0) & (actual_t < 64))


def test_moe_data_movement_and_sum():
    topk_ids = torch.tensor(
        [[3, 1], [0, 3], [2, 1], [0, 2]], device=DEVICE, dtype=torch.int32
    )
    ref_tpe, ref_sorted, ref_inv = argsort_bincount_with_inv_pos(topk_ids, 4)
    tpe = torch.empty(4, device=DEVICE, dtype=torch.int32)
    sorted_ids = torch.empty(8, device=DEVICE, dtype=torch.int32)
    inv = torch.empty_like(sorted_ids)
    infinicore.moe_argsort_bincount_with_inv_pos_(
        ic(tpe), ic(sorted_ids), ic(inv), ic(topk_ids), 4
    )
    close(tpe, ref_tpe.to(DEVICE), rtol=0, atol=0)
    close(sorted_ids, ref_sorted, rtol=0, atol=0)
    close(inv, ref_inv, rtol=0, atol=0)

    hidden = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    expanded = torch.empty(8, 128, device=DEVICE, dtype=DTYPE)
    infinicore.moe_expand_input_with_inv_pos_(
        ic(expanded), None, ic(hidden), ic(inv), 2
    )
    close(expanded, hidden.repeat_interleave(2, dim=0)[ref_sorted])

    gated = torch.randn(8, 256, device=DEVICE, dtype=DTYPE)
    activated = torch.empty(8, 128, device=DEVICE, dtype=DTYPE)
    infinicore.moe_silu_and_mul_quant_(ic(activated), None, ic(gated))
    expected = torch.nn.functional.silu(gated[:, :128]) * gated[:, 128:]
    close(activated, expected)

    expert_out = torch.randn(4, 2, 128, device=DEVICE, dtype=DTYPE)
    weights = torch.rand(4, 2, device=DEVICE)
    residual = torch.randn(4, 128, device=DEVICE, dtype=DTYPE)
    ref = torch.empty_like(residual)
    out = torch.empty_like(residual)
    moe_sum(expert_out, ref, weights, residual, 1.25, 0.5)
    infinicore.moe_sum_vendor_(
        ic(out), ic(expert_out), ic(weights), ic(residual), 1.25, 0.5
    )
    close(out, ref)


def test_scaled_mm():
    m, n, k = 16, 128, 128
    a = torch.randint(-8, 8, (m, k), device=DEVICE, dtype=torch.int8)
    b4 = torch.randint(-128, 127, (k, n // 2), device=DEVICE, dtype=torch.int8)
    a_scale = torch.rand(m, 1, device=DEVICE)
    b_scale = torch.rand(n, 1, device=DEVICE)
    ref = torch.empty(m, n, device=DEVICE, dtype=torch.float16)
    scaled_mm_w4a8(ref, a, b4, a_scale, b_scale)
    out = torch.empty_like(ref)
    infinicore.scaled_mm_w4a8(
        ic(a), ic(b4), ic(a_scale), ic(b_scale), trans_weight=False, out=ic(out)
    )
    close(out, ref)

    b8 = torch.randint(-8, 8, (n, k), device=DEVICE, dtype=torch.int8)
    out8 = torch.empty_like(ref)
    infinicore.scaled_mm_w8a8(
        ic(a), ic(b8), ic(a_scale), ic(b_scale), trans_weight=True, out=ic(out8)
    )
    expected8 = ((a.float() * a_scale) @ (b8.float() * b_scale).transpose(0, 1)).half()
    close(out8, expected8, rtol=5e-2, atol=1.0)


def test_group_gemm():
    experts, rows, n, k = 2, 16, 128, 128
    tokens = torch.tensor([8, 8], dtype=torch.int32)

    x16 = torch.randn(rows, k, device=DEVICE, dtype=DTYPE)
    w16 = torch.randn(experts, n, k, device=DEVICE, dtype=DTYPE)
    ref16, out16 = (
        torch.empty(rows, n, device=DEVICE, dtype=DTYPE),
        torch.empty(rows, n, device=DEVICE, dtype=DTYPE),
    )
    w16a16_group_gemm(ref16, x16, w16, tokens)
    infinicore.w16a16_group_gemm_(ic(out16), ic(x16), ic(w16), ic(tokens))
    close(out16, ref16)

    x8 = torch.randint(-8, 8, (rows, k), device=DEVICE, dtype=torch.int8)
    w8 = torch.randint(-8, 8, (experts, n, k), device=DEVICE, dtype=torch.int8)
    xs = torch.rand(rows, 1, device=DEVICE)
    ws = torch.rand(experts, n, 1, device=DEVICE)
    ref8, out8 = torch.empty_like(ref16), torch.empty_like(ref16)
    w8a8_group_gemm(ref8, x8, w8, xs, ws, tokens)
    infinicore.w8a8_group_gemm_(ic(out8), ic(x8), ic(w8), ic(xs), ic(ws), ic(tokens))
    close(out8, ref8)

    w4 = torch.randint(-128, 127, (experts, n, k // 2), device=DEVICE, dtype=torch.int8)
    ref4, out4 = torch.empty_like(ref16), torch.empty_like(ref16)
    w4a8_group_gemm(ref4, x8, w4, xs, ws, tokens)
    infinicore.w4a8_group_gemm_(ic(out4), ic(x8), ic(w4), ic(xs), ic(ws), ic(tokens))
    close(out4, ref4)


def test_vendor_graph_replay():
    kv = torch.randn(2, 512, device=DEVICE, dtype=DTYPE)
    kpe = torch.randn(2, 64, device=DEVICE, dtype=DTYPE)
    slots = torch.tensor([1, 66], device=DEVICE, dtype=torch.int64)
    scale = torch.ones(1, device=DEVICE, dtype=torch.float32)
    cache_ref = torch.zeros(2, 64, 576, device=DEVICE, dtype=DTYPE)
    cache_out = cache_ref.clone()
    concat_and_cache_mla(kv, kpe, cache_ref, slots, "auto", scale)

    torch.cuda.synchronize()
    infinicore.start_graph_recording()
    infinicore.concat_and_cache_mla(
        ic(kv), ic(kpe), ic(cache_out), ic(slots), "auto", ic(scale)
    )
    graph = infinicore.stop_graph_recording()
    graph.run()
    close(cache_out, cache_ref)


def test_paged_attention_mla_smoke():
    query = torch.randn(1, 8, 576, device=DEVICE, dtype=DTYPE)
    cache = torch.randn(1, 16, 576, device=DEVICE, dtype=DTYPE)
    blocks = torch.tensor([[0]], device=DEVICE, dtype=torch.int32)
    lens = torch.tensor([16], device=DEVICE, dtype=torch.int32)
    out = torch.empty(1, 8, 512, device=DEVICE, dtype=DTYPE)
    infinicore.paged_attention_mla_(
        ic(out), ic(query), ic(cache), 576**-0.5, ic(blocks), ic(lens), 16
    )
    infinicore.sync_device()
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    torch.manual_seed(0)
    for test in (
        test_add_rms_norm_inplace,
        test_mla_concat_and_cache,
        test_mla_int8_cache,
        test_dynamic_int8_quant,
        test_allocating_wrappers,
        test_rotary_embedding,
        test_routing_ops,
        test_moe_data_movement_and_sum,
        test_scaled_mm,
        test_group_gemm,
        test_vendor_graph_replay,
        test_paged_attention_mla_smoke,
    ):
        test()
        print(f"{test.__name__} ok")
