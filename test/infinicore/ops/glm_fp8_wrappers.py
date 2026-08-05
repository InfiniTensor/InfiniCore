import math

import torch
from infinicore.lib import _infinicore

import infinicore


def wrap(tensor):
    return infinicore.from_torch(tensor)


def sync():
    infinicore.sync_device()


def test_indexer_quant():
    torch.manual_seed(20)
    q = (torch.rand(2, 3, 128, device="cuda") * 8 - 4).to(torch.bfloat16)
    weights = (torch.rand(2, 3, device="cuda") * 2 - 1).to(torch.bfloat16)
    q_fp8 = torch.zeros_like(q, dtype=torch.float8_e4m3fn)
    weights_fp32 = torch.zeros_like(weights, dtype=torch.float32)
    tensors = [wrap(x) for x in (q_fp8, weights_fp32, q, weights)]
    _infinicore.fp8_indexer_quant_(*(x._underlying for x in tensors))
    sync()

    scales = torch.exp2(
        torch.ceil(torch.log2(q.float().abs().amax(dim=-1).clamp_min(1.0e-4) / 448.0))
    )
    expected_q = (
        (q.float() / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    )
    torch.testing.assert_close(q_fp8.float(), expected_q.float(), rtol=0, atol=0)
    torch.testing.assert_close(
        weights_fp32, weights.float() * scales, rtol=1e-6, atol=1e-7
    )


def test_fused_indexer():
    torch.manual_seed(21)
    tokens, heads, head_dim, rope_dim = 2, 3, 128, 64
    q_raw = torch.randn(tokens, heads, head_dim, device="cuda", dtype=torch.bfloat16)
    k_weights = torch.randn(
        tokens, head_dim + heads, device="cuda", dtype=torch.bfloat16
    )
    norm_weight = torch.ones(head_dim, device="cuda", dtype=torch.bfloat16)
    norm_bias = torch.zeros(head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    cos_sin = torch.zeros(2, rope_dim, device="cuda", dtype=torch.bfloat16)
    cos_sin[:, : rope_dim // 2] = 1
    slots = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    q_fp8 = torch.zeros_like(q_raw, dtype=torch.float8_e4m3fn)
    weights_fp32 = torch.zeros(tokens, heads, device="cuda", dtype=torch.float32)
    k_cache = torch.zeros(1, 64, 132, device="cuda", dtype=torch.uint8)
    tensors = [
        wrap(x)
        for x in (
            q_fp8,
            weights_fp32,
            k_cache,
            q_raw,
            k_weights,
            norm_weight,
            norm_bias,
            positions,
            cos_sin,
            slots,
        )
    ]
    _infinicore.fused_fp8_indexer_(
        *(x._underlying for x in tensors), rope_dim, 1.0e-5, 1.0
    )
    sync()
    assert torch.isfinite(q_fp8.float()).all()
    assert torch.isfinite(weights_fp32).all()
    assert torch.count_nonzero(q_fp8.float()) > 0
    assert torch.count_nonzero(k_cache[:, :2]) > 0


def test_mla_cache_and_sparse():
    torch.manual_seed(22)
    tokens = 70
    compressed = torch.randn(tokens, 512, device="cuda", dtype=torch.bfloat16)
    norm_weight = torch.randn(512, device="cuda", dtype=torch.bfloat16)
    rope = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    cache = torch.zeros(2, 64, 656, device="cuda", dtype=torch.uint8)
    vendor = torch.zeros(2, 64, 576, device="cuda", dtype=torch.bfloat16)
    tensors = [wrap(x) for x in (cache, vendor, compressed, norm_weight, rope, slots)]
    _infinicore.fp8_mla_rmsnorm_dual_cache_(*(x._underlying for x in tensors), 1.0e-5)
    sync()

    entries = cache.view(-1, 656)[:tokens]
    latent_fp8 = entries[:, :512].contiguous().view(torch.float8_e4m3fn)
    scales = entries[:, 512:528].contiguous().view(torch.float32)
    latent = (
        latent_fp8.float().view(tokens, 4, 128) * scales.view(tokens, 4, 1)
    ).reshape(tokens, 512)
    cached_rope = entries[:, 528:].contiguous().view(torch.bfloat16)
    torch.testing.assert_close(
        vendor.view(-1, 576)[:tokens, :512],
        latent.to(torch.bfloat16),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        vendor.view(-1, 576)[:tokens, 512:], cached_rope, rtol=0, atol=0
    )

    query = torch.randn(1, 2, 576, device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(tokens, device="cuda", dtype=torch.int32).view(1, 1, tokens)
    lens = torch.tensor([tokens], device="cuda", dtype=torch.int32)
    output = torch.zeros(1, 2, 512, device="cuda", dtype=torch.bfloat16)
    sparse_tensors = [
        wrap(x) for x in (output, query, cache.view(-1, 1, 656), indices, lens)
    ]
    scale = float(576**-0.5)
    _infinicore.fp8_sparse_mla_(*(x._underlying for x in sparse_tensors), scale)
    sync()

    keys = torch.cat([latent, cached_rope.float()], dim=-1)
    expected = []
    for head in range(query.shape[1]):
        logits = keys @ query[0, head].float() * scale
        expected.append((torch.softmax(logits, dim=0)[:, None] * latent).sum(0))
    expected = torch.stack(expected).unsqueeze(0)
    torch.testing.assert_close(output.float(), expected, rtol=0.02, atol=0.03)


def test_indexer_logits():
    torch.manual_seed(23)
    tokens, heads, blocks = 2, 3, 4
    q = (torch.rand(tokens, heads, 128, device="cuda") * 8 - 4).to(torch.float8_e4m3fn)
    keys = (torch.rand(blocks, 64, 128, device="cuda") * 8 - 4).to(torch.float8_e4m3fn)
    scales = torch.rand(blocks, 64, device="cuda", dtype=torch.float32) * 0.02
    cache = torch.zeros(blocks, 64, 132, device="cuda", dtype=torch.uint8)
    raw = cache.view(blocks, -1)
    raw[:, : 64 * 128] = keys.view(torch.uint8).reshape(blocks, -1)
    raw[:, 64 * 128 :] = scales.view(torch.uint8).reshape(blocks, -1)
    block_tables = torch.arange(blocks, device="cuda", dtype=torch.int32).view(2, 2)
    weights = torch.rand(tokens, heads, device="cuda", dtype=torch.float32)
    positions = torch.tensor([70, 45], device="cuda", dtype=torch.int64)
    request_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    logits = torch.zeros(tokens, 128, device="cuda", dtype=torch.float32)
    tensors = [
        wrap(x)
        for x in (
            logits,
            q,
            cache,
            block_tables,
            weights,
            positions,
            request_ids,
        )
    ]
    _infinicore.fp8_indexer_logits_(*(x._underlying for x in tensors))
    sync()

    expected = torch.full_like(logits, -math.inf)
    for token in range(tokens):
        request = int(request_ids[token])
        for pos in range(int(positions[token]) + 1):
            block = int(block_tables[request, pos // 64])
            offset = pos % 64
            dots = (q[token].float() * keys[block, offset].float()[None, :]).sum(-1)
            expected[token, pos] = (
                torch.relu(dots * scales[block, offset]) * weights[token]
            ).sum()
    assert torch.equal(torch.isneginf(logits), torch.isneginf(expected))
    finite = torch.isfinite(expected)
    torch.testing.assert_close(
        logits[finite], expected[finite], rtol=2.0e-4, atol=2.0e-3
    )


def test_select_last_token_hidden():
    torch.manual_seed(24)
    hidden = torch.randn(1, 9, 16, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 2, 5, 9], device="cuda", dtype=torch.int32)
    output = torch.zeros(1, 3, 16, device="cuda", dtype=torch.bfloat16)
    tensors = [wrap(x) for x in (output, hidden, offsets)]
    _infinicore.select_last_token_hidden_(*(x._underlying for x in tensors))
    sync()
    torch.testing.assert_close(output, hidden[:, [1, 4, 8]], rtol=0, atol=0)


if __name__ == "__main__":
    test_indexer_quant()
    test_fused_indexer()
    test_mla_cache_and_sparse()
    test_indexer_logits()
    test_select_last_token_hidden()
    print("GLM FP8 InfiniCore wrapper tests passed")
