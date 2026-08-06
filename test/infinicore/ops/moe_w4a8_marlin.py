import argparse

import torch
from lightop import awq_marlin_repack_w4a8
from lightop import op as lightop_ops

import infinicore


def to_adjacent_w4(source, k):
    """Convert GLM block-32 nibble packing to adjacent-K AWQ packing."""
    source_u8 = source.view(torch.uint8)
    logical = torch.empty(
        (*source.shape[:-1], k), device=source.device, dtype=torch.uint8
    )
    for block in range(k // 32):
        packed = source_u8[..., block * 16 : (block + 1) * 16]
        logical[..., block * 32 : block * 32 + 16] = packed & 0x0F
        logical[..., block * 32 + 16 : (block + 1) * 32] = packed >> 4
    adjacent = (logical[..., 0::2] << 4) | logical[..., 1::2]
    return adjacent.view(torch.int8)


def ic(tensor):
    return infinicore.from_torch(tensor)


def test_repack():
    torch.manual_seed(1)
    experts, n, k = 8, 64, 64
    source = torch.randint(
        -128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8
    )
    actual_storage = torch.empty_like(source)
    infinicore.prepare_w4a8_marlin_weight_(ic(actual_storage), ic(source))
    expected = torch.empty(
        (experts, k // 32, n * 4),
        device="cuda",
        dtype=torch.int32,
    )
    awq_marlin_repack_w4a8(to_adjacent_w4(source, k), expected, experts, k, n)
    infinicore.sync_device()
    actual = actual_storage.view(torch.int32).reshape(expected.shape)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def aligned_routes(ids, experts, routing_topk):
    routes = ids.numel()
    counts = torch.empty((experts,), device="cuda", dtype=torch.int32)
    sorted_ids = torch.empty((routes,), device="cuda", dtype=torch.int32)
    inverse = torch.empty_like(sorted_ids)
    infinicore.moe_argsort_bincount_with_inv_pos_(
        ic(counts), ic(sorted_ids), ic(inverse), ic(ids), experts
    )
    padded = torch.empty((routes * 16,), device="cuda", dtype=torch.int32)
    expert_ids = torch.empty((routes,), device="cuda", dtype=torch.int32)
    post_pad = torch.empty((1,), device="cuda", dtype=torch.int32)
    infinicore.moe_align_block_size_from_counts_(
        ic(padded),
        ic(expert_ids),
        ic(post_pad),
        ic(sorted_ids),
        ic(counts),
        16,
        routing_topk,
    )
    infinicore.sync_device()
    return counts, sorted_ids, padded, expert_ids, post_pad


def test_align():
    torch.manual_seed(2)
    ids = torch.randint(0, 32, (4, 8), device="cuda", dtype=torch.int32)
    counts, sorted_ids, padded, expert_ids, post_pad = aligned_routes(ids, 32, 8)
    counts_cpu = counts.cpu().tolist()
    sorted_cpu = sorted_ids.cpu().tolist()
    expected_padded = []
    expected_experts = []
    source_offset = 0
    for expert, count in enumerate(counts_cpu):
        expected_padded.extend(sorted_cpu[source_offset : source_offset + count])
        source_offset += count
        padded_count = ((count + 15) // 16) * 16
        expected_padded.extend([ids.numel()] * (padded_count - count))
        expected_experts.extend([expert] * (padded_count // 16))
    used = post_pad.item()
    assert used == len(expected_padded)
    torch.testing.assert_close(
        padded[:used].cpu(), torch.tensor(expected_padded, dtype=torch.int32)
    )
    torch.testing.assert_close(
        expert_ids[: len(expected_experts)].cpu(),
        torch.tensor(expected_experts, dtype=torch.int32),
    )


def run_gemm_case(input_rows, topk, routing_topk, n, k):
    experts = 256
    route_tokens = input_rows * topk // routing_topk
    ids = torch.arange(routing_topk, device="cuda", dtype=torch.int32).repeat(
        route_tokens, 1
    )
    _, _, padded, expert_ids, post_pad = aligned_routes(ids, experts, routing_topk)
    source_weight = torch.randint(
        -128, 128, (experts, n, k // 2), device="cuda", dtype=torch.int8
    )
    marlin_storage = torch.empty_like(source_weight)
    infinicore.prepare_w4a8_marlin_weight_(ic(marlin_storage), ic(source_weight))
    marlin = marlin_storage.view(torch.int32).reshape(experts, k // 32, n * 4)
    input = torch.randint(-127, 128, (input_rows, k), device="cuda", dtype=torch.int8)
    input_scale = torch.rand((input_rows, 1), device="cuda", dtype=torch.float32)
    weight_scale = torch.rand((experts, n, 1), device="cuda", dtype=torch.float32)
    actual = torch.empty((input_rows * topk, n), device="cuda", dtype=torch.bfloat16)
    topk_weights = None
    if topk == 1 and routing_topk > 1:
        topk_weights = torch.rand(
            (route_tokens, routing_topk), device="cuda", dtype=torch.float32
        )
    infinicore.moe_w4a8_marlin_(
        ic(actual),
        ic(input),
        ic(marlin_storage),
        ic(input_scale),
        ic(weight_scale),
        None if topk_weights is None else ic(topk_weights),
        ic(padded),
        ic(expert_ids),
        ic(post_pad),
        topk,
        routing_topk,
    )
    expected = torch.empty_like(actual)
    lightop_ops.moe_w4a8_marlin_asm(
        input,
        marlin,
        expected,
        input_scale,
        weight_scale,
        topk_weights,
        padded,
        expert_ids,
        post_pad,
        topk,
        1000,
    )
    infinicore.sync_device()
    assert torch.isfinite(actual).all()
    assert torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    del source_weight, marlin_storage, marlin, weight_scale, actual, expected


def run_semantic_case(n, k, topk, routing_topk, topk_weights):
    experts = 256
    route_tokens = 1
    ids = torch.arange(routing_topk, device="cuda", dtype=torch.int32).repeat(
        route_tokens, 1
    )
    _, _, padded, expert_ids, post_pad = aligned_routes(ids, experts, routing_topk)
    source_weight = torch.full(
        (experts, n, k // 2), 0x11, device="cuda", dtype=torch.int8
    )
    marlin_storage = torch.empty_like(source_weight)
    infinicore.prepare_w4a8_marlin_weight_(ic(marlin_storage), ic(source_weight))
    input_rows = route_tokens * routing_topk if topk == 1 else route_tokens
    input = torch.ones((input_rows, k), device="cuda", dtype=torch.int8)
    input_scale = torch.ones((input_rows, 1), device="cuda", dtype=torch.float32)
    # LightOp's signed-int4 Marlin kernel applies an internal factor of 16.
    # GLM checkpoint scales require a factor of 18, so convert once by 18/16.
    weight_scale = torch.full(
        (experts, n, 1), 18.0 / 16.0, device="cuda", dtype=torch.float32
    )
    actual = torch.empty((input_rows * topk, n), device="cuda", dtype=torch.bfloat16)
    infinicore.moe_w4a8_marlin_(
        ic(actual),
        ic(input),
        ic(marlin_storage),
        ic(input_scale),
        ic(weight_scale),
        None if topk_weights is None else ic(topk_weights),
        ic(padded),
        ic(expert_ids),
        ic(post_pad),
        topk,
        routing_topk,
    )
    infinicore.sync_device()
    expected = torch.full_like(actual, float(k * 18))
    if topk_weights is not None:
        expected *= topk_weights.reshape(-1, 1).to(torch.bfloat16)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=16)


def test_semantic_gemm():
    run_semantic_case(512, 6144, 8, 8, None)
    topk_weights = (
        torch.arange(1, 9, device="cuda", dtype=torch.float32).reshape(1, 8) / 10
    )
    run_semantic_case(6144, 256, 1, 8, topk_weights)


def test_gemm():
    torch.manual_seed(3)
    run_gemm_case(1, 8, 8, 512, 6144)
    run_gemm_case(8, 1, 8, 6144, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test_repack()
    test_align()
    test_gemm()
    test_semantic_gemm()
    print("moe_w4a8_marlin ok")
