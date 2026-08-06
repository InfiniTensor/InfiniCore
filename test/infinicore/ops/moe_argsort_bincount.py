import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def run_case(tokens, topk, experts):
    torch.manual_seed(tokens + topk + experts)
    ids = torch.randint(0, experts, (tokens, topk), device="cuda", dtype=torch.int32)
    counts = torch.empty((experts,), device="cuda", dtype=torch.int32)
    sorted_indices = torch.empty((tokens * topk,), device="cuda", dtype=torch.int32)
    inv_pos = torch.empty_like(sorted_indices)
    infinicore.moe_argsort_bincount_with_inv_pos_(
        ic(counts), ic(sorted_indices), ic(inv_pos), ic(ids), experts
    )
    infinicore.sync_device()

    flat_ids = ids.flatten()
    torch.testing.assert_close(
        counts, torch.bincount(flat_ids, minlength=experts).to(torch.int32)
    )
    positions = torch.arange(tokens * topk, device="cuda", dtype=torch.int32)
    torch.testing.assert_close(inv_pos[sorted_indices.long()], positions)
    grouped_experts = flat_ids[sorted_indices.long()]
    assert torch.all(grouped_experts[1:] >= grouped_experts[:-1])

    hidden = torch.randn((tokens, 128), device="cuda", dtype=torch.bfloat16)
    expanded = torch.empty((tokens * topk, 128), device="cuda", dtype=torch.int8)
    scales = torch.empty((tokens * topk, 1), device="cuda", dtype=torch.float32)
    infinicore.moe_expand_input_with_inv_pos_(
        ic(expanded), ic(scales), ic(hidden), ic(inv_pos), topk, 128, 1
    )
    infinicore.sync_device()
    expected = hidden.repeat_interleave(topk, dim=0)[sorted_indices.long()]
    expected_scale = expected.float().abs().amax(dim=-1, keepdim=True) / 127.0
    # expected_quant = torch.clamp(
    #     torch.round(expected.float() / expected_scale), -127, 127
    # ).to(torch.int8)
    torch.testing.assert_close(scales, expected_scale, rtol=2e-3, atol=1e-6)


def test():
    run_case(4, 8, 256)
    run_case(1024, 8, 256)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("moe_argsort_bincount ok")
