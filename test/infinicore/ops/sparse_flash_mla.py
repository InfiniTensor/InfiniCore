import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def reference(query, cache, indices, lens, scale, value_dim):
    outputs = []
    for token in range(query.shape[0]):
        valid_indices = indices[token, 0, : lens[token]].long()
        valid_indices = valid_indices[valid_indices >= 0]
        keys = cache[valid_indices, 0]
        scores = torch.einsum("hd,kd->hk", query[token].float(), keys.float()) * scale
        weights = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("hk,kd->hd", weights, keys[:, :value_dim].float()))
    return torch.stack(outputs).to(query.dtype)


def run_case(dtype, tokens, heads, query_dim, value_dim, topk):
    cache_slots = max(topk + 3, 16)
    query = torch.randn((tokens, heads, query_dim), device="cuda", dtype=dtype)
    cache = torch.randn((cache_slots, 1, query_dim), device="cuda", dtype=dtype)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(tokens, 1, 1)
    if topk > 2:
        indices[-1, 0, -1] = -1
    lens = torch.full((tokens,), topk, device="cuda", dtype=torch.int32)
    scale = query_dim**-0.5
    expected = reference(query, cache, indices, lens, scale, value_dim)
    actual = torch.empty((tokens, heads, value_dim), device="cuda", dtype=dtype)
    infinicore.sparse_flash_mla_(
        ic(actual), ic(query), ic(cache), ic(indices), ic(lens), scale
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        run_case(dtype, 2, 4, 16, 12, 5)
    run_case(torch.bfloat16, 1, 64, 576, 512, 64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("sparse_flash_mla ok")
