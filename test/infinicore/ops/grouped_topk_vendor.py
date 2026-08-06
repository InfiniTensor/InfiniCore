import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test():
    torch.manual_seed(0)
    tokens, experts, topk = 7, 256, 8
    scores = torch.randn((tokens, experts), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((experts,), device="cuda", dtype=torch.float32) * 0.1
    routed_scale = 2.5

    sigmoid = torch.sigmoid(scores.float())
    selected = torch.topk(sigmoid + bias, topk, dim=-1, sorted=True).indices
    expected = torch.gather(sigmoid, 1, selected)
    expected = expected / expected.sum(dim=-1, keepdim=True) * routed_scale

    actual_weights = torch.empty((tokens, topk), device="cuda", dtype=torch.float32)
    actual_ids = torch.empty((tokens, topk), device="cuda", dtype=torch.int32)
    infinicore.grouped_topk_vendor(
        ic(scores),
        1,
        1,
        topk,
        True,
        routed_scale,
        ic(bias),
        "sigmoid",
        out=(ic(actual_weights), ic(actual_ids)),
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual_ids, selected.to(torch.int32), rtol=0, atol=0)
    torch.testing.assert_close(actual_weights, expected, rtol=2e-4, atol=2e-5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("grouped_topk_vendor ok")
