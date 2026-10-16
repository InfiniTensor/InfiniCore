import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test():
    torch.manual_seed(0)
    tokens, topk, hidden = 7, 8, 6144
    input = torch.randn((tokens, topk, hidden), device="cuda", dtype=torch.bfloat16)
    weights = torch.rand((tokens, topk), device="cuda", dtype=torch.float32)
    residual = torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    output = torch.empty_like(residual)
    infinicore.moe_sum_vendor_(
        ic(output), ic(input), ic(weights), ic(residual), 1.25, 0.5
    )
    infinicore.sync_device()
    expected = (
        (input.float() * weights.unsqueeze(-1)).sum(dim=1) * 1.25
        + residual.float() * 0.5
    ).to(torch.bfloat16)
    torch.testing.assert_close(output, expected, rtol=2e-2, atol=3e-2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("moe_sum_vendor ok")
