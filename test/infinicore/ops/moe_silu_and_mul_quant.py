import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test():
    torch.manual_seed(0)
    rows, width = 9, 256
    input = torch.randn((rows, width * 2), device="cuda", dtype=torch.bfloat16)
    expected = (
        torch.nn.functional.silu(input[:, :width].float()) * input[:, width:].float()
    )
    expected = expected.to(torch.bfloat16).float()
    expected_scale = expected.abs().amax(dim=-1, keepdim=True) / 127.0
    expected_quant = torch.clamp(torch.round(expected / expected_scale), -127, 127).to(
        torch.int8
    )

    output = torch.empty((rows, width), device="cuda", dtype=torch.int8)
    scale = torch.empty((rows, 1), device="cuda", dtype=torch.float32)
    infinicore.moe_silu_and_mul_quant_(ic(output), ic(scale), ic(input), 1)
    infinicore.sync_device()
    torch.testing.assert_close(scale, expected_scale, rtol=2e-3, atol=1e-6)
    torch.testing.assert_close(output, expected_quant, rtol=0, atol=1)

    dense_input = input[:, :width].contiguous()
    dense_output = torch.empty_like(output)
    dense_scale = torch.empty_like(scale)
    infinicore.dynamic_scaled_int8_quant(
        ic(dense_input), ic(dense_scale), out=ic(dense_output)
    )
    infinicore.sync_device()
    dense_expected_scale = dense_input.float().abs().amax(dim=-1, keepdim=True) / 127.0
    torch.testing.assert_close(dense_scale, dense_expected_scale, rtol=2e-3, atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("moe_silu_and_mul_quant ok")
