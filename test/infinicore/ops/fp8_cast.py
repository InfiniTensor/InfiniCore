"""Check FP8 weights and boolean acceptance masks across the ATen cast bridge."""

import torch
from infinicore.lib import _infinicore

import infinicore


def test_fp8_cast():
    bits = torch.arange(256, dtype=torch.int16)
    bits = bits[(bits != 127) & (bits != 255)].to(torch.uint8)
    source = bits.view(torch.float8_e4m3fn).cuda()
    for dtype in (torch.float32, torch.bfloat16):
        output = torch.empty(source.shape, dtype=dtype, device="cuda")
        _infinicore.cast_(
            infinicore.from_torch(output)._underlying,
            infinicore.from_torch(source)._underlying,
        )
        infinicore.sync_device()
        torch.testing.assert_close(output, source.to(dtype), rtol=0, atol=0)


def test_bool_cast():
    candidates = torch.tensor([3, 5, 7, 9], device="cuda")
    expected = torch.tensor([4, 5, 7, 8], device="cuda")
    source = infinicore.equal(
        infinicore.from_torch(candidates), infinicore.from_torch(expected)
    )
    for dtype in (torch.float32, torch.int64):
        output = torch.empty(source.shape, dtype=dtype, device="cuda")
        _infinicore.cast_(
            infinicore.from_torch(output)._underlying,
            source._underlying,
        )
        infinicore.sync_device()
        torch.testing.assert_close(
            output, (candidates == expected).to(dtype), rtol=0, atol=0
        )


if __name__ == "__main__":
    test_fp8_cast()
    test_bool_cast()
    print("Finite E4M3 and boolean acceptance mask casts passed")
