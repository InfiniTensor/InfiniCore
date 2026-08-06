import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def pack_int4(weight, trans_weight):
    """Pack signed int4 values, with the even element in the low nibble."""
    values = weight.to(torch.int16) & 0xF
    if trans_weight:
        packed = values[:, 0::2] | (values[:, 1::2] << 4)
    else:
        packed = values[:, 0::2] | (values[:, 1::2] << 4)
    return packed.to(torch.int8).contiguous()


def reference(a, weight, a_scales, b_scales, bias, trans_weight, dtype):
    rhs = weight.float().transpose(0, 1) if trans_weight else weight.float()
    result = torch.matmul(a.float(), rhs)
    result *= a_scales
    result *= b_scales.transpose(0, 1)
    if bias is not None:
        result += bias.float()
    return result.to(dtype)


def run_case(m, n, k, dtype, trans_weight, with_bias):
    a = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    weight_shape = (n, k) if trans_weight else (k, n)
    weight = torch.randint(-8, 8, weight_shape, device="cuda", dtype=torch.int8)
    packed = pack_int4(weight, trans_weight)
    a_scales = torch.rand((m, 1), device="cuda", dtype=torch.float32) * 0.02
    b_scales = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.02
    bias = torch.randn((n,), device="cuda", dtype=dtype) * 0.1 if with_bias else None
    expected = reference(a, weight, a_scales, b_scales, bias, trans_weight, dtype)
    actual = torch.empty((m, n), device="cuda", dtype=dtype)
    infinicore.scaled_mm_w4a8(
        ic(a),
        ic(packed),
        ic(a_scales),
        ic(b_scales),
        None if bias is None else ic(bias),
        trans_weight,
        out=ic(actual),
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for trans_weight in (False, True):
            for with_bias in (False, True):
                run_case(3, 66, 128, dtype, trans_weight, with_bias)

    # GLM-5.2 uses weights packed along K. Keep a real K-sized smoke case
    # to catch indexing overflow or launch-geometry regressions.
    run_case(2, 256, 6144, torch.bfloat16, True, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hygon", action="store_true", help="run on the Hygon CUDA-compatible device"
    )
    parser.parse_args()
    test()
    print("scaled_mm_w4a8 ok")
