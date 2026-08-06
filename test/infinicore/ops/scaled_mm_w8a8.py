import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def reference(a, weight, a_scales, b_scales, bias, trans_weight, dtype):
    rhs = weight.float().transpose(0, 1) if trans_weight else weight.float()
    result = torch.matmul(a.float(), rhs)
    result *= a_scales
    result *= b_scales.transpose(0, 1)
    if bias is not None:
        result += bias.float()
    return result.to(dtype)


def run_case(m, n, k, dtype, trans_weight, with_bias):
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    weight_shape = (n, k) if trans_weight else (k, n)
    weight = torch.randint(-128, 128, weight_shape, device="cuda", dtype=torch.int8)
    a_scales = torch.rand((m, 1), device="cuda", dtype=torch.float32) * 0.002
    b_scales = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.002
    bias = torch.randn((n,), device="cuda", dtype=dtype) * 0.1 if with_bias else None
    expected = reference(a, weight, a_scales, b_scales, bias, trans_weight, dtype)
    actual = torch.empty((m, n), device="cuda", dtype=dtype)
    infinicore.scaled_mm_w8a8(
        ic(a),
        ic(weight),
        ic(a_scales),
        ic(b_scales),
        None if bias is None else ic(bias),
        trans_weight,
        out=ic(actual),
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def run_graph_replay_case():
    m, n, k = 1, 512, 2048
    device = infinicore.device("cuda", 0)
    a = torch.randint(-128, 128, (m, k), device="cuda", dtype=torch.int8)
    weight = torch.randint(-128, 128, (k, n), device="cuda", dtype=torch.int8)
    a_scales = torch.rand((m, 1), device="cuda", dtype=torch.float32) * 0.002
    b_scales = torch.rand((n, 1), device="cuda", dtype=torch.float32) * 0.002
    actual = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    infinicore.start_graph_recording(device)
    infinicore.scaled_mm_w8a8(
        ic(a), ic(weight), ic(a_scales), ic(b_scales), None, False, out=ic(actual)
    )
    graph = infinicore.stop_graph_recording()

    replacement = torch.randint_like(a, -128, 128)
    ic(a).copy_(ic(replacement))
    infinicore.sync_stream()
    graph.run()
    infinicore.sync_stream()

    expected = reference(
        replacement, weight, a_scales, b_scales, None, False, torch.bfloat16
    )
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for trans_weight in (False, True):
            for with_bias in (False, True):
                run_case(3, 66, 128, dtype, trans_weight, with_bias)

    run_case(2, 256, 6144, torch.bfloat16, True, False)
    run_graph_replay_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hygon", action="store_true", help="run on the Hygon CUDA-compatible device"
    )
    parser.parse_args()
    test()
    print("scaled_mm_w8a8 ok")
