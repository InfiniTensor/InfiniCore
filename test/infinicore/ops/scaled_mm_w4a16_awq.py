import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def prepare_awq(weight, channel_scales):
    k, n = weight.shape
    assert k % 32 == 0
    checkpoint = weight.transpose(0, 1).contiguous()
    checkpoint_nibbles = checkpoint.to(torch.int16) & 0xF
    blocks = checkpoint_nibbles.reshape(n, -1, 32)
    checkpoint_packed = (
        (blocks[..., :16] | (blocks[..., 16:] << 4))
        .reshape(n, -1)
        .to(torch.int8)
        .contiguous()
    )
    qweight = torch.empty((k, n // 2), dtype=torch.int8, device=weight.device)
    qzeros = torch.empty((k // 64, n // 2), dtype=torch.int8, device=weight.device)
    scales = torch.empty((k // 64, n), dtype=torch.bfloat16, device=weight.device)
    infinicore.prepare_glm_w4a16_awq_(
        ic(qweight),
        ic(qzeros),
        ic(scales),
        ic(checkpoint_packed),
        ic(channel_scales.reshape(n, 1).contiguous()),
    )
    return qweight, qzeros, scales


def run_case(m, n, k, with_bias):
    input = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    channel_scales = torch.rand((n,), device="cuda", dtype=torch.float32) + 0.01
    qweight, qzeros, scales = prepare_awq(weight, channel_scales)
    unsigned_weight = weight.to(torch.int16) + 8
    expected_qweight = (unsigned_weight[:, 0::2] | (unsigned_weight[:, 1::2] << 4)).to(
        torch.int8
    )
    expected_qzeros = torch.full_like(qzeros, -120)
    expected_scales = (
        (channel_scales * 18.0).to(torch.bfloat16).reshape(1, n).expand(k // 64, n)
    )
    torch.testing.assert_close(qweight, expected_qweight, rtol=0, atol=0)
    torch.testing.assert_close(qzeros, expected_qzeros, rtol=0, atol=0)
    torch.testing.assert_close(scales, expected_scales, rtol=0, atol=0)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16) if with_bias else None
    prepared_scales = (channel_scales * 18.0).to(torch.bfloat16).float()
    expected = torch.matmul(
        input.float(),
        weight.float() * prepared_scales.reshape(1, n),
    )
    if bias is not None:
        expected += bias.float()
    expected = expected.to(torch.bfloat16)

    actual = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    infinicore.scaled_mm_w4a16_awq(
        ic(input),
        ic(qweight),
        ic(qzeros),
        ic(scales),
        None if bias is None else ic(bias),
        out=ic(actual),
    )
    infinicore.sync_device()
    error = actual.float() - expected.float()
    relative_l2 = error.norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1), expected.float().reshape(1, -1)
    )
    assert relative_l2.item() < 1e-2
    assert cosine.item() > 0.9999


def run_graph_replay_case():
    m, n, k = 1, 512, 2048
    device = infinicore.device("cuda", 0)
    input = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    weight = torch.randint(-8, 8, (k, n), device="cuda", dtype=torch.int8)
    channel_scales = torch.rand((n,), device="cuda", dtype=torch.float32) + 0.01
    qweight, qzeros, scales = prepare_awq(weight, channel_scales)
    actual = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)

    infinicore.start_graph_recording(device)
    infinicore.scaled_mm_w4a16_awq(
        ic(input), ic(qweight), ic(qzeros), ic(scales), None, out=ic(actual)
    )
    graph = infinicore.stop_graph_recording()

    replacement = torch.randn_like(input)
    ic(input).copy_(ic(replacement))
    infinicore.sync_stream()
    graph.run()
    infinicore.sync_stream()

    prepared_scales = (channel_scales * 18.0).to(torch.bfloat16).float()
    expected = torch.matmul(
        replacement.float(),
        weight.float() * prepared_scales.reshape(1, n),
    ).to(torch.bfloat16)
    error = actual.float() - expected.float()
    relative_l2 = error.norm() / expected.float().norm()
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1), expected.float().reshape(1, -1)
    )
    assert relative_l2.item() < 1e-2
    assert cosine.item() > 0.9999


def test():
    torch.manual_seed(0)
    run_case(1, 512, 2048, False)
    run_case(1, 4096, 2048, False)
    run_case(32, 512, 2048, False)
    run_graph_replay_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("scaled_mm_w4a16_awq ok")
