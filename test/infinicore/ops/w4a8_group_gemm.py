import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def pack_int4(weight):
    assert weight.shape[-1] % 32 == 0
    values = weight.to(torch.int16) & 0xF
    blocks = values.reshape(*values.shape[:-1], -1, 32)
    packed = blocks[..., :16] | (blocks[..., 16:] << 4)
    return packed.reshape(*values.shape[:-1], -1).to(torch.int8).contiguous()


def reference(
    input,
    weight,
    input_scale,
    weight_scale,
    tokens_per_experts,
    sorted_token_ids,
    bias,
    trans_weight,
    dtype,
):
    chunks = []
    row = 0
    for expert, count in enumerate(tokens_per_experts.cpu().tolist()):
        if count == 0:
            continue
        rhs = (
            weight[expert].float().transpose(0, 1)
            if trans_weight
            else weight[expert].float()
        ) * 18.0
        value = torch.matmul(input[row : row + count].float(), rhs)
        value *= input_scale[row : row + count]
        value *= weight_scale[expert].transpose(0, 1)
        if bias is not None:
            value += bias[expert].float()
        chunks.append(value)
        row += count
    grouped = torch.cat(chunks, dim=0).to(dtype)
    if sorted_token_ids is None:
        return grouped
    result = torch.empty_like(grouped)
    result[sorted_token_ids.long()] = grouped
    return result


def run_case(m, n, k, dtype, trans_weight, with_sorted, with_bias):
    counts = torch.tensor([2, 0, m - 3, 1], device="cuda", dtype=torch.int32)
    experts = counts.numel()
    input = torch.randint(-8, 8, (m, k), device="cuda", dtype=torch.int8)
    weight_shape = (experts, n, k) if trans_weight else (experts, k, n)
    weight = torch.randint(-8, 8, weight_shape, device="cuda", dtype=torch.int8)
    packed = pack_int4(weight)
    input_scale = torch.rand((m, 1), device="cuda") * 0.02
    weight_scale = torch.rand((experts, n, 1), device="cuda") * 0.02
    sorted_ids = (
        torch.randperm(m, device="cuda", dtype=torch.int64).to(torch.int32)
        if with_sorted
        else None
    )
    bias = (
        torch.randn((experts, n), device="cuda", dtype=dtype) * 0.1
        if with_bias
        else None
    )
    expected = reference(
        input,
        weight,
        input_scale,
        weight_scale,
        counts,
        sorted_ids,
        bias,
        trans_weight,
        dtype,
    )
    actual = torch.empty((m, n), device="cuda", dtype=dtype)
    infinicore.w4a8_group_gemm_(
        ic(actual),
        ic(input),
        ic(packed),
        ic(input_scale),
        ic(weight_scale),
        ic(counts),
        None if sorted_ids is None else ic(sorted_ids),
        None if bias is None else ic(bias),
        trans_weight,
        True,
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=3e-2)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for trans_weight in (False, True):
            for with_sorted in (False, True):
                run_case(6, 96, 128, dtype, trans_weight, with_sorted, True)

    run_case(6, 256, 6144, torch.bfloat16, True, True, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("w4a8_group_gemm ok")
