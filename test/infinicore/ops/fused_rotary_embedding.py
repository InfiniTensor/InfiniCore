import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def reference(x, positions, cache, is_neox):
    head_size = x.shape[-1]
    half = head_size // 2
    cosine = cache[positions, :half].float()
    sine = cache[positions, half:].float()
    if is_neox:
        first = x[..., :half].float()
        second = x[..., half:].float()
        return torch.cat(
            (
                first * cosine[:, None, :] - second * sine[:, None, :],
                second * cosine[:, None, :] + first * sine[:, None, :],
            ),
            dim=-1,
        ).to(x.dtype)

    even = x[..., 0::2].float()
    odd = x[..., 1::2].float()
    out = torch.empty_like(x)
    out[..., 0::2] = (even * cosine[:, None, :] - odd * sine[:, None, :]).to(x.dtype)
    out[..., 1::2] = (odd * cosine[:, None, :] + even * sine[:, None, :]).to(x.dtype)
    return out


def run_case(dtype, is_neox):
    positions = torch.tensor([0, 3, 7], device="cuda", dtype=torch.int64)
    query = torch.randn((3, 8, 64), device="cuda", dtype=dtype)
    key = torch.randn((3, 2, 64), device="cuda", dtype=dtype)
    cache = torch.randn((16, 64), device="cuda", dtype=dtype)
    expected_query = reference(query, positions, cache, is_neox)
    expected_key = reference(key, positions, cache, is_neox)
    actual_query = query.clone()
    actual_key = key.clone()
    infinicore.fused_rotary_embedding_(
        ic(actual_query),
        ic(actual_key),
        ic(positions),
        64,
        ic(cache),
        is_neox,
    )
    infinicore.sync_device()
    torch.testing.assert_close(actual_query, expected_query, rtol=2e-2, atol=3e-2)
    torch.testing.assert_close(actual_key, expected_key, rtol=2e-2, atol=3e-2)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for is_neox in (False, True):
            run_case(dtype, is_neox)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("fused_rotary_embedding ok")
