import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test(dtype, index_dtype):
    torch.manual_seed(0)
    tokens = 4
    kv_dim = 512
    rope_dim = 64
    cache = torch.zeros((3, 64, kv_dim + rope_dim), device="cuda", dtype=dtype)
    expected = cache.clone()
    kv_c = torch.randn((tokens, kv_dim), device="cuda", dtype=dtype)
    k_pe = torch.randn((tokens, rope_dim), device="cuda", dtype=dtype)
    slots = torch.tensor([0, 65, 127, 130], device="cuda", dtype=index_dtype)
    scale = torch.ones(1, device="cuda", dtype=torch.float32)

    expected.view(-1, kv_dim + rope_dim)[slots.long()] = torch.cat((kv_c, k_pe), dim=-1)
    infinicore.concat_and_cache_mla(
        ic(kv_c), ic(k_pe), ic(cache), ic(slots), "auto", ic(scale)
    )
    infinicore.sync_device()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    for dtype in (torch.float16, torch.bfloat16):
        for index_dtype in (torch.int32, torch.int64):
            test(dtype, index_dtype)
    print("concat_and_cache_mla auto cache ok")
