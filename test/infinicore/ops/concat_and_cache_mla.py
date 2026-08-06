import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def run_case(dtype, index_dtype):
    kv = torch.randn((3, 512), device="cuda", dtype=dtype)
    rope = torch.randn((3, 64), device="cuda", dtype=dtype)
    slots = torch.tensor([1, 66, -1], device="cuda", dtype=index_dtype)
    cache = torch.zeros((2, 64, 576), device="cuda", dtype=dtype)
    expected = cache.clone()
    expected.view(-1, 576)[slots[:2].long()] = torch.cat((kv[:2], rope[:2]), dim=1)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)
    infinicore.concat_and_cache_mla(
        ic(kv),
        ic(rope),
        ic(cache),
        ic(slots),
        "auto",
        ic(scale),
    )
    infinicore.sync_device()
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def run_graph_replay_case():
    device = infinicore.device("cuda", 0)
    kv = torch.zeros((3, 512), device="cuda", dtype=torch.bfloat16)
    rope = torch.zeros((3, 64), device="cuda", dtype=torch.bfloat16)
    slots = torch.full((3,), -1, device="cuda", dtype=torch.int64)
    cache = torch.zeros((2, 64, 576), device="cuda", dtype=torch.bfloat16)
    scale = torch.ones((1,), device="cuda", dtype=torch.float32)

    infinicore.start_graph_recording(device)
    infinicore.concat_and_cache_mla(
        ic(kv), ic(rope), ic(cache), ic(slots), "auto", ic(scale)
    )
    graph = infinicore.stop_graph_recording()

    kv_replacement = torch.randn_like(kv)
    rope_replacement = torch.randn_like(rope)
    slot_replacement = torch.tensor([1, 66, -1], device="cuda", dtype=torch.int64)
    ic(kv).copy_(ic(kv_replacement))
    ic(rope).copy_(ic(rope_replacement))
    ic(slots).copy_(ic(slot_replacement))
    infinicore.sync_stream()
    graph.run()
    infinicore.sync_stream()

    expected = torch.zeros_like(cache)
    expected.view(-1, 576)[slot_replacement[:2]] = torch.cat(
        (kv_replacement[:2], rope_replacement[:2]), dim=1
    )
    torch.testing.assert_close(cache, expected, rtol=0, atol=0)


def test():
    torch.manual_seed(0)
    for dtype in (torch.float16, torch.bfloat16):
        for index_dtype in (torch.int32, torch.int64):
            run_case(dtype, index_dtype)
    run_graph_replay_case()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("concat_and_cache_mla ok")
