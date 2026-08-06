import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test_map_decode():
    request_ids = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    block_table = torch.tensor([[2, 3], [5, 6]], device="cuda", dtype=torch.int32)
    token_indices = torch.tensor(
        [[0, 63, 64, -1, 128], [3, 66, -1, -1, 130]],
        device="cuda",
        dtype=torch.int32,
    )
    expected = torch.tensor(
        [[128, 191, 192, -1, -1], [323, 386, -1, -1, -1]],
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty_like(token_indices)
    infinicore.map_decode_request_block_indices_(
        ic(output), ic(request_ids), ic(block_table), ic(token_indices), 64
    )
    infinicore.sync_device()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test_context_lens():
    indices = torch.tensor(
        [[[1, 2, -1, -1]], [[3, -1, 5, -1]]],
        device="cuda",
        dtype=torch.int32,
    )
    output = torch.empty((2,), device="cuda", dtype=torch.int32)
    infinicore.topk_indices_context_lens_(ic(output), ic(indices))
    infinicore.sync_device()
    torch.testing.assert_close(
        output,
        torch.tensor([2, 2], device="cuda", dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test():
    test_map_decode()
    test_context_lens()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("dsa_index_utils ok")
