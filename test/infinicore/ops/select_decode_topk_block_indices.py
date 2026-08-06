import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def run_case(rows, columns, topk, seq_lens_values=None):
    logits = torch.randn((rows, columns), device="cuda", dtype=torch.float32)
    if seq_lens_values is None:
        seq_lens = torch.linspace(
            max(1, columns // 2), columns, rows, device="cuda", dtype=torch.int32
        )
    else:
        seq_lens = torch.tensor(seq_lens_values, device="cuda", dtype=torch.int32)
    expected = torch.full((rows, topk), -1, device="cuda", dtype=torch.int32)
    for row in range(rows):
        valid = int(seq_lens[row].item())
        selected = min(valid, topk)
        expected[row, :selected] = torch.topk(
            logits[row, :valid], selected
        ).indices.int()
    actual = torch.empty_like(expected)
    infinicore.select_decode_topk_block_indices_(ic(actual), ic(logits), ic(seq_lens))
    infinicore.sync_device()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test():
    torch.manual_seed(0)
    run_case(2, 128, 16)
    run_case(4, 4096, 2048)
    run_case(1, 8192, 2048)
    run_case(4, 8192, 2048, [1280, 2048, 4096, 8192])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("select_decode_topk_block_indices ok")
