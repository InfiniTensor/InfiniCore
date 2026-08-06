import argparse

import torch

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def run_case(rows, columns, topk):
    logits = torch.randn((rows, columns), device="cuda", dtype=torch.float32)
    starts = torch.tensor([0, 3, 7, 11][:rows], device="cuda", dtype=torch.int32)
    ends = torch.tensor([1, 17, columns, 11][:rows], device="cuda", dtype=torch.int32)
    output = torch.empty((rows, topk), device="cuda", dtype=torch.int32)
    expected = torch.full_like(output, -1)
    for row in range(rows):
        start = int(starts[row])
        end = min(int(ends[row]), columns)
        valid = max(end - start, 0)
        selected = min(valid, topk)
        if selected:
            expected[row, :selected] = (
                torch.topk(logits[row, start:end], selected, sorted=True).indices
                + start
            ).to(torch.int32)
    infinicore.select_prefill_topk_block_indices_(
        ic(output), ic(logits), ic(starts), ic(ends)
    )
    infinicore.sync_device()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


def test():
    torch.manual_seed(0)
    run_case(4, 64, 8)
    run_case(4, 4096, 2048)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    parser.parse_args()
    test()
    print("select_prefill_topk_block_indices ok")
