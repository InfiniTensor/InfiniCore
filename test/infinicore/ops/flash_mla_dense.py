import argparse
import os

import torch
from infinicore.lib import _infinicore

import infinicore


def ic(tensor):
    return infinicore.from_torch(tensor)


def test():
    torch.manual_seed(0)
    batch, heads, blocks = 2, 24, 4
    head_dim, value_dim, block_size = 576, 512, 64
    scale = head_dim**-0.5
    query = torch.randn(
        (batch, 1, heads, head_dim), device="cuda", dtype=torch.bfloat16
    )
    cache = torch.randn(
        (blocks, block_size, 1, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    block_tables = torch.tensor(
        [[0, 1], [2, 3]], device="cuda", dtype=torch.int32
    )
    sequence_lengths = torch.tensor([65, 97], device="cuda", dtype=torch.int32)
    output = torch.empty(
        (batch, 1, heads, value_dim), device="cuda", dtype=torch.bfloat16
    )

    tensors = [ic(x) for x in (output, query, cache, block_tables, sequence_lengths)]
    _infinicore.flash_mla_dense_decode_(
        *(tensor._underlying for tensor in tensors), scale
    )
    infinicore.sync_device()

    expected = []
    for batch_idx, length in enumerate(sequence_lengths.tolist()):
        block_ids = block_tables[batch_idx, : (length + block_size - 1) // block_size]
        kv = cache[block_ids].reshape(-1, head_dim)[:length].float()
        scores = torch.einsum("hd,kd->hk", query[batch_idx, 0].float(), kv) * scale
        expected.append(
            torch.einsum(
                "hk,kv->hv", torch.softmax(scores, dim=-1), kv[:, :value_dim]
            )
        )
    torch.testing.assert_close(
        output[:, 0].float(), torch.stack(expected), rtol=0.03, atol=0.04
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hygon", action="store_true")
    args = parser.parse_args()
    if not args.hygon:
        raise RuntimeError("flash_mla_dense currently requires Hygon")
    if "INFINICORE_HYGON_FLASH_MLA_SO" not in os.environ:
        raise RuntimeError("set INFINICORE_HYGON_FLASH_MLA_SO to the FlashMLA library")
    test()
    print("flash_mla_dense_decode ok")
