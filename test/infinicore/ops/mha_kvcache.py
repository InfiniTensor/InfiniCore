import os
import sys

import torch

import infinicore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

# Test Cases: (batch_size, seqlen_q, num_heads, num_kv_heads, head_size, block_size, seqlens_k)
_TEST_CASES_DATA = [
    (1, 1, 1, 1, 64, 256, [1]),
    (2, 1, 4, 4, 64, 256, [7, 250]),
    (2, 1, 8, 2, 128, 256, [73, 260]),
    (3, 1, 8, 1, 128, 256, [1, 257, 511]),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]


def _make_cache(batch_size, num_kv_heads, head_size, block_size, seqlens_k):
    max_blocks_per_seq = max(
        (seq_len + block_size - 1) // block_size for seq_len in seqlens_k
    )
    num_blocks = batch_size * max_blocks_per_seq

    k_cache = torch.zeros((num_blocks, block_size, num_kv_heads, head_size))
    v_cache = torch.zeros_like(k_cache)
    block_table = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32)

    next_block = 0
    for batch_idx, seq_len in enumerate(seqlens_k):
        num_seq_blocks = (seq_len + block_size - 1) // block_size
        blocks = list(range(next_block, next_block + num_seq_blocks))
        next_block += num_seq_blocks
        block_table[batch_idx, :num_seq_blocks] = torch.tensor(
            blocks, dtype=torch.int32
        )

        for logical_pos in range(seq_len):
            block_id = blocks[logical_pos // block_size]
            offset = logical_pos % block_size
            k_cache[block_id, offset] = torch.randn(num_kv_heads, head_size)
            v_cache[block_id, offset] = torch.randn(num_kv_heads, head_size)

    return k_cache, v_cache, block_table


def parse_test_cases():
    test_cases = []

    for (
        batch_size,
        seqlen_q,
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        seqlens_k_data,
    ) in _TEST_CASES_DATA:
        assert batch_size == len(seqlens_k_data)
        assert num_heads % num_kv_heads == 0

        scale = head_size**-0.5
        q = torch.randn((batch_size, seqlen_q, num_heads, head_size))
        k_cache, v_cache, block_table = _make_cache(
            batch_size, num_kv_heads, head_size, block_size, seqlens_k_data
        )
        seqlens_k = torch.tensor(seqlens_k_data, dtype=torch.int32)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype)

            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(
                            q.shape,
                            init_mode=TensorInitializer.MANUAL,
                            set_tensor=q.clone(),
                            dtype=dtype,
                        ),
                        TensorSpec.from_tensor(
                            k_cache.shape,
                            init_mode=TensorInitializer.MANUAL,
                            set_tensor=k_cache.clone(),
                            dtype=dtype,
                        ),
                        TensorSpec.from_tensor(
                            v_cache.shape,
                            init_mode=TensorInitializer.MANUAL,
                            set_tensor=v_cache.clone(),
                            dtype=dtype,
                        ),
                        TensorSpec.from_tensor(
                            seqlens_k.shape,
                            init_mode=TensorInitializer.MANUAL,
                            set_tensor=seqlens_k.clone(),
                            dtype=infinicore.int32,
                        ),
                        TensorSpec.from_tensor(
                            block_table.shape,
                            init_mode=TensorInitializer.MANUAL,
                            set_tensor=block_table.clone(),
                            dtype=infinicore.int32,
                        ),
                    ],
                    kwargs={"scale": scale},
                    tolerance=tolerance,
                    description=f"MHA_KVCache_{str(dtype).split('.')[-1]}",
                )
            )

    return test_cases


def ref_mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, scale):
    output = torch.empty_like(q)
    batch_size, seqlen_q, num_heads, head_size = q.shape
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]

    assert num_heads % num_kv_heads == 0
    group_size = num_heads // num_kv_heads

    for batch_idx in range(batch_size):
        seq_len = seqlens_k[batch_idx].item()
        keys = []
        values = []

        for logical_pos in range(seq_len):
            block_id = block_table[batch_idx, logical_pos // block_size].item()
            offset = logical_pos % block_size
            keys.append(k_cache[block_id, offset])
            values.append(v_cache[block_id, offset])

        K = torch.stack(keys, dim=0)
        V = torch.stack(values, dim=0)
        if group_size > 1:
            K = K.repeat_interleave(group_size, dim=1)
            V = V.repeat_interleave(group_size, dim=1)

        cur_q = q[batch_idx]
        scores = torch.einsum("qhd,khd->hqk", cur_q.float(), K.float()) * scale

        # Decode uses causal attention. For seqlen_q == 1 this permits the whole
        # prefix; the mask also keeps the reference correct if wider decode is added.
        mask = torch.full((seqlen_q, seq_len), float("-inf"), device=q.device)
        prefix_len = seq_len - seqlen_q
        for query_pos in range(seqlen_q):
            mask[query_pos, : prefix_len + query_pos + 1] = 0.0

        attn = torch.softmax(scores + mask.unsqueeze(0), dim=-1).to(q.dtype)
        output[batch_idx] = torch.einsum("hqk,khd->qhd", attn, V)

    return output


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("MhaKVCache")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        scale=1.0,
    ):
        return ref_mha_kvcache(q, k_cache, v_cache, seqlens_k, block_table, scale)

    def infinicore_operator(
        self,
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        scale=1.0,
    ):
        out = infinicore.mha_kvcache(
            q,
            k_cache,
            v_cache,
            seqlens_k,
            block_table,
            alibi_slopes=None,
            scale=scale,
        )
        infinicore.sync_stream()
        return out


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
