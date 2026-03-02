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

# Test Cases: (num_heads, num_kv_heads, head_size, block_size, [request_batch])
_TEST_CASES_DATA = [
    (1, 1, 128, 256, [(250,), (7,)]),
    (4, 4, 128, 256, [(250,), (7,)]),
    (1, 1, 128, 256, [(260, 73), (1, 1)]),
    (8, 2, 128, 256, [(250,), (7,)]),
    (8, 2, 128, 256, [(260, 73), (1, 1)]),
]

_MAX_SEQUENCE_LENGTH = 8192

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]


class SimpleCacheManager:
    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.free_blocks = list(range(num_blocks))
        self.request_to_blocks = {}
        self.request_to_len = {}

    def allocate_slots(self, request_id, num_new_tokens):
        if request_id not in self.request_to_len:
            self.request_to_len[request_id] = 0
            self.request_to_blocks[request_id] = []

        start_pos = self.request_to_len[request_id]
        new_total_len = start_pos + num_new_tokens
        needed_blocks = (new_total_len + self.block_size - 1) // self.block_size
        added_blocks = needed_blocks - len(self.request_to_blocks[request_id])

        for _ in range(added_blocks):
            self.request_to_blocks[request_id].append(self.free_blocks.pop(0))

        self.request_to_len[request_id] = new_total_len
        return self.request_to_blocks[request_id], new_total_len


def parse_test_cases():
    test_cases = []

    for (
        num_heads,
        num_kv_heads,
        head_size,
        block_size,
        request_batches,
    ) in _TEST_CASES_DATA:
        scale = head_size**-0.5
        num_blocks = 512
        manager = SimpleCacheManager(num_blocks, block_size)
        num_seqs = len(request_batches[0])
        kv_lens = torch.zeros(num_seqs, dtype=torch.int32)

        persistent_k = torch.zeros((num_blocks, num_kv_heads, block_size, head_size))
        persistent_v = torch.zeros((num_blocks, num_kv_heads, block_size, head_size))

        for r, req in enumerate(request_batches):
            assert len(req) == num_seqs, "All requests should have the same length"
            q_lens = torch.tensor(req, dtype=torch.int32)
            kv_lens = kv_lens + q_lens
            total_q_tokens = q_lens.sum().item()
            cum_seqlens_q = torch.zeros(num_seqs + 1, dtype=torch.int32)
            cum_seqlens_q[1:] = torch.cumsum(q_lens, dim=0)
            cum_seqlens_k = torch.zeros(num_seqs + 1, dtype=torch.int32)
            cum_seqlens_k[1:] = torch.cumsum(kv_lens, dim=0)

            query_base = torch.randn((total_q_tokens, num_heads, head_size))

            round_block_tables_list = []
            for i in range(num_seqs):
                p_blocks, total_len = manager.allocate_slots(i, q_lens[i].item())
                round_block_tables_list.append(p_blocks)

                h_len = kv_lens[i].item() - q_lens[i].item()

                for t in range(q_lens[i].item()):
                    logical_pos = h_len + t
                    b_id = p_blocks[logical_pos // block_size]
                    off = logical_pos % block_size
                    persistent_k[b_id, :, off, :] = torch.randn(num_kv_heads, head_size)
                    persistent_v[b_id, :, off, :] = torch.randn(num_kv_heads, head_size)

            max_blks = max(len(t) for t in round_block_tables_list)
            padded_tables = torch.tensor(
                [t + [0] * (max_blks - len(t)) for t in round_block_tables_list]
            )

            for dtype in _TENSOR_DTYPES:
                tolerance = _TOLERANCE_MAP.get(dtype)

                test_cases.append(
                    TestCase(
                        inputs=[
                            TensorSpec.from_tensor(
                                query_base.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=query_base.clone(),
                                dtype=dtype,
                            ),
                            TensorSpec.from_tensor(
                                persistent_k.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=persistent_k.clone(),
                                dtype=dtype,
                            ),
                            TensorSpec.from_tensor(
                                persistent_v.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=persistent_v.clone(),
                                dtype=dtype,
                            ),
                            TensorSpec.from_tensor(
                                padded_tables.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=padded_tables.clone(),
                                dtype=infinicore.int32,
                            ),
                            TensorSpec.from_tensor(
                                cum_seqlens_q.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=cum_seqlens_q.clone(),
                                dtype=infinicore.int32,
                            ),
                            TensorSpec.from_tensor(
                                cum_seqlens_k.shape,
                                init_mode=TensorInitializer.MANUAL,
                                set_tensor=cum_seqlens_k.clone(),
                                dtype=infinicore.int32,
                            ),
                        ],
                        kwargs={
                            "scale": scale,
                            "max_seqlen_q": _MAX_SEQUENCE_LENGTH,
                            "max_seqlen_k": _MAX_SEQUENCE_LENGTH,
                        },
                        tolerance=tolerance,
                        description=f"MHA_Varlen_Round_{r}_{str(dtype).split('.')[-1]}",
                    )
                )

    return test_cases


def ref_paged_attention_multi_turn(
    query, k_cache, v_cache, block_tables, cum_seqlens_q, cum_seqlens_k, scale
):
    output = torch.zeros_like(query)
    num_seqs = len(cum_seqlens_q) - 1
    block_size = k_cache.shape[2]

    for i in range(num_seqs):
        q_start, q_end = cum_seqlens_q[i].item(), cum_seqlens_q[i + 1].item()
        cur_q = query[q_start:q_end]
        q_len = q_end - q_start
        h_len = (cum_seqlens_k[i + 1].item() - cum_seqlens_k[i].item()) - q_len
        total_len = h_len + q_len

        table = block_tables[i]
        keys, values = [], []
        for j in range(total_len):
            b_id = table[j // block_size].item()
            off = j % block_size
            keys.append(k_cache[b_id, :, off, :])
            values.append(v_cache[b_id, :, off, :])

        K = torch.stack(keys, dim=0)
        V = torch.stack(values, dim=0)

        q_heads = cur_q.shape[1]
        kv_heads = K.shape[1]

        assert q_heads % kv_heads == 0
        group_size = q_heads // kv_heads
        if group_size > 1:
            K = K.repeat_interleave(group_size, dim=1)
            V = V.repeat_interleave(group_size, dim=1)

        scores = torch.einsum("qhd,khd->hqk", cur_q.float(), K.float()) * scale
        mask = torch.full((q_len, total_len), float("-inf"), device=query.device)
        for t in range(q_len):
            mask[t, : h_len + t + 1] = 0.0

        attn = torch.softmax(scores + mask.unsqueeze(0), dim=-1).to(query.dtype)
        output[q_start:q_end] = torch.einsum("hqk,khd->qhd", attn, V)
    return output


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("PagedAttentionPrefill")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        query,
        k_cache,
        v_cache,
        block_tables,
        cum_seqlens_q,
        cum_seqlens_k,
        scale=1.0,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):
        return ref_paged_attention_multi_turn(
            query, k_cache, v_cache, block_tables, cum_seqlens_q, cum_seqlens_k, scale
        )

    def infinicore_operator(
        self,
        query,
        k_cache,
        v_cache,
        block_tables,
        cum_seqlens_q,
        cum_seqlens_k,
        scale=1.0,
        max_seqlen_q=0,
        max_seqlen_k=0,
    ):
        out = infinicore.mha_varlen(
            query,
            k_cache.permute([0, 2, 1, 3]),
            v_cache.permute([0, 2, 1, 3]),
            cum_seqlens_q,
            cum_seqlens_k,
            block_tables,
            max_seqlen_q,
            max_seqlen_k,
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
