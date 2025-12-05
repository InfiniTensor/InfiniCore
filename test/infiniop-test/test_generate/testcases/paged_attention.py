import numpy as np
import gguf
from typing import List
from enum import Enum, auto

# Assuming these helpers are in a shared utility file
from .. import (
    InfiniopTestWriter,
    InfiniopTestCase,
    np_dtype_to_ggml,
    gguf_strides,
    contiguous_gguf_strides,
)


# ==============================================================================
#  NumPy Reference Implementation
# ==============================================================================
def ref_paged_attention_np(
    q, k_cache, v_cache, block_tables, seq_lens, scale, alibi_slopes
):
    # This is a simplified NumPy implementation for correctness checking.
    # It mirrors the logic of the PyTorch reference.
    output = np.empty_like(q, dtype=np.float64)
    num_seqs, num_heads, head_size = q.shape
    num_kv_heads = v_cache.shape[1]
    num_queries_per_kv = num_heads // num_kv_heads
    block_size = v_cache.shape[3]

    for i in range(num_seqs):
        seq_len = seq_lens[i]
        q_i = q[i]

        keys_lst = []
        values_lst = []
        for j in range(seq_len):
            block_num = block_tables[i, j // block_size]
            block_off = j % block_size
            k = k_cache[block_num, :, :, block_off, :].reshape(num_kv_heads, head_size)
            v = v_cache[block_num, :, :, block_off]
            keys_lst.append(k)
            values_lst.append(v)

        keys = np.stack(keys_lst, axis=0)
        values = np.stack(values_lst, axis=0)
        if num_queries_per_kv > 1:
            keys = np.repeat(keys, num_queries_per_kv, axis=1)
            values = np.repeat(values, num_queries_per_kv, axis=1)

        # einsum in numpy: qhd,khd->hqk
        attn_scores = np.einsum("hd,khd->hk", q_i, keys) * scale

        if alibi_slopes is not None:
            pos = np.arange(seq_len)
            alibi_bias = (pos - seq_len + 1).astype(np.float32)
            alibi_bias = alibi_slopes.reshape(-1, 1) * alibi_bias.reshape(1, -1)
            attn_scores += alibi_bias

        exp_scores = np.exp(attn_scores - np.max(attn_scores, axis=-1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # einsum in numpy: hqk,khd->qhd -> hd
        out_i = np.einsum("hk,khd->hd", probs, values)
        output[i] = out_i

    return output


# ==============================================================================
#  Test Case Definition and Generation
# ==============================================================================
class PagedAttentionTestCase(InfiniopTestCase):
    def __init__(self, **kwargs):
        super().__init__("paged_attention")
        self.tensors = kwargs

    def write_test(self, test_writer: "InfiniopTestWriter"):
        super().write_test(test_writer)
        for name, tensor in self.tensors.items():
            test_writer.add_tensor(
                test_writer.gguf_key(name),
                tensor,
                raw_dtype=np_dtype_to_ggml(tensor.dtype),
            )

        ans = ref_paged_attention_np(
            self.tensors["q"].astype(np.float64),
            self.tensors["k_cache"].astype(np.float64),
            self.tensors["v_cache"].astype(np.float64),
            self.tensors["block_tables"],
            self.tensors["seq_lens"],
            self.tensors["scale"].item(),
            self.tensors.get("alibi_slopes", None),
        )
        test_writer.add_tensor(
            test_writer.gguf_key("ans"), ans, raw_dtype=gguf.GGMLQuantizationType.F64
        )


if __name__ == "__main__":
    test_writer = InfiniopTestWriter("paged_attention.gguf")
    test_cases = []

    # Test case configurations
    _TEST_CASES_ = [(7, 40, 40, 128, 16, 1024), (5, 64, 8, 80, 32, 2048)]
    _TENSOR_DTYPES_ = [np.float16, np.float32]
    _NUM_BLOCKS = 2048

    for dtype in _TENSOR_DTYPES_:
        for (
            num_seqs,
            num_heads,
            num_kv_heads,
            head_size,
            block_size,
            max_seq_len,
        ) in _TEST_CASES_:
            scale = 1.0 / (head_size**0.5)
            x = 16 // dtype().itemsize

            tensors = {
                "q": np.random.randn(num_seqs, num_heads, head_size).astype(dtype),
                "k_cache": np.random.randn(
                    _NUM_BLOCKS, num_kv_heads, head_size // x, block_size, x
                ).astype(dtype),
                "v_cache": np.random.randn(
                    _NUM_BLOCKS, num_kv_heads, head_size, block_size
                ).astype(dtype),
                "seq_lens": np.random.randint(1, max_seq_len, num_seqs, dtype=np.int32),
                "block_tables": np.random.randint(
                    0,
                    _NUM_BLOCKS,
                    (num_seqs, (max_seq_len + block_size - 1) // block_size),
                    dtype=np.int32,
                ),
                "scale": np.array(scale, dtype=np.float32),
                "out": np.empty((num_seqs, num_heads, head_size), dtype=dtype),
            }
            test_cases.append(PagedAttentionTestCase(**tensors))

    test_writer.add_tests(test_cases)
    test_writer.save()
