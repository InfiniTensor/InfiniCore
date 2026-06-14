import os
import sys

import torch

import infinicore

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)

# Test Cases: (batch_size, seq_len, num_heads, num_kv_heads, head_size, is_causal)
_TEST_CASES_DATA = [
    (1, 1, 1, 1, 64, True),
    (1, 1, 1, 1, 64, False),
    (2, 1, 4, 4, 128, True),
    (2, 1, 4, 4, 128, False),
    (6, 5, 1, 1, 64, True),
    (6, 5, 1, 1, 64, False),
    (4, 1, 8, 2, 128, True),
    (4, 1, 8, 2, 128, False),
    (3, 14, 8, 2, 128, True),
    (3, 14, 8, 2, 128, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]


def parse_test_cases():
    test_cases = []

    for (
        batch_size,
        seq_len,
        num_heads,
        num_kv_heads,
        head_size,
        is_causal,
    ) in _TEST_CASES_DATA:
        scale = head_size**-0.5
        q_shape = [batch_size, seq_len, num_heads, head_size]
        kv_shape = [batch_size, seq_len, num_kv_heads, head_size]
        qkv_strides = [
            seq_len * (num_heads + 2 * num_kv_heads) * head_size,
            (num_heads + 2 * num_kv_heads) * head_size,
            head_size,
            1,
        ]

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype)

            test_cases.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(
                            q_shape,
                            qkv_strides,
                            dtype=dtype,
                        ),
                        TensorSpec.from_tensor(
                            kv_shape,
                            qkv_strides,
                            dtype=dtype,
                        ),
                        TensorSpec.from_tensor(
                            kv_shape,
                            qkv_strides,
                            dtype=dtype,
                        ),
                    ],
                    kwargs={
                        "scale": scale,
                        "is_causal": is_causal,
                    },
                    tolerance=tolerance,
                    description=f"MHA_{str(dtype).split('.')[-1]}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("PagedAttentionPrefill")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        q,
        k,
        v,
        scale=1.0,
        is_causal=False,
    ):
        assert (
            len(q.shape) == len(k.shape)
            and len(k.shape) == len(v.shape)
            and len(v.shape) == 4
        )
        assert q.shape[3] == k.shape[3] and k.shape[3] == v.shape[3]
        assert k.shape[2] == v.shape[2]

        if k.shape[2] != q.shape[2]:
            k = (
                k.unsqueeze(3)
                .repeat_interleave(q.shape[2] // k.shape[2], 3)
                .reshape(q.shape)
            )
            v = (
                v.unsqueeze(3)
                .repeat_interleave(q.shape[2] // v.shape[2], 3)
                .reshape(q.shape)
            )

        attn_weight = (q.permute(0, 2, 1, 3)) @ (k.permute(0, 2, 3, 1)) * scale
        if is_causal:
            mask = torch.tril(torch.ones_like(attn_weight), -1).flip(dims=[-2, -1])
            attn_weight = torch.where(mask == 1, -torch.inf, attn_weight)
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_o = attn_weight @ v.permute(0, 2, 1, 3)
        return attn_o.permute(0, 2, 1, 3)

    def infinicore_operator(
        self,
        q,
        k,
        v,
        scale=1.0,
        is_causal=False,
    ):
        out = infinicore.mha(
            q,
            k,
            v,
            alibi_slopes=None,
            scale=scale,
            is_causal=is_causal,
        )
        infinicore.sync_stream()
        return out


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
