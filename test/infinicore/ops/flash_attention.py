import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

import infinicore

# Test cases format: (q_shape, k_shape, v_shape, attn_mask_or_None, dropout_p, is_causal)
# q/k/v typically have shape (..., seq_len, head_dim) or (batch, seq_len, num_heads, head_dim)

_TEST_CASES_DATA = [
    ((1, 1, 2, 16), (1, 1, 8, 16), (1, 1, 8, 16), None, 0.0, False),
    ((1, 2, 128, 16), (1, 2, 256, 16), (1, 2, 256, 16), None, 0.0, False),
    ((1, 1, 4, 32), (1, 1, 32, 32), (1, 1, 32, 32), None, 0.0, True),
    ((1, 8, 256, 16), (1, 8, 512, 16), (1, 8, 512, 16), None, 0.0, True),
    ((1, 8, 4, 16), (1, 8, 64, 16), (1, 8, 64, 16), None, 0.0, False),
    ((2, 4, 3, 16), (2, 2, 8, 16), (2, 2, 8, 16), None, 0.0, True),
    ((8, 28, 256, 128), (8, 28, 512, 128), (8, 28, 512, 128), None, 0.0, True),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    import random

    cases = []
    for q_shape, k_shape, v_shape, attn_mask, dropout_p, is_causal in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            q_spec = TensorSpec.from_tensor(q_shape, None, dtype)
            k_spec = TensorSpec.from_tensor(k_shape, None, dtype)
            v_spec = TensorSpec.from_tensor(v_shape, None, dtype)

            len_shape = (q_shape[0],)
            total_len = random.randint(1, k_shape[2])
            total_kv_len_spec = TensorSpec.from_tensor(
                len_shape,
                None,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=total_len,
                high=total_len + 1,
            )

            kwargs = {
                "attn_mask": attn_mask,
                "dropout_p": dropout_p,
                "is_causal": is_causal,
            }
            # remove None keys
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            cases.append(
                TestCase(
                    inputs=[q_spec, k_spec, v_spec, total_kv_len_spec, total_len],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Flash Attention",
                )
            )

    return cases


def torch_flash_attn(q, k, v, total_kv_len, cheat, **kwargs):
    assert kwargs.get("attn_mask") is None
    assert kwargs.get("dropout_p", 0.0) == 0.0

    is_causal = kwargs.get("is_causal", False)
    scale = kwargs.get("scale", q.shape[-1] ** -0.5)
    batch, num_q_heads, query_len, head_dim = q.shape
    num_kv_heads = k.shape[1]
    assert num_q_heads % num_kv_heads == 0

    outs = []
    lengths = total_kv_len.detach().cpu().tolist()
    for b in range(batch):
        kv_len = min(int(lengths[b]), k.shape[2])
        q_b = q[b : b + 1].float()
        k_b = k[b : b + 1, :, :kv_len, :].float()
        v_b = v[b : b + 1, :, :kv_len, :].float()
        if num_q_heads != num_kv_heads:
            group_size = num_q_heads // num_kv_heads
            k_b = k_b.repeat_interleave(group_size, dim=1)
            v_b = v_b.repeat_interleave(group_size, dim=1)

        logits = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
        if is_causal:
            first_query_key = max(kv_len - query_len, 0)
            q_pos = torch.arange(query_len, device=q.device)[:, None]
            k_pos = torch.arange(kv_len, device=q.device)[None, :]
            mask = k_pos <= (first_query_key + q_pos)
            logits = logits.masked_fill(
                ~mask.view(1, 1, query_len, kv_len), float("-inf")
            )

        attn = torch.softmax(logits, dim=-1)
        outs.append(torch.matmul(attn, v_b).to(q.dtype))

    return torch.cat(outs, dim=0)


def infini_flash_attn(q, k, v, total_kv_len, cheat, **kwargs):
    return infinicore.nn.functional.flash_attention(q, k, v, total_kv_len, **kwargs)


class OpTest(BaseOperatorTest):
    """ScaledDotProductAttention operator test with simplified implementation"""

    def __init__(self):
        super().__init__("ScaledDotProductAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_flash_attn(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infini_flash_attn(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
