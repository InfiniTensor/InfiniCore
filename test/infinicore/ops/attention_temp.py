"""
This is for framework validation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (n_q_head, n_kv_head, seq_len, head_dim, pos, k_cache_buf_len, v_cache_buf_len,
#                    q_strides, k_strides, v_strides, k_cache_strides, v_cache_strides)
_TEST_CASES_DATA = [
    # Prefill stage
    (
        32,
        4,
        5,
        64,
        0,
        2048,
        2048,
        [64, 2560, 1],
        [64, 2560, 1],
        [64, 2560, 1],
        [64, 11264, 1],
        [64, 11264, 1],
    ),
    # Decode stage
    (
        32,
        4,
        1,
        64,
        3,
        2048,
        2048,
        [64, 2560, 1],
        [64, 2560, 1],
        [64, 2560, 1],
        [64, 11264, 1],
        [64, 11264, 1],
    ),
    # Small test case
    (8, 4, 2, 16, 1, 8, 8, None, None, None, None, None),
    # Another prefill case
    (
        28,
        28,
        15,
        128,
        0,
        2048,
        2048,
        [128, 10752, 1],
        [128, 10752, 1],
        [128, 10752, 1],
        [128, 3584, 1],
        [128, 3584, 1],
    ),
]

# Epsilon constant for causal softmax
_EPSILON = 1e-5


def causal_softmax(x):
    """Apply causal mask and softmax to attention scores"""
    input_dtype = x.dtype
    # Create causal mask
    mask = torch.tril(torch.ones_like(x), diagonal=-1).flip(dims=[-2, -1])
    # Apply mask: set masked positions to -inf
    masked = torch.where(mask == 1, -torch.inf, x.to(torch.float32))
    # Apply softmax and convert back to original dtype
    return torch.nn.functional.softmax(masked, dim=-1).to(input_dtype)


def torch_attention(q, k, v, k_cache, v_cache, pos):
    """PyTorch reference implementation of attention"""
    input_dtype = q.dtype

    n_q_head = q.shape[0]
    n_kv_head = k.shape[0]

    # Concatenate key and value caches
    k_cache = k_cache[:, :pos, :]  # (n_kv_head, pos, head_dim)
    v_cache = v_cache[:, :pos, :]  # (n_kv_head, pos, head_dim)
    k = torch.cat([k_cache, k], dim=1)  # (n_kv_head, total_seq_len, head_dim)
    v = torch.cat([v_cache, v], dim=1)  # (n_kv_head, total_seq_len, head_dim)

    total_seq_len = k.shape[1]
    head_dim = v.shape[-1]

    # Handle grouped query attention (GQA)
    if n_q_head != n_kv_head:
        q = q.reshape(
            n_kv_head, -1, head_dim
        )  # (n_kv_head, n_group * seq_len, head_dim)

    # Scaled dot-product attention
    attn_scores = (
        torch.einsum("hqd,hkd->hqk", q.to(torch.float32), k.to(torch.float32))
        .to(input_dtype)
        .reshape(n_q_head, -1, total_seq_len)
    )  # (n_q_head, seq_len, total_seq_len)

    # Scale by sqrt(head_dim)
    attn_scores = attn_scores / (head_dim**0.5)

    # Apply causal softmax
    attn_weights = causal_softmax(attn_scores).reshape(
        n_kv_head, -1, total_seq_len
    )  # (n_kv_head, seq_len, total_seq_len)

    # Weighted sum of values
    attn_output = (
        torch.einsum(
            "hqk,hkd->hqd", attn_weights.to(torch.float32), v.to(torch.float32)
        )
        .to(input_dtype)
        .reshape(n_q_head, -1, head_dim)
        .permute(1, 0, 2)
    )  # (seq_len, n_q_head, head_dim)

    return attn_output


def parse_test_cases():
    """
    Parse attention test case data according to format:
    (n_q_head, n_kv_head, seq_len, head_dim, pos, k_cache_buf_len, v_cache_buf_len,
     q_strides, k_strides, v_strides, k_cache_strides, v_cache_strides)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        n_q_head, n_kv_head, seq_len, head_dim, pos = (
            data[0],
            data[1],
            data[2],
            data[3],
            data[4],
        )
        k_cache_buf_len, v_cache_buf_len = data[5], data[6]
        q_strides = data[7] if len(data) > 7 else None
        k_strides = data[8] if len(data) > 8 else None
        v_strides = data[9] if len(data) > 9 else None
        k_cache_strides = data[10] if len(data) > 10 else None
        v_cache_strides = data[11] if len(data) > 11 else None

        # Check if output tensor supports in-place operations
        # For attention, output shape is (seq_len, n_q_head, head_dim)
        output_shape = (seq_len, n_q_head, head_dim)
        output_supports_inplace = True  # Output is always contiguous for attention

        # Generate test cases for all data types
        for dtype in [infinicore.float16, infinicore.bfloat16, infinicore.float32]:
            tolerance = {
                infinicore.float16: {"atol": 1e-4, "rtol": 1e-2},
                infinicore.float32: {"atol": 1e-5, "rtol": 1e-3},
                infinicore.bfloat16: {"atol": 1e-3, "rtol": 5e-2},
            }.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create typed tensor specs
            q_spec = TensorSpec.from_tensor(
                (n_q_head, seq_len, head_dim), q_strides, dtype
            )
            k_spec = TensorSpec.from_tensor(
                (n_kv_head, seq_len, head_dim), k_strides, dtype
            )
            v_spec = TensorSpec.from_tensor(
                (n_kv_head, seq_len, head_dim), v_strides, dtype
            )
            k_cache_spec = TensorSpec.from_tensor(
                (n_kv_head, k_cache_buf_len, head_dim), k_cache_strides, dtype
            )
            v_cache_spec = TensorSpec.from_tensor(
                (n_kv_head, v_cache_buf_len, head_dim), v_cache_strides, dtype
            )
            pos_spec = TensorSpec.from_scalar(pos)
            output_spec = TensorSpec.from_tensor(
                output_shape, None, dtype
            )  # Output is always contiguous

            # Inputs list
            inputs = [q_spec, k_spec, v_spec, k_cache_spec, v_cache_spec, pos_spec]

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description="Attention - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (attention(q, k, v, k_cache, v_cache, pos, out=output))
            if output_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=inputs,
                        kwargs=None,
                        output_spec=output_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description="Attention - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Attention operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Attention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, q, k, v, k_cache, v_cache, pos, out=None, **kwargs):
        """PyTorch attention implementation"""
        result = torch_attention(q, k, v, k_cache, v_cache, pos)

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, q, k, v, k_cache, v_cache, pos, out=None, **kwargs):
        """InfiniCore attention implementation"""
        return infinicore.attention(q, k, v, k_cache, v_cache, pos, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
