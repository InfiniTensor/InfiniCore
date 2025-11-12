import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (q_shape, k_shape, v_shape, attn_mask_or_None, dropout_p, is_causal)
# q/k/v typically have shape (..., seq_len, head_dim) or (batch, seq_len, num_heads, head_dim)

_TEST_CASES_DATA = [
    ((2, 8, 16), (2, 8, 16), (2, 8, 16), None, 0.0, False),
    ((1, 4, 32), (1, 4, 32), (1, 4, 32), None, 0.0, False),
    ((2, 6, 12), (2, 6, 12), (2, 6, 12), None, 0.0, True),
    ((3, 8, 8), (3, 8, 8), (3, 8, 8), None, 0.0, False),
    ((2, 4, 16), (2, 4, 16), (2, 4, 16), None, 0.0, True),
    ((1, 2, 64), (1, 2, 64), (1, 2, 64), None, 0.0, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for q_shape, k_shape, v_shape, attn_mask, dropout_p, is_causal in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            q_spec = TensorSpec.from_tensor(q_shape, None, dtype)
            k_spec = TensorSpec.from_tensor(k_shape, None, dtype)
            v_spec = TensorSpec.from_tensor(v_shape, None, dtype)
            kwargs = {
                "attn_mask": attn_mask,
                "dropout_p": dropout_p,
                "is_causal": is_causal,
            }
            # remove None keys
            kwargs = {k: v for k, v in kwargs.items() if v is not None}

            cases.append(
                TestCase(
                    inputs=[q_spec, k_spec, v_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ScaledDotProductAttention",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """ScaledDotProductAttention operator test with simplified implementation"""

    def __init__(self):
        super().__init__("ScaledDotProductAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.scaled_dot_product_attention(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.scaled_dot_product_attention(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
