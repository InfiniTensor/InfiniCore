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
_TEST_CASES_DATA = [
    # bs, ntoken, total_token, num_attention_heads, num_key_value_heads, head_dim
    (1, 4, 4, 8, 8, 64),
    (1, 1, 4, 8, 8, 64),
    (4, 16, 16, 32, 8, 64),
    (4, 1, 128, 32, 8, 64),
]


# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 5e-2, "rtol": 5e-2},
}


# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
# _TENSOR_DTYPES = [infinicore.bfloat16]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for sdpa operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        bs = data[0]
        ntoken, total_token = data[1], data[2]
        num_attention_heads, num_key_value_heads = data[3], data[4]
        head_dim = data[5]

        # Determine shapes based on batch dimension
        query_shape = (bs, num_attention_heads, ntoken, head_dim)
        key_shape = (bs, num_key_value_heads, total_token, head_dim)
        value_shape = (bs, num_key_value_heads, total_token, head_dim)
        out_shape = (bs, num_attention_heads, ntoken, head_dim)

        # Check if tensors support in-place operations
        c_supports_inplace = not is_broadcast(out_shape)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            query_spec = TensorSpec.from_tensor(query_shape, None, dtype)
            key_spec = TensorSpec.from_tensor(key_shape, None, dtype)
            value_spec = TensorSpec.from_tensor(value_shape, None, dtype)
            out_spec = TensorSpec.from_tensor(out_shape, None, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[query_spec, key_spec, value_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"sdpa - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[query_spec, key_spec, value_spec],
                        kwargs=None,
                        output_spec=out_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"sdpa - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """sdpa operator test with simplified implementation"""

    def __init__(self):
        super().__init__("sdpa")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, query, key, value, out=None, **kwargs):
        """PyTorch sdpa implementation"""
        ntoken = query.shape[-2]
        total_token = key.shape[-2]

        is_causal = True
        if 1 == ntoken and total_token > 1:
            is_causal = False

        result = torch.nn.functional.scaled_dot_product_attention(
            query, key, value, is_causal=is_causal, enable_gqa=True
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, query, key, value, out=None, **kwargs):
        """InfiniCore sdpa implementation"""
        return infinicore.nn.functional.self_attention(query, key, value, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
