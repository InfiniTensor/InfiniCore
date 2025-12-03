import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner
from framework.tensor import TensorInitializer
from framework.utils import (
    convert_infinicore_to_torch,
    infinicore_tensor_from_torch,
    to_torch_dtype,
)

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================
_TEST_CASES_DATA = [
    # bs, ntok, vocab_size, embedding_dim, type
    (1, 5, 32000, 4, infinicore.int64),
    (2, 10, 32000, 2048, infinicore.int32),
    (1, 5, 10, 10, infinicore.int64),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for Embedding operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        bs, ntok = data[0], data[1]
        vocab_size, embedding_dim = data[2], data[3]
        input_type = data[4]

        input_strides = None
        weight_strides = None

        # Determine shapes
        input_shape = (bs, ntok)

        weight_shape = (vocab_size, embedding_dim)

        # Check if tensors support in-place operations
        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            input_spec = TensorSpec.from_tensor(
                input_shape,
                input_strides,
                input_type,
                init_mode=TensorInitializer.RANDINT,
                low=1,
                high=9,
            )
            weight_spec = TensorSpec.from_tensor(weight_shape, weight_strides, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, weight_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Embedding - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Embedding operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Embedding")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, out=None, **kwargs):
        """PyTorch Embedding implementation"""

        return torch.nn.functional.embedding(*args, **kwargs)

    def infinicore_operator(self, input, weight, out=None, **kwargs):
        """InfiniCore Embedding implementation"""

        if input.device.type == "cpu":
            input_cpu = input
        else:
            # 将 input的数据 转移到 cpu 上
            torch_reference = torch.zeros(
                input.shape,
                dtype=to_torch_dtype(input.dtype),
                device="cpu" if "cpu" == input.device.type else "cuda",
            )
            torch_reference = convert_infinicore_to_torch(input)
            torch_reference = torch_reference.contiguous().cpu()

            # 创建cpu的 input
            input_cpu = infinicore_tensor_from_torch(torch_reference)

        return infinicore.nn.functional.embedding(input_cpu, weight, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
