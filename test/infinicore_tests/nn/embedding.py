import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer
from framework.utils import convert_infinicore_to_torch

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (x_shape, weight_shape)
#  weight (Tensor) – the weights of the module of shape (num_embeddings, embedding_dim).
_TEST_CASES_DATA = [
    # Basic cases
    ((1, 5), (32000, 2048)),
    ((2, 5), (32000, 2048)),
    ((2, 2, 5), (32000, 2048)),
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
    Parse test case data and return list of TestCase objects for all operation types.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for x_shape, weight_shape in _TEST_CASES_DATA:
        strides = None

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            x_spec = TensorSpec.from_tensor(
                x_shape,
                strides,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=1,
                high=10000,
                name="x",
            )

            weight_spec = TensorSpec.from_tensor(
                weight_shape, strides, dtype, name="weight"
            )

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[x_spec, weight_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"nn.Embedding - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """nn.Embedding test with simplified implementation"""

    def __init__(self):
        super().__init__("nn.Embedding")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, x, weight):
        """PyTorch nn.Embedding implementation"""

        num_embeddings, embedding_dim = weight.shape

        model = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )

        params_dict = {"weight": weight}
        model.load_state_dict(params_dict)

        with torch.no_grad():
            y = model(x)
        return y

    def infinicore_operator(self, x, weight):
        """InfiniCore nn.Embedding implementation"""

        if x.device.type != "cpu":
            # 将 input的数据 转移到 cpu 上
            x_torch = convert_infinicore_to_torch(x)
            x_torch_cpu = x_torch.contiguous().cpu()

            x = infinicore.from_torch(x_torch_cpu)

        num_embeddings, embedding_dim = weight.shape

        model = infinicore.nn.Embedding(
            num_embeddings,
            embedding_dim,
            device=weight.device,
            dtype=weight.dtype,
        )

        params_dict = {"weight": weight}
        model.load_state_dict(params_dict)

        y = model(x)
        return y


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
