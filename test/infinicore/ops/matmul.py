import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (operation_mode, nbatch, m, n, k, a_strides, b_strides, c_strides)
# If nbatch is None: a_shape=(m, k), b_shape=(k, n), c_shape=(m, n)
# If nbatch is provided: a_shape=(nbatch, m, k), b_shape=(nbatch, k, n), c_shape=(nbatch, m, n)
_TEST_CASES_DATA = [
    # Basic 2D matmul
    (TestCase.BOTH, None, 2, 4, 3, None, None, None),
    (TestCase.BOTH, None, 128, 64, 256, None, None, None),
    # Batched matmul
    (TestCase.BOTH, 2, 4, 2048, 2048, None, None, None),
    (TestCase.BOTH, 4, 48, 6, 64, None, None, None),
    # Strided tensors
    (TestCase.BOTH, None, 1, 2048, 2048, (4096, 1), (4096, 1), (4096, 1)),
    (TestCase.BOTH, None, 6, 2560, 2048, (2048, 1), (1, 2048), (2560, 1)),
    # Mixed cases
    (TestCase.BOTH, 8, 16, 32, 16, None, None, None),
]


def parse_matmul_test_case(data):
    """
    Parse matmul test case data according to format:
    (operation_mode, nbatch, m, n, k, a_strides, b_strides, c_strides)
    """
    operation_mode = data[0]
    nbatch = data[1]
    m, n, k = data[2], data[3], data[4]
    a_strides = data[5] if len(data) > 5 else None
    b_strides = data[6] if len(data) > 6 else None
    c_strides = data[7] if len(data) > 7 else None

    # Determine shapes based on batch dimension
    if nbatch is None:
        a_shape = (m, k)
        b_shape = (k, n)
        c_shape = (m, n)
    else:
        a_shape = (nbatch, m, k)
        b_shape = (nbatch, k, n)
        c_shape = (nbatch, m, n)

    # Create input specifications
    inputs = []

    # Tensor a
    if a_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(a_shape, a_strides))
    else:
        inputs.append(TensorSpec.from_tensor(a_shape))

    # Tensor b
    if b_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(b_shape, b_strides))
    else:
        inputs.append(TensorSpec.from_tensor(b_shape))

    # Output tensor
    if c_strides is not None:
        output = TensorSpec.from_strided_tensor(c_shape, c_strides)
    else:
        output = TensorSpec.from_tensor(c_shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_matmul_test_case(data) for data in _TEST_CASES_DATA]

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}


class OpTest(BaseOperatorTest):
    """Matmul test with simplified test case parsing"""

    def __init__(self):
        super().__init__("Matmul")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, b, out=None, **kwargs):
        return torch.matmul(a, b, out=out)

    def infinicore_operator(self, a, b, out=None, **kwargs):
        return infinicore.matmul(a, b, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
