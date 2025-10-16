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

# Test cases format: (operation_mode, shape, a_strides, b_strides, c_strides)
_TEST_CASES_DATA = [
    (TestCase.BOTH, (13, 4), None, None, None),
    (TestCase.BOTH, (13, 4), (10, 1), (10, 1), (10, 1)),
    (TestCase.BOTH, (13, 4), (0, 1), None, None),
    (TestCase.BOTH, (13, 4, 4), None, None, None),
    (TestCase.BOTH, (13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    (TestCase.BOTH, (13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    (TestCase.BOTH, (16, 5632), None, None, None),
    (TestCase.BOTH, (16, 5632), (13312, 1), (13312, 1), (13312, 1)),
]


def parse_add_test_case(data):
    """
    Parse add test case data according to format:
    (operation_mode, shape, a_strides, b_strides, c_strides)
    """
    operation_mode = data[0]
    shape = data[1]
    a_strides = data[2] if len(data) > 2 else None
    b_strides = data[3] if len(data) > 3 else None
    c_strides = data[4] if len(data) > 4 else None

    # Create input specifications
    inputs = []

    # Input tensor a
    if a_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, a_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Input tensor b (same shape as a)
    if b_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(shape, b_strides))
    else:
        inputs.append(TensorSpec.from_tensor(shape))

    # Output tensor
    if c_strides is not None:
        output = TensorSpec.from_strided_tensor(shape, c_strides)
    else:
        output = TensorSpec.from_tensor(shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_add_test_case(data) for data in _TEST_CASES_DATA]

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}


class AddTest(BaseOperatorTest):
    """Add test with simplified test case parsing"""

    def __init__(self):
        super().__init__("Add")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def torch_operator(self, a, b, out=None, **kwargs):
        return torch.add(a, b, out=out)

    def infinicore_operator(self, a, b, out=None, **kwargs):
        return infinicore.add(a, b, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(AddTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
