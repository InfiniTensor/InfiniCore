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

# Test cases format: (operation_mode, y_shape, x_shape, w_shape, y_strides, x_strides)
_TEST_CASES_DATA = [
    (TestCase.BOTH, (1, 4), (1, 4), (4,), None, None),
    (TestCase.BOTH, (2, 4), (2, 4), (4,), None, None),
    (TestCase.BOTH, (2, 2, 4), (2, 2, 4), (4,), None, None),
    (TestCase.BOTH, (2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1)),
    (TestCase.BOTH, (16, 2048), (16, 2048), (2048,), None, None),
    (TestCase.BOTH, (16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
]


def parse_rms_norm_test_case(data):
    """
    Parse RMSNorm test case data according to format:
    (operation_mode, y_shape, x_shape, w_shape, y_strides, x_strides)
    """
    operation_mode = data[0]
    y_shape = data[1]  # Output shape
    x_shape = data[2]  # Input shape
    w_shape = data[3]  # Weight shape (1D)
    y_strides = data[4] if len(data) > 4 else None
    x_strides = data[5] if len(data) > 5 else None

    # Create input specifications
    inputs = []

    # Input tensor x
    if x_strides is not None:
        inputs.append(TensorSpec.from_strided_tensor(x_shape, x_strides))
    else:
        inputs.append(TensorSpec.from_tensor(x_shape))

    # Weight tensor (1D, always contiguous)
    inputs.append(TensorSpec.from_tensor(w_shape))

    # Output tensor
    if y_strides is not None:
        output = TensorSpec.from_strided_tensor(y_shape, y_strides)
    else:
        output = TensorSpec.from_tensor(y_shape)

    return TestCase(operation_mode, inputs, output)


# Parse test cases
_TEST_CASES = [parse_rms_norm_test_case(data) for data in _TEST_CASES_DATA]

# Data types for individual tensors
_INPUT_DTYPES = [infinicore.float16, infinicore.bfloat16]
_WEIGHT_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Generate all dtype combinations
_DTYPE_COMBINATIONS = []
for input_dtype in _INPUT_DTYPES:
    for weight_dtype in _WEIGHT_DTYPES:
        _DTYPE_COMBINATIONS.append(
            {
                "input_0": input_dtype,  # x tensor
                "input_1": weight_dtype,  # weight tensor
                "output": input_dtype,  # output tensor (same as input)
            }
        )

# Base data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

# EPSILON constant for RMSNorm
_EPSILON = 1e-5


class RMSNormTest(BaseOperatorTest):
    """RMSNorm test with simplified test case parsing"""

    def __init__(self):
        super().__init__("RMS_Norm")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def get_dtype_combinations(self):
        return _DTYPE_COMBINATIONS

    def torch_operator(self, x, weight, out=None, **kwargs):
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        scale = hidden_states.pow(2).mean(-1, keepdim=True).add_(_EPSILON).rsqrt_()
        result = (hidden_states * scale * weight).to(input_dtype)

        if out is not None:
            out.set_(result)
            return out
        else:
            return result

    def infinicore_operator(self, x, weight, out=None, **kwargs):
        return infinicore.rms_norm(x, weight, _EPSILON, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(RMSNormTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
