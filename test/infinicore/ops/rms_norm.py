import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import create_test_cases
from framework.base import BaseOperatorTest, TensorSpec
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases: (y_shape, x_shape, w_shape, y_stride, x_stride)
_TEST_CASES_DATA = [
    ((1, 4), (1, 4), (4,), None, None),
    ((2, 4), (2, 4), (4,), None, None),
    ((2, 2, 4), (2, 2, 4), (4,), None, None),
    ((2, 2, 4), (2, 2, 4), (4,), (12, 8, 1), (12, 8, 1)),
    ((16, 2048), (16, 2048), (2048,), None, None),
    ((16, 2048), (16, 2048), (2048,), (4096, 1), (4096, 1)),
    ((15, 3584), (15, 3584), (3584,), None, None),
    ((4, 4, 2048), (4, 4, 2048), (2048,), None, None),
    ((4, 4, 2048), (4, 4, 2048), (2048,), (2048, 8192, 1), (2048, 8192, 1)),
    ((4, 4, 2048), (4, 4, 2048), (2048,), (16384, 4096, 1), (16384, 4096, 1)),
    ((15, 3584), (15, 3584), (3584,), None, None),
    ((15, 8192), (15, 8192), (8192,), None, None),
]

# Parameter mapping configuration for rms_norm operator
# Format: (y_shape, x_shape, w_shape, y_stride, x_stride)
# Call signature: rms_norm(x, weight)
_RMS_NORM_PARAMETER_MAPPING = (
    "rms_norm",  # operator_name
    "rms_norm(x, weight)",  # call_signature
    [  # input_configs
        {"shape": 1, "stride": 4},  # x: shape from index 1, stride from index 4
        {
            "shape": 2,
            "stride": None,
        },  # weight: shape from index 2, no stride (1D tensor)
    ],
    {"shape": 0, "stride": 3},  # output y: shape from index 0, stride from index 3
)

# Parse test cases using rms_norm parameter mapping
_TEST_CASES = create_test_cases(_TEST_CASES_DATA, _RMS_NORM_PARAMETER_MAPPING)

# Data types for individual tensors
_INPUT_DTYPES = [infinicore.float16, infinicore.bfloat16]
_WEIGHT_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Generate all dtype combinations (2 input dtypes × 3 weight dtypes = 6 combinations)
_DTYPE_COMBINATIONS = []
for input_dtype in _INPUT_DTYPES:
    for weight_dtype in _WEIGHT_DTYPES:
        # For RMSNorm: input and output have same dtype, weight can be different
        _DTYPE_COMBINATIONS.append(
            {
                "input_0": input_dtype,  # x tensor
                "input_1": weight_dtype,  # weight tensor
                "output": input_dtype,  # output tensor (same as input)
            }
        )

# Base data types (for backward compatibility)
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 2e-3, "rtol": 2e-3},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
}

# EPSILON constant for RMSNorm
_EPSILON = 1e-5

# ==============================================================================
# Operator test class with specific test functions
# ==============================================================================


class RMSNormTest(BaseOperatorTest):
    """RMSNorm test with operator-specific test functions and mixed dtype support"""

    def __init__(self):
        super().__init__("rms_norm")

    def get_test_cases(self):
        return _TEST_CASES

    def get_tensor_dtypes(self):
        return _TENSOR_DTYPES

    def get_tolerance_map(self):
        return _TOLERANCE_MAP

    def get_dtype_combinations(self):
        """Override to provide mixed dtype combinations for RMSNorm"""
        return _DTYPE_COMBINATIONS

    def torch_operator_inplace(self, x, weight, out=None, **kwargs):
        """PyTorch in-place RMSNorm operation"""
        # Use the provided RMSNorm implementation
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        scale = hidden_states.pow(2).mean(-1, keepdim=True).add_(_EPSILON).rsqrt_()
        out.set_((hidden_states.mul_(scale).mul_(weight)).to(input_dtype))

    def infinicore_operator_inplace(self, x, weight, out=None, **kwargs):
        """Infinicore in-place RMSNorm operation"""
        infinicore.rms_norm(x, weight, _EPSILON, out=out)

    def torch_operator_out_of_place(self, x, weight, **kwargs):
        """PyTorch out-of-place RMSNorm operation"""
        # For out-of-place, we need to create the output tensor first
        input_dtype = x.dtype
        hidden_states = x.to(torch.float32)
        scale = hidden_states.pow(2).mean(-1, keepdim=True).add_(_EPSILON).rsqrt_()
        result = (hidden_states * scale * weight).to(input_dtype)
        return result

    def infinicore_operator_out_of_place(self, x, weight, **kwargs):
        """Infinicore out-of-place RMSNorm operation"""
        return infinicore.rms_norm(x, weight, _EPSILON)


# ==============================================================================
# Main execution
# ==============================================================================


def main():
    """Main entry point"""
    runner = GenericTestRunner(RMSNormTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
