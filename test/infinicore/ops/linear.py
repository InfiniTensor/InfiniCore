import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

import infinicore

# ==============================================================================
# Operator-specific configuration
# ==============================================================================
_TEST_CASES_DATA = [
    # bs, n, in_features, out_features, bias
    (1, 5, 2048, 5632, True, None, None, None),
    (1, 1, 2048, 32000, False, None, None, None),
    (2, 5, 2048, 5632, True, None, None, None),
    (2, 5, 256, 2048, False, None, None, None),
    (None, 5, 256, 2048, False, None, None, None),
    (None, 1, 2048, 5632, True, None, None, None),
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
    Parse test case data and return list of TestCase objects for linear operation.
    Each test case contains all necessary information for execution and validation.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        bs = data[0]
        n, in_features, out_features = data[1], data[2], data[3]
        bias = data[4]
        input_strides = data[5] if len(data) > 5 else None
        weight_strides = data[6] if len(data) > 6 else None
        out_strides = data[7] if len(data) > 7 else None

        # Determine shapes based on batch dimension
        if bs is None:
            input_shape = (n, in_features)
            weight_shape = (out_features, in_features)
            out_shape = (n, out_features)
        else:
            input_shape = (bs, n, in_features)
            weight_shape = (out_features, in_features)
            out_shape = (bs, n, out_features)

        if bias is True:
            bias_shape = (out_features,)
        else:
            bias_shape = None

        # Check if tensors support in-place operations
        c_supports_inplace = not is_broadcast(out_shape)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create typed tensor specs
            input_spec = TensorSpec.from_tensor(input_shape, input_strides, dtype)
            weight_spec = TensorSpec.from_tensor(weight_shape, weight_strides, dtype)
            out_spec = TensorSpec.from_tensor(out_shape, out_strides, dtype)

            if bias_shape is not None:
                bias_spec = TensorSpec.from_tensor(bias_shape, None, dtype)
            else:
                bias_spec = None

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, weight_spec, bias_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Linear - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (Linear(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec, weight_spec, bias_spec],
                        kwargs=None,
                        output_spec=out_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Linear - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Linear operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Linear")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch linear implementation"""
        return torch.nn.functional.linear(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore linear implementation"""
        return infinicore.nn.functional.linear(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
