import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (shape, input_strides, output_strides)
# SiLU is a single-input activation function: output = input * sigmoid(input)
_TEST_CASES_DATA = [
    # Basic 2D SiLU
    ((2, 4), None, None),
    ((128, 64), None, None),
    # 3D SiLU
    ((2, 4, 8), None, None),
    ((4, 48, 6), None, None),
    # Strided tensors
    ((1, 2048), (4096, 1), (4096, 1)),
    # Mixed cases
    ((8, 16, 32), None, None),
    # Large tensors
    ((16, 5632), None, None),
    ((4, 4, 5632), None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse SiLU test case data according to format:
    (shape, input_strides, output_strides)
    SiLU only supports out-of-place and in-place modes
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        input_strides = data[1] if len(data) > 1 else None
        output_strides = data[2] if len(data) > 2 else None

        # Check if tensors support in-place operations
        input_supports_inplace = not is_broadcast(input_strides)
        output_supports_inplace = not is_broadcast(output_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create typed tensor specs
            input_spec = TensorSpec.from_tensor(shape, input_strides, dtype)
            output_spec = TensorSpec.from_tensor(shape, output_strides, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"SiLU - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (silu(input, out=output))
            if output_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=output_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"SiLU - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on first input (silu(input, out=input))
            if input_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs={"out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input
                        tolerance=tolerance,
                        description=f"SiLU - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """SiLU operator test with simplified implementation"""

    def __init__(self):
        super().__init__("SiLU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, out=None, **kwargs):
        """PyTorch SiLU implementation: input * sigmoid(input)"""
        sigmoid_input = torch.sigmoid(input)
        result = input * sigmoid_input

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        """InfiniCore SiLU implementation"""
        import infinicore.nn.functional as F

        return F.silu(input, out=out)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
