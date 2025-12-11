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

# Test cases format: (operation_mode, shape, input_strides, output_strides)
# Causal softmax is a single-input function that applies causal masking before softmax
_TEST_CASES_DATA = [
    # Basic 2D causal softmax
    ((3, 3), None, None),
    ((32, 512), None, None),
    # Strided tensors
    ((32, 512), (1024, 1), (1024, 1)),
    # 3D causal softmax
    ((32, 5, 5), None, None),
    ((32, 20, 512), None, None),
    ((32, 20, 512), (20480, 512, 1), None),
    ((28, 15, 15), None, None),
]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 3e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 5e-2},
}

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse causal_softmax test case data according to format:
    (shape, input_strides, output_strides)
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
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

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
                    description=f"Causal Softmax - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place with explicit output tensor (causal_softmax(input, out=output))
            if output_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=output_spec,  # Specify the output tensor spec
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Causal Softmax - INPLACE(out)",
                    )
                )

            # Test Case 3: In-place on first input (causal_softmax(input, out=input))
            if input_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs={"out": 0},  # Use index 0 for first input
                        output_spec=None,
                        comparison_target=0,  # Compare first input
                        tolerance=tolerance,
                        description=f"Causal Softmax - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """CausalSoftmax test with simplified test case parsing"""

    def __init__(self):
        super().__init__("CausalSoftmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_causal_softmax(self, input, out=None, **kwargs):
        # Causal softmax implementation: apply causal mask then softmax
        dtype = input.dtype

        # Create causal mask
        mask = torch.tril(torch.ones_like(input), diagonal=-1).flip(dims=[-2, -1])
        masked = torch.where(mask == 1, -torch.inf, input.to(torch.float32))

        result = torch.nn.functional.softmax(masked, dim=-1, dtype=dtype)

        if out is not None:
            out.copy_(result)
            return out
        return result

    def torch_operator(self, *args, **kwargs):
        return self.torch_causal_softmax(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        import infinicore.nn.functional as F

        return F.causal_softmax(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
