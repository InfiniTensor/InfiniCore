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

# Test cases format: (shape, input_strides, alpha)
_TEST_CASES_DATA = [
    # Basic ELU tests without alpha (default alpha=1.0)
    ((13, 4), None, None),
    ((13, 4), (10, 1), None),
    ((13, 4), (0, 1), None),
    # 3D tensor tests
    ((13, 4, 4), None, None),
    ((13, 4, 4), (20, 4, 1), None),
    ((13, 4, 4), (4, 0, 1), None),
    # Large tensor tests
    ((16, 5632), None, None),
    ((16, 5632), (13312, 1), None),
    # ELU with different alpha values
    ((8, 4), None, 0.5),
    ((8, 4), (10, 1), 0.5),
    ((8, 4), None, 1.5),
    ((8, 4), (10, 1), 1.5),
    ((16, 8), None, 2.0),
    ((16, 8), (20, 1), 2.0),
    ((16, 8), None, 0.3),
    ((16, 8), (20, 1), 0.3),
    ((32, 16), None, 1.0),
    ((32, 16), (40, 1), 1.0),
    ((32, 16), None, 1.8),
    ((32, 16), (40, 1), 1.8),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse ELU test case data according to format:
    (shape, input_strides, alpha)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        input_strides = data[1] if len(data) > 1 else None
        alpha = data[2] if len(data) > 2 else None

        # Check if input tensor supports in-place operations
        input_supports_inplace = not is_broadcast(input_strides)

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create typed tensor spec
            input_spec = TensorSpec.from_tensor(shape, input_strides, dtype)

            # Build description
            description_parts = ["ELU"]
            if alpha is not None:
                description_parts.append(f"alpha={alpha}")
            if input_strides is not None:
                description_parts.append(f"input_strides={input_strides}")

            base_description = " - ".join(description_parts)

            # Test Case 1: Out-of-place (return value)
            kwargs = {}
            if alpha is not None:
                kwargs["alpha"] = alpha

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"{base_description} - OUT_OF_PLACE",
                )
            )

            # Test Case 2: In-place operation using PyTorch's inplace parameter
            if input_supports_inplace:
                inplace_kwargs = {"inplace": True}
                if alpha is not None:
                    inplace_kwargs["alpha"] = alpha

                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=inplace_kwargs,
                        output_spec=None,
                        comparison_target=0,  # Compare first input (modified in-place)
                        tolerance=tolerance,
                        description=f"{base_description} - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """ELU operator test with PyTorch-compatible implementation"""

    def __init__(self):
        super().__init__("ELU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch ELU implementation"""
        return torch.nn.functional.elu(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore ELU implementation"""
    #     return infinicore.nn.functional.elu(*args, **kwargs)
    #     # return None


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
