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

# Test cases format: (input_shape)
# The operator splits the last dimension: Input (..., 2*d) -> Output (..., d)
_TEST_CASES_DATA = [
    (2, 4),
    (1024, 1024),
    (2, 4, 8),
    (1, 22016),
    (2, 4, 256),
    (2, 4, 16, 256),
]

# Tolerance configuration for different precisions
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse SiLUAndMul test case data.
    Input shape: [..., 2*d], Output shape: [..., d]
    Note: In-place is not supported due to shape mismatch between input and output.
    """
    test_cases = []

    for input_shape in _TEST_CASES_DATA:
        # Calculate output shape based on SwiGLU logic
        output_shape = list(input_shape)
        output_shape[-1] //= 2
        output_shape = tuple(output_shape)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(input_shape, None, dtype)
            output_spec = TensorSpec.from_tensor(output_shape, None, dtype)

            # Case 1: Functional style (allocates new memory for output)
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"SiLUAndMul_Functional_{dtype}",
                )
            )

            # Case 2: Explicit output tensor style (uses pre-allocated buffer)
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=None,
                    output_spec=output_spec,
                    comparison_target="out",
                    tolerance=tolerance,
                    description=f"SiLUAndMul_OutParam_{dtype}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """SiLUAndMul operator test (SwiGLU activation)"""

    def __init__(self):
        super().__init__("SiLUAndMul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, out=None, **kwargs):
        """
        PyTorch SwiGLU reference implementation:
        Formula: SiLU(gate) * up, where [gate, up] = split(input)
        """
        d = input.shape[-1] // 2
        # Split the last dimension into two equal parts
        gate, up = torch.split(input, [d, d], dim=-1)
        result = torch.nn.functional.silu(gate) * up

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        """InfiniCore SiLUAndMul implementation wrapper"""
        import infinicore.nn.functional as F

        return F.silu_and_mul(input, out=out)


def main():
    """Main entry point for the test runner"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
