import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, alpha_or_None)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

_TEST_CASES_DATA = [
    ((13, 4), None, None),
    ((13, 4), (10, 1), None),
    ((8, 8), None, 0.5),
    ((16, 16), (256, 1), 1.5),
    ((32, 8), None, 1.0),
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

    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None
        alpha = data[2] if len(data) > 2 else None

        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            # Out-of-place
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
                    description=f"CELU - OUT_OF_PLACE",
                )
            )

            # In-place
            if input_supports_inplace:
                inplace_kwargs = {"inplace": True}
                if alpha is not None:
                    inplace_kwargs["alpha"] = alpha

                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=inplace_kwargs,
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description=f"CELU - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """CELU operator test with simplified implementation"""

    def __init__(self):
        super().__init__("CELU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.celu(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.celu(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
