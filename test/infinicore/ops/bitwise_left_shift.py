import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, a_strides_or_None, b_strides_or_None, out_strides_or_None)
# ==============================================================================
# Operator-specific configuration
# ==============================================================================

_TEST_CASES_DATA = [
    # small shapes
    ((8, 8), None, None, None),
    ((8, 8), (16, 1), None, None),
    # different shapes (broadcasting second operand)
    ((8, 8), None, (0, 1), None),
    ((4, 1), None, None, (8, 1)),
    # 3D tensor
    ((2, 3, 4), None, None, None),
    # large but strided
    ((16, 512), (1024, 1), (0, 1), None),
]

# Integers require exact comparison
_TOLERANCE_MAP = {
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
    infinicore.uint8: {"atol": 0, "rtol": 0},
}

# Data types to test (integer types)
_TENSOR_DTYPES = [infinicore.int32, infinicore.int64, infinicore.uint8]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for bitwise_left_shift.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        out_strides = data[3] if len(data) > 3 else None

        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        out_supports_inplace = not is_broadcast(out_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})

            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)
            out_spec = TensorSpec.from_tensor(shape, out_strides, dtype)

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description="Bitwise left shift - OUT_OF_PLACE",
                )
            )

            # explicit out
            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description="Bitwise left shift - INPLACE(out)",
                    )
                )

            # in-place into first input
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description="Bitwise left shift - INPLACE(a)",
                    )
                )

            # in-place into second input
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=tolerance,
                        description="Bitwise left shift - INPLACE(b)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """BitwiseLeftShift operator test with simplified implementation"""

    def __init__(self):
        super().__init__("BitwiseLeftShift")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.bitwise_left_shift(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.bitwise_left_shift(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
