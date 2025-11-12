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

# Test cases format: (shape, a_strides, b_strides, c_strides)
_TEST_CASES_DATA = [
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), (10, 1)),
    ((13, 4), (0, 1), None, None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), (20, 4, 1)),
    ((13, 4, 4), (4, 0, 1), (0, 4, 1), None),
    ((16, 5632), None, None, None),
    ((16, 5632), (13312, 1), (13312, 1), (13312, 1)),
]

# Data types
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]

# Tolerance
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}


def build_test_cases():
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        c_strides = data[3] if len(data) > 3 else None

        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)
        c_supports_inplace = not is_broadcast(c_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)
            c_spec = TensorSpec.from_tensor(shape, c_strides, dtype)

            # Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"Mul - OUT_OF_PLACE",
                )
            )

            # With explicit output tensor (mul(a, b, out=c))
            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={},
                        output_spec=c_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"Mul - INPLACE(out)",
                    )
                )

            # In-place on first input (mul(a, b, out=a))
            if a_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description=f"Mul - INPLACE(a)",
                    )
                )

            # In-place on second input (mul(a, b, out=b))
            if b_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=tolerance,
                        description=f"Mul - INPLACE(b)",
                    )
                )

    return test_cases


_TEST_CASES = build_test_cases()


class OpTest(BaseOperatorTest):
    """Mul test with simplified test case parsing"""

    def __init__(self):
        super().__init__("Mul")

    def get_test_cases(self):
        return _TEST_CASES

    def torch_operator(self, a, b, out=None, **kwargs):
        return torch.mul(a, b, out=out)

    def infinicore_operator(self, a, b, out=None, **kwargs):
        try:
            return infinicore.mul(a, b, out=out)
        except AttributeError as exc:
            raise NotImplementedError("InfiniCore mul operator not available") from exc


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
