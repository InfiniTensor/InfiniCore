import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (a_shape, a_strides_or_None, b_shape_or_None)
# infinicore.hypot(a, b)

_TEST_CASES_DATA = [
    ((2, 3, 4), None, None),
    ((1, 4, 8), (32, 8, 1), None),
    ((3, 2, 5, 7), None, None),
    ((2, 1, 16), None, None),
    ((1, 8, 9, 11), (792, 99, 11, 1), None),
    ((2, 6, 10), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for a_shape, a_strides, b_shape in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            a_spec = TensorSpec.from_tensor(a_shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(
                a_shape if b_shape is None else b_shape, None, dtype
            )

            cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="hypot_out_of_place",
                )
            )

            out_spec = TensorSpec.from_tensor(a_shape, None, dtype)
            cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="hypot_explicit_out",
                )
            )

            if not is_broadcast(a_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="hypot_inplace_a",
                    )
                )
            if not is_broadcast(b_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs={"out": 1},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=tol,
                        description="hypot_inplace_b",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    """Hypot operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Hypot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.hypot(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.hypot(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
