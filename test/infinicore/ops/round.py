import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# round(input, decimals=0)
# We'll test with various decimals including negative values and None.

_TEST_CASES_DATA = [
    ((2, 3), None, 0),
    ((1, 4, 8), None, 1),
    ((3, 2, 5, 7), None, -1),
    ((2, 1, 16), None, 2),
    ((1, 8, 9, 11), None, 0),
    ((2, 6, 10), None, 3),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, strides, decimals in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            # out-of-place
            kwargs = {"decimals": decimals} if decimals is not None else {}
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="round_out",
                )
            )

            # explicit out
            out_spec = TensorSpec.from_tensor(shape, strides, dtype)
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="round_explicit_out",
                )
            )

            # in-place when not broadcast
            if not is_broadcast(strides):
                cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="round_inplace",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    """Round operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Round")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.round(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.round(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
