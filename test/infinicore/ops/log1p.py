import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None)
# infinicore.log1p(input)

_TEST_CASES_DATA = [
    ((2, 3), None),
    ((1, 4, 8), (32, 8, 1)),
    ((3, 2, 5, 7), None),
    ((2, 1, 16), None),
    ((1, 8, 9, 11), (792, 99, 11, 1)),
    ((2, 6, 10), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="log1p_out",
                )
            )
            out_spec = TensorSpec.from_tensor(shape, None, dtype)
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="log1p_out_explicit",
                )
            )
            if not is_broadcast(in_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="log1p_inplace",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    """Log1p operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Log1p")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.log1p(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.log1p(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
