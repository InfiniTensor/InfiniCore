import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# ldexp(input, other) computes input * (2**other)

_TEST_CASES_DATA = [
    ((2, 3), (3,)),
    ((1, 4, 8), None),
    ((3, 2, 5, 7), (1,)),
    ((2, 1, 16), None),
    ((1, 8, 9, 11), None),
    ((2, 6, 10), (1,)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, other_shape in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(shape, None, dtype)

            if other_shape is None:
                other_spec = TensorSpec.from_tensor((1,), None, infinicore.int32)
            else:
                other_spec = TensorSpec.from_tensor(other_shape, None, infinicore.int32)

            # out-of-place
            cases.append(
                TestCase(
                    inputs=[in_spec, other_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ldexp_out",
                )
            )

            # explicit out
            out_spec = TensorSpec.from_tensor(shape, None, dtype)
            cases.append(
                TestCase(
                    inputs=[in_spec, other_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="ldexp_explicit_out",
                )
            )

            # in-place
            cases.append(
                TestCase(
                    inputs=[in_spec, other_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=0,
                    tolerance=tol,
                    description="ldexp_inplace",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """Ldexp operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Ldexp")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.ldexp(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.ldexp(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
