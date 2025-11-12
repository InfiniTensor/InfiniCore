import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, t1_shape_or_None, t2_shape_or_None, value)

_TEST_CASES_DATA = [
    ((2, 3, 4), None, None, None, 1.0),
    ((1, 4, 8), (32, 8, 1), None, None, 0.5),
    ((3, 2, 5, 7), None, None, None, 2.0),
    ((2, 1, 16), None, None, None, 1.0),
    ((1, 8, 9, 11), (792, 99, 11, 1), None, None, 1.5),
    ((2, 6, 10), None, None, None, 0.25),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for in_shape, in_strides, t1_shape, t2_shape, value in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            input_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            t1_spec = TensorSpec.from_tensor(
                in_shape if t1_shape is None else t1_shape, None, dtype
            )
            t2_spec = TensorSpec.from_tensor(
                in_shape if t2_shape is None else t2_shape, None, dtype
            )

            # Out-of-place
            kwargs = {"value": value}
            cases.append(
                TestCase(
                    inputs=[input_spec, t1_spec, t2_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="addcdiv - OUT_OF_PLACE",
                )
            )

            # Explicit out
            out_spec = TensorSpec.from_tensor(in_shape, None, dtype)
            cases.append(
                TestCase(
                    inputs=[input_spec, t1_spec, t2_spec],
                    kwargs=kwargs,
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="addcdiv - INPLACE(out)",
                )
            )

            # In-place on input (out=0)
            if not is_broadcast(input_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[input_spec, t1_spec, t2_spec],
                        kwargs={"value": value, "out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tol,
                        description="addcdiv - INPLACE(a)",
                    )
                )

            # In-place on tensor1 (out=1)
            if not is_broadcast(t1_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[input_spec, t1_spec, t2_spec],
                        kwargs={"value": value, "out": 1},
                        output_spec=None,
                        comparison_target=1,
                        tolerance=tol,
                        description="addcdiv - INPLACE(b)",
                    )
                )

            # In-place on tensor2 (out=2)
            if not is_broadcast(t2_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[input_spec, t1_spec, t2_spec],
                        kwargs={"value": value, "out": 2},
                        output_spec=None,
                        comparison_target=2,
                        tolerance=tol,
                        description="addcdiv - INPLACE(c)",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    """AddCdiv operator test with simplified implementation"""

    def __init__(self):
        super().__init__("AddCdiv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.addcdiv(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.addcdiv(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
