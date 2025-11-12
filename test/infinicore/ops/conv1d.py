import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases: (in_shape, in_strides_or_None, weight_shape, bias_shape_or_None, stride, padding, dilation, groups)

_TEST_CASES_DATA = [
    ((2, 4, 16), None, (8, 4, 3), None, 1, 0, 1, 1),
    ((1, 6, 15), (90, 15, 1), (4, 6, 5), (4,), 2, 2, 1, 1),
    ((2, 16, 32), None, (8, 8, 1), None, 1, 0, 1, 2),
    ((3, 3, 7), (21, 7, 1), (6, 3, 3), None, 1, 0, 1, 1),
    ((2, 2, 31), None, (4, 2, 4), (4,), 2, 1, 1, 1),
    ((1, 8, 9), (72, 9, 1), (8, 8, 3), None, 1, 1, 2, 1),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    tests = []
    for (
        in_shape,
        in_strides,
        w_shape,
        b_shape,
        stride,
        padding,
        dilation,
        groups,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            weight_spec = TensorSpec.from_tensor(w_shape, None, dtype)
            if b_shape is not None:
                bias_spec = TensorSpec.from_tensor(b_shape, None, dtype)
            else:
                bias_spec = None

            kwargs = {
                "stride": stride,
                "padding": padding,
                "dilation": dilation,
                "groups": groups,
            }
            inputs = [in_spec, weight_spec]
            if bias_spec is not None:
                inputs.append(bias_spec)

            tests.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Conv1d - OUT_OF_PLACE",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    """Conv1d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Conv1d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.conv1d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.conv1d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
