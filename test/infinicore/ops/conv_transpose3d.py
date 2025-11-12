import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases: (in_shape, in_strides_or_None, weight_shape, bias_shape_or_None, stride, padding, output_padding, groups)

_TEST_CASES_DATA = [
    ((1, 2, 8, 8, 8), None, (2, 2, 3, 3, 3), None, (1, 1, 1), (0, 0, 0), (0, 0, 0), 1),
    (
        (2, 3, 7, 9, 5),
        (756, 252, 36, 4, 1),
        (3, 3, 3, 3, 1),
        (3,),
        (2, 2, 1),
        (1, 1, 0),
        (0, 0, 0),
        1,
    ),
    (
        (1, 4, 16, 16, 6),
        None,
        (4, 2, 1, 1, 2),
        None,
        (1, 1, 2),
        (0, 1, 0),
        (0, 0, 0),
        1,
    ),
    (
        (2, 1, 9, 11, 7),
        (693, 77, 77, 7, 1),
        (1, 6, 3, 3, 3),
        None,
        1,
        (1, 0, 1),
        (0, 0, 0),
        1,
    ),
    ((3, 2, 5, 6, 4), None, (2, 2, 2, 2, 2), (2,), (1, 1, 1), (0, 1, 0), (0, 0, 0), 1),
    (
        (2, 6, 10, 9, 8),
        (4320, 720, 72, 8, 1),
        (6, 8, 3, 3, 2),
        None,
        (2, 1, 2),
        (1, 0, 1),
        (1, 0, 1),
        1,
    ),
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
        out_pad,
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
                "output_padding": out_pad,
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
                    description="ConvTranspose3d - OUT_OF_PLACE",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    """ConvTranspose3d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("ConvTranspose3d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.conv_transpose3d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.conv_transpose3d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
