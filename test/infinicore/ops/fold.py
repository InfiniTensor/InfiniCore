import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (in_shape, in_strides_or_None, output_size, kernel_size, dilation, padding, stride)
# For fold/unfold: input is the output of unfold; fold reconstructs image from patches.

_TEST_CASES_DATA = [
    # ((2, 6, 9), None, (4, 4), (2, 2), 1, 0, (2, 2)),
    # ((1, 8, 16), None, (8, 8), (4, 4), 1, 0, (2, 2)),
    # ((3, 4, 12), None, (6, 6), (3, 2), 1, 1, (1, 2)),
    # ((2, 2, 20), None, (10, 2), (2, 2), 1, 0, (2, 1)),
    # ((1, 3, 25), None, (5, 5), (5, 5), 1, 0, (5, 5)),
    # ((2, 5, 18), (90, 18, 1), (9, 2), (3, 2), 1, 0, (2, 1)),
    # 原来对应 ((2,3,8,8), None, (3,3), 1, 0, (1,1))
    # 计算得到 L=6*6=36, channels_for_fold = 3*3*3 = 27
    ((2, 27, 36), None, (8, 8), (3, 3), 1, 0, (1, 1)),
    # 原来对应 ((1,4,10,12), None, (5,3), 1, 1, (2,1))
    # L = 4 * 12 = 48, channels = 4*5*3 = 60
    ((1, 60, 48), None, (10, 12), (5, 3), 1, 1, (2, 1)),
    # 原来对应 ((2,2,16,16), (512,256,16,1), (4,4), 1, 0, (4,4))
    # L = 4 * 4 = 16, channels = 2*4*4 = 32
    ((2, 32, 16), None, (16, 16), (4, 4), 1, 0, (4, 4)),
    # 原来对应 ((3,6,7,9), None, (3,2), 1, 0, (1,1))
    # L = 5 * 8 = 40, channels = 6*3*2 = 36
    ((3, 36, 40), None, (7, 9), (3, 2), 1, 0, (1, 1)),
    # 原来对应 ((1,8,9,11), None, (2,3), 1, 1, (1,2))
    # L = 10 * 6 = 60, channels = 8*2*3 = 48
    ((1, 48, 60), None, (9, 11), (2, 3), 1, 1, (1, 2)),
    # 原来对应 ((2,5,12,6), (360,72,6,1), (3,3), 1, 0, (2,1))
    # L = 5 * 4 = 20, channels = 5*3*3 = 45
    ((2, 45, 20), None, (12, 6), (3, 3), 1, 0, (2, 1)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for (
        in_shape,
        in_strides,
        output_size,
        kernel_size,
        dilation,
        padding,
        stride,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype)
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)

            kwargs = {
                "output_size": output_size,
                "kernel_size": kernel_size,
                "dilation": dilation,
                "padding": padding,
                "stride": stride,
            }

            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Fold - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """Fold operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Fold")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.fold(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.fold(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
