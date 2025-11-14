import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, upscale_factor, input_strides_or_None)
# infinicore.nn.functional.pixel_shuffle(input, upscale_factor)

_TEST_CASES_DATA = [
    ((1, 4, 8, 8), 2, None),
    ((2, 9, 4, 4), 3, (288, 144, 36, 9)),
    ((1, 16, 4, 4), 4, None),
    ((3, 8, 6, 6), 2, None),
    ((2, 12, 3, 3), 2, None),
    ((4, 27, 2, 2), 3, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, factor = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"upscale_factor": factor}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"pixel_shuffle - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """PixelShuffle operator test with simplified implementation"""

    def __init__(self):
        super().__init__("PixelShuffle")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.pixel_shuffle(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.pixel_shuffle(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
