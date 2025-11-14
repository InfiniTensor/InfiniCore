import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, size_or_scale_factor, mode, align_corners_or_None, input_strides_or_None)

_TEST_CASES_DATA = [
    ((1, 3, 16, 16), (32, 32), "bilinear", True, None),
    ((2, 3, 8, 8), (16, 16), "nearest", None, (384, 128, 16, 1)),
    ((1, 1, 10), 20, "linear", False, None),
    ((2, 3, 6, 6), (12, 12), "area", None, None),
    ((1, 3, 4, 4, 4), (8, 8, 8), "trilinear", True, None),
    ((4, 3, 7, 7), 2.0, "bilinear", False, None),
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
        shape, size_or_scale, mode, align, in_strides = data

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"mode": mode}
            if isinstance(size_or_scale, tuple):
                kwargs["size"] = size_or_scale
            else:
                kwargs["scale_factor"] = size_or_scale
            if align is not None:
                kwargs["align_corners"] = align

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"upsample - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Upsample operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Upsample")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        # Map deprecated upsample to interpolate
        return torch.nn.functional.interpolate(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.interpolate(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
