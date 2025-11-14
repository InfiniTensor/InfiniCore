import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (theta_shape, out_shape, theta_strides_or_None)

_TEST_CASES_DATA = [
    ((1, 2, 3), (1, 3, 4, 4), None),
    ((2, 2, 3), (2, 3, 8, 8), None),
    ((1, 2, 3), (1, 4, 6, 6), (6, 2, 1)),
    ((4, 2, 3), (4, 3, 5, 5), None),
    ((2, 2, 3), (2, 1, 7, 7), None),
    ((3, 2, 3), (3, 3, 2, 2), None),
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
        theta_shape, out_shape = data[0], data[1]
        theta_strides = data[2] if len(data) > 2 else None

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            theta_spec = TensorSpec.from_tensor(theta_shape, theta_strides, dtype)

            # Out-of-place with align_corners variations
            for align in (True, False):
                kwargs = {"size": out_shape}
                if align is not None:
                    kwargs["align_corners"] = align

                test_cases.append(
                    TestCase(
                        inputs=[theta_spec],
                        kwargs=kwargs,
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="affine_grid - OUT_OF_PLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """AffineGrid operator test with simplified implementation"""

    def __init__(self):
        super().__init__("AffineGrid")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.affine_grid(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.affine_grid(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
