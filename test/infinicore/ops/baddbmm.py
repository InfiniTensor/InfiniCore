import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (input_shape, batch1_shape, batch2_shape, input_strides_or_None, batch1_strides_or_None, batch2_strides_or_None, beta_or_None, alpha_or_None)

_TEST_CASES_DATA = [
    ((3, 5), (2, 3, 4), (2, 4, 5), None, None, None, None, None),
    ((8, 8), (4, 8, 8), (4, 8, 8), None, None, None, 0.5, 2.0),
    ((5, 7), (2, 5, 6), (2, 6, 7), (30, 1), (0, 5, 1), None, None, None),
    ((16, 16), (2, 16, 16), (2, 16, 16), None, None, (512, 1, 1), 1.0, None),
    ((1, 1), (1, 1, 1), (1, 1, 1), None, None, None, None, None),
    ((6, 8), (3, 6, 7), (3, 7, 8), None, None, None, None, 0.2),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for data in _TEST_CASES_DATA:
        in_shape, b1_shape, b2_shape = data[0], data[1], data[2]
        in_strides = data[3]
        b1_strides = data[4]
        b2_strides = data[5]
        beta = data[6]
        alpha = data[7]

        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)
            out_spec = TensorSpec.from_tensor(
                (b1_shape[0], in_shape[0], b2_shape[2]), None, dtype
            )
            b1_spec = TensorSpec.from_tensor(b1_shape, b1_strides, dtype)
            b2_spec = TensorSpec.from_tensor(b2_shape, b2_strides, dtype)

            kwargs = {}
            if beta is not None:
                kwargs["beta"] = beta
            if alpha is not None:
                kwargs["alpha"] = alpha

            test_cases.append(
                TestCase(
                    inputs=[in_spec, b1_spec, b2_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="baddbmm - OUT_OF_PLACE",
                )
            )

            if out_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[in_spec, b1_spec, b2_spec],
                        kwargs=kwargs,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="baddbmm - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """baddbmm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("baddbmm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.baddbmm(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.baddbmm(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
