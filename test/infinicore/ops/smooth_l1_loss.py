import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, target_shape, input_strides_or_None, beta_or_None, reduction_or_None)
# infinicore.nn.functional.smooth_l1_loss(input, target, reduction='mean', beta=1.0)

_TEST_CASES_DATA = [
    ((4, 5), (4, 5), None, None, None),
    ((8, 8), (8, 8), (512, 64), 1.0, "sum"),
    ((1, 10), (1, 10), None, 0.5, "mean"),
    ((16, 100), (16, 100), None, 0.1, None),
    ((3, 7), (3, 7), None, None, "none"),
    ((2, 2), (2, 2), None, 1.5, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, tgt_shape, strides, beta, reduction in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)
            tgt = TensorSpec.from_tensor(tgt_shape, None, dtype)
            kwargs = {}
            if beta is not None:
                kwargs["beta"] = beta
            if reduction is not None:
                kwargs["reduction"] = reduction

            test_cases.append(
                TestCase(
                    inputs=[inp, tgt],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="smooth_l1_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """smooth_l1_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("smooth_l1_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.smooth_l1_loss(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.smooth_l1_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
