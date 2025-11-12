import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, target_shape, input_strides_or_None, reduction_or_None)
# infinicore.nn.functional.multilabel_soft_margin_loss(input, target, reduction='mean')

_TEST_CASES_DATA = [
    ((4, 5), (4, 5), None, None),
    ((8, 6), (8, 6), None, "sum"),
    ((1, 3), (1, 3), None, "mean"),
    ((16, 10), (16, 10), None, None),
    ((3, 4), (3, 4), None, "none"),
    ((2, 2), (2, 2), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for inp_shape, tgt_shape, strides, reduction in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(inp_shape, strides, dtype)
            tgt = TensorSpec.from_tensor(tgt_shape, None, dtype)

            kwargs = {}
            if reduction is not None:
                kwargs["reduction"] = reduction

            test_cases.append(
                TestCase(
                    inputs=[inp, tgt],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="multilabel_soft_margin_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """multilabel_soft_margin_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("multilabel_soft_margin_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.multilabel_soft_margin_loss(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.multilabel_soft_margin_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
