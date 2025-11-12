import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, reduction_or_None, log_target_bool_or_None)
# infinicore.nn.functional.kl_div(input, target, reduction='mean', log_target=False)

_TEST_CASES_DATA = [
    ((4, 5), None, "batchmean", None),
    ((8, 8), (512, 64), "sum", False),
    ((1, 10), None, "batchmean", True),
    ((16, 100), None, "batchmean", False),
    ((3, 7), None, "batchmean", None),
    ((2, 2), None, "sum", None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, reduction, log_target in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a = TensorSpec.from_tensor(shape, strides, dtype)
            b = TensorSpec.from_tensor(shape, None, dtype)

            kwargs = {}
            if reduction is not None:
                kwargs["reduction"] = reduction
            if log_target is not None:
                kwargs["log_target"] = log_target

            test_cases.append(
                TestCase(
                    inputs=[a, b],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="kl_div - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """kl_div operator test with simplified implementation"""

    def __init__(self):
        super().__init__("kl_div")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.kl_div(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.kl_div(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
