import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, var_present_bool, full_or_None, eps_or_None, input_strides_or_None)
# infinicore.nn.functional.gaussian_nll_loss(input, target, var, full=False, eps=1e-6, reduction='mean')

_TEST_CASES_DATA = [
    ((4, 5), True, None, None, None),
    ((8, 8), True, True, 1e-6, (512, 64)),
    ((1, 10), True, False, 1e-3, None),
    ((16, 100), True, None, None, None),
    ((3, 7), True, True, 1e-5, None),
    ((2, 2), True, False, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, var_present, full, eps, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)
            tgt = TensorSpec.from_tensor(shape, None, dtype)
            var = TensorSpec.from_tensor(shape, None, dtype) if var_present else None

            inputs = [inp, tgt]
            if var is not None:
                inputs.append(var)

            kwargs = {}
            if full is not None:
                kwargs["full"] = full
            if eps is not None:
                kwargs["eps"] = eps

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="gaussian_nll_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """gaussian_nll_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("gaussian_nll_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.gaussian_nll_loss(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.gaussian_nll_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
