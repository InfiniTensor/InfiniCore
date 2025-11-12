import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, p_or_None, dim_or_None, eps_or_None)
# infinicore.nn.functional.normalize(input, p=2.0, dim=1, eps=1e-12)

_TEST_CASES_DATA = [
    ((4, 3), None, None, None, None),
    ((8, 5), (40, 5), 1.0, 1, 1e-12),
    ((1, 10), None, 2.0, 1, 1e-6),
    ((16, 100), None, float("inf"), 1, None),
    ((3, 7), None, 0.5, 1, None),
    ((2, 2), None, None, 0, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, p, dim, eps in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {}
            if p is not None:
                kwargs["p"] = p
            if dim is not None:
                kwargs["dim"] = dim
            if eps is not None:
                kwargs["eps"] = eps

            test_cases.append(
                TestCase(
                    inputs=[inp],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="normalize - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """normalize operator test with simplified implementation"""

    def __init__(self):
        super().__init__("normalize")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.normalize(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.normalize(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
