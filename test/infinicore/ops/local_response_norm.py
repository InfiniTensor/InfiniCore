import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, size, alpha_or_None, beta_or_None, k_or_None)
# infinicore.nn.functional.local_response_norm(input, size, alpha=1e-4, beta=0.75, k=1.0)

_TEST_CASES_DATA = [
    ((4, 3, 8, 8), None, 5, None, None, None),
    ((2, 6, 4, 4), (384, 96, 1, 1), 3, 1e-4, 0.75, 1.0),
    ((1, 3, 16, 16), None, 7, None, None, None),
    ((8, 5, 2, 2), None, 1, 1e-3, 0.5, 0.0),
    ((6, 4, 7, 7), None, 9, None, None, None),
    ((3, 2, 9, 9), None, 4, 1e-5, 0.9, 2.0),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}

_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, size, alpha, beta, k in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)

            kwargs = {"size": size}
            if alpha is not None:
                kwargs["alpha"] = alpha
            if beta is not None:
                kwargs["beta"] = beta
            if k is not None:
                kwargs["k"] = k

            test_cases.append(
                TestCase(
                    inputs=[inp],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="local_response_norm - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """local_response_norm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("local_response_norm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.local_response_norm(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.local_response_norm(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
