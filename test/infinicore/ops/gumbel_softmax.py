import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, tau_or_None, hard_or_None, dim_or_None)

_TEST_CASES_DATA = [
    ((4, 10), None, 1.0, False, -1),
    ((8, 20), (160, 1), 0.5, False, -1),
    ((2, 5, 6), None, 1.5, True, 2),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    gumbel_softmax: infinicore.nn.functional.gumbel_softmax(input, tau=1, hard=False, eps=1e-10, dim=-1)
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None
        tau = data[2] if len(data) > 2 else 1.0
        hard = data[3] if len(data) > 3 else False
        dim = data[4] if len(data) > 4 else -1

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"tau": tau, "hard": hard, "dim": dim}

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"GumbelSoftmax - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """GumbelSoftmax operator test with simplified implementation"""

    def __init__(self):
        super().__init__("GumbelSoftmax")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.gumbel_softmax(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.gumbel_softmax(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
