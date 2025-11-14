import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (anchor_shape, positive_shape, negative_shape, strides_or_None, margin_or_None, p_or_None, eps_or_None, swap_or_None)
# infinicore.nn.functional.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-6, swap=False, reduction='mean')

_TEST_CASES_DATA = [
    ((4, 3), (4, 3), (4, 3), None, None, None, None, None),
    ((8, 5), (8, 5), (8, 5), (40, 5), 1.0, 2, 1e-6, False),
    ((1, 10), (1, 10), (1, 10), None, 0.5, 1, 1e-3, None),
    ((16, 12), (16, 12), (16, 12), None, 2.0, 2, None, True),
    ((3, 7), (3, 7), (3, 7), None, None, None, None, None),
    ((2, 4), (2, 4), (2, 4), None, None, None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for a_shape, p_shape, n_shape, strides, margin, p, eps, swap in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a = TensorSpec.from_tensor(a_shape, strides, dtype)
            pos = TensorSpec.from_tensor(p_shape, strides, dtype)
            neg = TensorSpec.from_tensor(n_shape, strides, dtype)

            kwargs = {}
            if margin is not None:
                kwargs["margin"] = margin
            if p is not None:
                kwargs["p"] = p
            if eps is not None:
                kwargs["eps"] = eps
            if swap is not None:
                kwargs["swap"] = swap

            test_cases.append(
                TestCase(
                    inputs=[a, pos, neg],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="triplet_margin_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """triplet_margin_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("triplet_margin_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.triplet_margin_loss(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.triplet_margin_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
