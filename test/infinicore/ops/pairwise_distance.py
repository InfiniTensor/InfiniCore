import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, p, eps, keepdim, a_strides_or_None, b_strides_or_None)
# infinicore.nn.functional.pairwise_distance(x1, x2, p=2.0, eps=1e-6, keepdim=False)

_TEST_CASES_DATA = [
    ((8, 16), 2.0, 1e-6, False, None, None),
    ((8, 16), 1.0, 1e-6, False, (128, 1), (128, 1)),
    ((2, 3, 4), 2.0, 1e-6, True, None, None),
    ((16, 64), 3.0, 1e-6, False, None, None),
    ((4, 5, 6), 2.0, 1e-6, False, None, None),
    ((3, 4, 5), 2.0, 1e-6, True, (60, 20, 4), None),
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
        shape, p, eps, keepdim = data[0], data[1], data[2], data[3]
        a_strides = data[4] if len(data) > 4 else None
        b_strides = data[5] if len(data) > 5 else None

        a_supports_inplace = not is_broadcast(a_strides)
        b_supports_inplace = not is_broadcast(b_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype)
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype)

            kwargs = {"p": p, "eps": eps, "keepdim": keepdim}

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"pairwise_distance - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """PairwiseDistance operator test with simplified implementation"""

    def __init__(self):
        super().__init__("PairwiseDistance")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.pairwise_distance(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.pairwise_distance(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
