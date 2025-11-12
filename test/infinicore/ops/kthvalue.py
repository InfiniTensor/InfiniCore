import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides, k, dim, keepdim)
_TEST_CASES_DATA = [
    ((6, 8), None, 1, 1, False),
    ((8, 4), (16, 1), 2, 0, True),
    ((5, 5), None, 3, -1, False),
    ((3, 7), (14, 1), 2, 1, True),
    ((10, 3), None, 1, 1, False),
    ((2, 16), (32, 1), 5, 1, False),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides, k, dim, keepdim = data

        out_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            # kthvalue returns (values, indices). We'll request out-of-place and explicit out for values.
            values_spec = TensorSpec.from_tensor(shape, None, dtype)
            indices_spec = TensorSpec.from_tensor(shape, None, infinicore.int64)

            kwargs = {"k": k, "dim": dim, "keepdim": keepdim}

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"kthvalue - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """KthValue operator test with simplified implementation"""

    def __init__(self):
        super().__init__("KthValue")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.kthvalue(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.kthvalue(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
