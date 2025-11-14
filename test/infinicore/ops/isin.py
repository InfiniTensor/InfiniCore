import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, input_strides_or_None, test_elements_strides_or_None, extra_or_None)
# isin checks membership of each element in provided test_elements (tensor or list)

_TEST_CASES_DATA = [
    ((8, 8), None, None, None),
    ((8, 8), (16, 1), None, None),
    ((8, 8), None, (1,), None),
    ((2, 3, 4), None, None, None),
    ((1, 8), None, None, None),
    ((16, 64), (128, 1), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.int32: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.int32, infinicore.float32, infinicore.float16]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, in_strides, elements_strides, _ = data[0], data[1], data[2], data[3]

        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            # Build "test elements" as a tensor of small set of values (same dtype)
            elements_spec = TensorSpec.from_tensor(
                (4,), elements_strides if elements_strides else None, dtype
            )

            # Out-of-place
            test_cases.append(
                TestCase(
                    inputs=[input_spec, elements_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="IsIn - OUT_OF_PLACE",
                )
            )

            # explicit out
            out_spec = TensorSpec.from_tensor(shape, None, infinicore.bool)
            test_cases.append(
                TestCase(
                    inputs=[input_spec, elements_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tol,
                    description="IsIn - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """IsIn operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IsIn")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.isin(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.isin(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
