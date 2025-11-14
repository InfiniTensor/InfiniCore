import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner

# Test cases format: (indices_shape, indices_strides_or_None, num_classes_or_None)
_TEST_CASES_DATA = [
    ((5,), None, 10),
    ((3, 4), None, 8),
    ((2, 2), None, None),
    ((1,), None, 3),
    ((6,), None, 6),
    ((4, 3), None, 12),
]

_TOLERANCE_MAP = {infinicore.int64: {"atol": 0, "rtol": 0}}
_TENSOR_DTYPES = [infinicore.int64]


def parse_test_cases():
    test_cases = []
    for shape, strides, num_classes in _TEST_CASES_DATA:
        # ensure indices are non-negative and within [0, num_classes) when provided
        if num_classes is not None:
            high = num_classes
        else:
            high = max(1, shape[0])
        idx_spec = TensorSpec.from_tensor(
            shape,
            strides,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=high,
        )

        kwargs = {}
        if num_classes is not None:
            kwargs["num_classes"] = num_classes

        out_spec = None

        test_cases.append(
            TestCase(
                inputs=[idx_spec],
                kwargs=kwargs,
                output_spec=out_spec,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP.get(infinicore.int64),
                description=f"one_hot - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """OneHot operator test with simplified implementation"""

    def __init__(self):
        super().__init__("OneHot")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.one_hot(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.one_hot(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
