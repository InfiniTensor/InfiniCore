import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (condition_shape, cond_strides_or_None, x_shape_or_None, y_shape_or_None)
# infinicore.where can be used as where(condition, x, y) or where(condition) returning indices.
_TEST_CASES_DATA = [
    ((3, 4), None, (3, 4), (3, 4)),
    ((5,), None, (5,), (5,)),
    ((2, 2, 3), (12, 6, 2), None, None),
    ((1, 6), None, (1, 6), (1, 6)),
    ((4, 4), None, None, None),
    ((2, 3, 2), None, (2, 3, 2), (2, 3, 2)),
]

_TOLERANCE_MAP = {
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.int64: {"atol": 0, "rtol": 0},
}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for cond_shape, cond_strides, x_shape, y_shape in _TEST_CASES_DATA:
        cond_spec = TensorSpec.from_tensor(cond_shape, cond_strides, infinicore.bool)

        if x_shape is None or y_shape is None:
            # where(condition) -> returns indices
            test_cases.append(
                TestCase(
                    inputs=[cond_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[infinicore.int64],
                    description=f"where - condition_only shape={cond_shape}",
                )
            )
        else:
            x_spec = TensorSpec.from_tensor(x_shape, None, infinicore.float32)
            y_spec = TensorSpec.from_tensor(y_shape, None, infinicore.float32)
            test_cases.append(
                TestCase(
                    inputs=[cond_spec, x_spec, y_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[infinicore.float32],
                    description=f"where - select shape={cond_shape}",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Where operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Where")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.where(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.where(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
