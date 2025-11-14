import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (list_of_matrix_shapes, list_of_strides_or_None, dtype)

_TEST_CASES_DATA = [
    ([(3, 4), (2, 2)], [None, None], None),
    ([(1, 1), (1, 1), (1, 1)], [None, None, None], None),
    ([(4, 4), (2, 3), (3, 2)], [None, (6, 1), (0, 3)], None),
    ([(8, 8)], [None], None),
    ([(5, 2), (2, 5)], [(10, 1), None], None),
    ([(6, 6), (6, 6), (6, 6)], [None, None, None], None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for shapes, strides_list, _ in _TEST_CASES_DATA:
        # prepare TensorSpec list for inputs
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            input_specs = []
            for s, st in zip(shapes, strides_list):
                input_specs.append(TensorSpec.from_tensor(s, st, dtype))

            test_cases.append(
                TestCase(
                    inputs=input_specs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="block_diag - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """block_diag operator test with simplified implementation"""

    def __init__(self):
        super().__init__("block_diag")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.block_diag(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.block_diag(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
