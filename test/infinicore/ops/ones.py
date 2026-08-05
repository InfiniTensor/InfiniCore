import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase


_TEST_CASES_DATA = [
    (13, 4),
    (8, 16),
    (2, 3, 4),
    (16, 5632),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 0},
    infinicore.float32: {"atol": 0, "rtol": 0},
    infinicore.bfloat16: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for shape in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            ref = TensorSpec.from_tensor(shape, None, dtype)
            test_cases.append(
                TestCase(
                    inputs=[ref],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description="ones",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Ones")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, ref, **kwargs):
        return torch.ones(ref.shape, dtype=ref.dtype, device=ref.device)

    def infinicore_operator(self, ref, **kwargs):
        return infinicore.ones(ref.shape, dtype=ref.dtype, device=ref.device)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
