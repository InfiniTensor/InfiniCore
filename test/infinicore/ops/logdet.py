import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner
from framework.tensor import TensorInitializer

# Test cases format: (matrix_shape, strides_or_None)
# logdet(input) — returns (sign, logabsdet) in PyTorch

_TEST_CASES_DATA = [
    ((1, 1), None),
    ((2, 2), None),
    ((3, 3), (3, 1)),
    ((4, 4), None),
    ((8, 8), (512, 1)),
    ((16, 16), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float32]


def _stable_matrix_spec(shape, strides, dtype):
    rows, cols = shape
    if rows != cols:
        raise ValueError("logdet test matrices must be square")

    matrix = torch.full(shape, 0.01, dtype=torch.float32)
    matrix += torch.eye(rows, dtype=torch.float32) * (rows + 1.0)

    if strides is None:
        return TensorSpec.from_tensor(
            shape,
            None,
            dtype,
            init_mode=TensorInitializer.MANUAL,
            set_tensor=matrix,
        )

    storage_size = 1
    for size, stride in zip(shape, strides):
        storage_size += (size - 1) * abs(stride)

    storage = torch.zeros((storage_size,), dtype=torch.float32)
    for i in range(rows):
        for j in range(cols):
            storage[i * strides[0] + j * strides[1]] = matrix[i, j]

    return TensorSpec.from_tensor(
        shape,
        strides,
        dtype,
        init_mode=TensorInitializer.MANUAL,
        set_tensor=storage,
    )


def parse_test_cases():
    test_cases = []
    for shape, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            spec = _stable_matrix_spec(shape, strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="logdet - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """logdet operator test with simplified implementation"""

    def __init__(self):
        super().__init__("logdet")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.logdet(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.logdet(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
