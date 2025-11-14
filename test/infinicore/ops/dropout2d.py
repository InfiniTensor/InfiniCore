import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, p, training, in_strides_or_None)
# infinicore.nn.functional.dropout2d(input, p=0.5, training=True)

_TEST_CASES_DATA = [
    ((8, 16, 8, 8), 0.1, True, None),
    ((8, 16, 8, 8), 0.2, False, (1024, 64, 8, 1)),
    ((2, 3, 4, 8), 0.5, True, None),
    ((16, 64, 4, 4), 0.3, True, None),
    ((4, 5, 6, 8), 0.5, False, None),
    ((3, 4, 5, 5), 0.4, True, (60, 20, 4, 1)),
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
        shape, p, training = data[0], data[1], data[2]
        in_strides = data[3] if len(data) > 3 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-2, "rtol": 1e-2})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"p": p, "training": training}

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"dropout2d - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Dropout2d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Dropout2d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.dropout2d(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.dropout2d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
