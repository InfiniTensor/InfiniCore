import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, new_shape)
# reshape can change shape; out parameter is not used in infinicore.reshape API (returns view or tensor)

_TEST_CASES_DATA = [
    ((2, 6), None, (3, 4)),
    ((3, 4), (12, 1), (12,)),
    ((4, 2, 3), None, (2, 12)),
    ((2, 3, 4), (48, 16, 4), (6, 4)),
    ((16, 64), None, (64, 16)),
    ((1, 24), None, (2, 12)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        in_shape, in_strides, new_shape = data[0], data[1], data[2]

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(in_shape, in_strides, dtype)

            # Out-of-place (reshape returns tensor)
            # Following reference pattern: pass new shape as positional arg to infinicore.reshape
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"shape": new_shape},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"reshape - OUT_OF_PLACE",
                )
            )

            # In-place reshape (view-based) is not an explicit API; skip INPLACE cases.
            # Note: infinicore.reshape may return a view; framework will compare returned tensor.

    return test_cases


class OpTest(BaseOperatorTest):
    """Reshape operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Reshape")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.reshape(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.reshape(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
