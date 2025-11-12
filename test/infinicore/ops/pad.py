import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, pad_tuple, mode, value_or_None, input_strides_or_None)
# infinicore.nn.functional.pad(input, pad, mode='constant', value=0)

_TEST_CASES_DATA = [
    ((1, 3, 4, 4), (1, 1, 2, 2), "constant", 0.0, None),
    ((2, 3, 8, 8), (2, 2, 2, 2), "reflect", None, (384, 128, 16, 1)),
    ((1, 1, 10), (3, 3), "replicate", None, None),
    ((2, 3, 6, 6), (1, 0, 1, 0), "constant", 1.5, None),
    ((3, 4, 5), (2, 2, 1, 1), "circular", None, None),
    ((4, 5), (1, 2), "constant", -1.0, None),
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
        shape, pad_t, mode, value, in_strides = data

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"pad": pad_t, "mode": mode}
            if value is not None:
                kwargs["value"] = value

            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"pad - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Pad operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Pad")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.pad(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.pad(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
