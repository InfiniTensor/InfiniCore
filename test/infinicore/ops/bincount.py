import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, weights_present_bool, minlength)
_TEST_CASES_DATA = [
    ((10,), None, False, 0),
    ((6,), None, True, 0),
    ((8,), None, False, 5),
    ((12,), None, True, 3),
    ((1,), None, False, 0),
    ((20,), None, True, 0),
]

_TOLERANCE_MAP = {infinicore.int64: {"atol": 0, "rtol": 0}}

_TENSOR_DTYPES = [infinicore.int64]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, strides, weights_present, minlength = data

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})

            # bincount requires 1-D non-negative integer inputs; use RANDINT with low=0
            high = max(1, shape[0])
            input_spec = TensorSpec.from_tensor(
                shape,
                strides,
                dtype,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=high,
            )
            if weights_present:
                weights_spec = TensorSpec.from_tensor(shape, None, infinicore.float32)
            else:
                weights_spec = None

            kwargs = (
                {"minlength": minlength}
                if minlength is not None and minlength != 0
                else {}
            )

            inputs = [input_spec] if not weights_present else [input_spec, weights_spec]

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="bincount - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """Bincount operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Bincount")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.bincount(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.bincount(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
