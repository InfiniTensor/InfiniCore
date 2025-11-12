import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (in_shape, in_strides_or_None, lower_or_None, upper_or_None)

_TEST_CASES_DATA = [
    ((13, 4), None, 0.125, 0.333),
    ((13, 4), (10, 1), 0.1, 0.3),
    ((8, 8, 8), None, 0.05, 0.2),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """rrelu(input, lower=0.125, upper=0.333..., training=False, inplace=False)"""
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        in_strides = data[1] if len(data) > 1 else None
        lower = data[2] if len(data) > 2 else 0.125
        upper = data[3] if len(data) > 3 else 0.333

        input_supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            input_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"lower": lower, "upper": upper}

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"RReLU - OUT_OF_PLACE",
                )
            )

            if input_supports_inplace:
                inplace_kwargs = {"lower": lower, "upper": upper, "inplace": True}
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=inplace_kwargs,
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description=f"RReLU - INPLACE",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """RReLU operator test with simplified implementation"""

    def __init__(self):
        super().__init__("RReLU")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.rrelu(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.rrelu(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
