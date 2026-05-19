import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)

_TEST_CASES_DATA = [
    ((13, 4), None),
    ((2, 3, 4), None),
    ((16, 5632), None),
]

_TOLERANCE_MAP = {
    infinicore.int32: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [infinicore.int32]


def parse_test_cases():
    test_cases = []
    for shape, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            input_spec = TensorSpec.from_tensor(
                shape,
                strides,
                dtype,
                init_mode=TensorInitializer.RANDINT,
                low=-100,
                high=100,
                name="input",
            )
            other_spec = TensorSpec.from_tensor(
                shape,
                strides,
                dtype,
                init_mode=TensorInitializer.RANDINT,
                low=-100,
                high=100,
                name="other",
            )
            out_spec = TensorSpec.from_tensor(shape, None, dtype, name="out")
            tolerance = _TOLERANCE_MAP[dtype]

            test_cases.append(
                TestCase(
                    inputs=[input_spec, other_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description="lcm - OUT_OF_PLACE",
                )
            )
            test_cases.append(
                TestCase(
                    inputs=[input_spec, other_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=tolerance,
                    description="lcm - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Lcm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.lcm(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.lcm(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
