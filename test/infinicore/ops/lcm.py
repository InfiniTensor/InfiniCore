import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
    is_broadcast,
)


_TEST_CASES_DATA = [
    ((13, 4), None, None, None),
    ((13, 4), (10, 1), (10, 1), None),
    ((13, 4, 4), None, None, None),
    ((13, 4, 4), (20, 4, 1), (20, 4, 1), None),
    ((16, 5632), None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.int8: {"atol": 0, "rtol": 0},
    infinicore.int16: {"atol": 0, "rtol": 0},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
}

_TENSOR_DTYPES = [
    infinicore.int8,
    infinicore.int16,
    infinicore.int32,
    infinicore.int64,
]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape = data[0]
        a_strides = data[1] if len(data) > 1 else None
        b_strides = data[2] if len(data) > 2 else None
        c_strides = data[3] if len(data) > 3 else None

        c_supports_inplace = not is_broadcast(c_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 0})
            a_spec = TensorSpec.from_tensor(shape, a_strides, dtype, name="a")
            b_spec = TensorSpec.from_tensor(shape, b_strides, dtype, name="b")
            c_spec = TensorSpec.from_tensor(shape, c_strides, dtype, name="c")

            test_cases.append(
                TestCase(
                    inputs=[a_spec, b_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="lcm - OUT_OF_PLACE",
                )
            )

            if c_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[a_spec, b_spec],
                        kwargs=None,
                        output_spec=c_spec,
                        comparison_target="out",
                        tolerance=tol,
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
