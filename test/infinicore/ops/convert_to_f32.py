import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# (in_shape, in_strides_or_None)

_TEST_CASES_DATA = [
    ((2, 3), None),
    ((1, 4, 8), (32, 8, 1)),
    ((3, 2, 5, 7), None),
    ((2, 1, 16), None),
    ((1, 8, 9, 11), (792, 99, 11, 1)),
    ((2, 6, 10), None),
]

_TOLERANCE = {"atol": 1e-5, "rtol": 1e-4}
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    cases = []
    for shape, strides in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            in_spec = TensorSpec.from_tensor(shape, strides, dtype)

            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE,
                    description="convert_to_f32_out_of_place",
                )
            )

            out_spec = TensorSpec.from_tensor(shape, None, infinicore.float32)
            cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={},
                    output_spec=out_spec,
                    comparison_target="out",
                    tolerance=_TOLERANCE,
                    description="convert_to_f32_explicit_out",
                )
            )

            if dtype == infinicore.float32 and not is_broadcast(in_spec.strides):
                cases.append(
                    TestCase(
                        inputs=[in_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=_TOLERANCE,
                        description="convert_to_f32_inplace_input0",
                    )
                )

    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("convert_to_f32")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        x = args[0]
        if kwargs.get("out") is not None:
            kwargs["out"].copy_(x.float())
            return kwargs["out"]
        return x.float()

    def infinicore_operator(self, *args, **kwargs):
        if kwargs.get("out") is not None:
            return infinicore.convert_to_f32(args[0], out=kwargs["out"])
        return infinicore.convert_to_f32(args[0])


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
