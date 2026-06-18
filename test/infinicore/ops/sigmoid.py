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

# Test cases format: (shape, input_strides, output_strides)
_TEST_CASES_DATA = [
    ((2, 4), None, None),
    ((128, 64), None, None),
    ((2, 4, 8), None, None),
    ((1, 2048), (4096, 1), (4096, 1)),
    ((16, 5632), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []

    for data in _TEST_CASES_DATA:
        shape = data[0]
        input_strides = data[1] if len(data) > 1 else None
        output_strides = data[2] if len(data) > 2 else None

        input_supports_inplace = not is_broadcast(input_strides)
        output_supports_inplace = not is_broadcast(output_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            input_spec = TensorSpec.from_tensor(shape, input_strides, dtype)
            output_spec = TensorSpec.from_tensor(shape, output_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description="Sigmoid - OUT_OF_PLACE",
                )
            )

            if output_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=output_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description="Sigmoid - INPLACE(out)",
                    )
                )

            if input_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs={"out": 0},
                        output_spec=None,
                        comparison_target=0,
                        tolerance=tolerance,
                        description="Sigmoid - INPLACE(input)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("Sigmoid")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, out=None, **kwargs):
        result = torch.sigmoid(input)
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        import infinicore.nn.functional as F

        return F.sigmoid(input, out=out)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
