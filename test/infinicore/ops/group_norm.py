import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, num_groups, weight_bias_present_bool, eps_or_None)
# infinicore.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)

_TEST_CASES_DATA = [
    ((4, 8, 16, 16), None, 4, True, None),
    ((2, 6, 8, 8), (768, 128, 1, 1), 3, False, 1e-3),
    ((1, 3, 10, 10), None, 3, True, None),
    ((8, 12, 6, 6), None, 6, True, 1e-4),
    ((6, 4, 7, 7), None, 2, False, None),
    ((3, 2, 9, 9), None, 1, True, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, num_groups, wb_present, eps in _TEST_CASES_DATA:
        C = shape[1]
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)

            # infinicore.nn.functional.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)
            # pass num_groups as positional argument to avoid duplicate kwarg issue
            inputs = [inp, num_groups]
            kwargs = {}
            if wb_present:
                weight = TensorSpec.from_tensor((C,), None, dtype)
                bias = TensorSpec.from_tensor((C,), None, dtype)
                inputs.append(weight)
                inputs.append(bias)
            else:
                # explicit None placeholders for weight and bias
                inputs.append(None)
                inputs.append(None)

            if eps is not None:
                kwargs["eps"] = eps

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="group_norm - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """group_norm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("group_norm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.group_norm(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.group_norm(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
