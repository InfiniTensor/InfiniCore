import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, weight_present_bool, pos_weight_present_bool, reduction_or_None)

_TEST_CASES_DATA = [
    ((4, 5), None, False, False, None),
    ((8, 8), (512, 64), True, False, "sum"),
    ((1, 10), None, False, True, "mean"),
    ((16, 100), None, False, True, "mean"),
    ((3, 7), (21, 7), True, False, None),
    ((2, 2), None, False, False, "none"),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for (
        shape,
        strides,
        weight_present,
        pos_weight_present,
        reduction,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)
            tgt = TensorSpec.from_tensor(shape, None, dtype)

            inputs = [inp, tgt]
            kwargs = {}
            if weight_present:
                weight_spec = TensorSpec.from_tensor(shape, None, dtype)
                inputs.append(weight_spec)
            if pos_weight_present:
                pos_weight_spec = TensorSpec.from_tensor(
                    (shape[1],) if len(shape) > 1 else (shape[0],), None, dtype
                )
                inputs.append(pos_weight_spec)
            if reduction is not None:
                kwargs["reduction"] = reduction

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="binary_cross_entropy_with_logits - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """binary_cross_entropy_with_logits operator test with simplified implementation"""

    def __init__(self):
        super().__init__("binary_cross_entropy_with_logits")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.binary_cross_entropy_with_logits(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
